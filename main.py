"""
╔══════════════════════════════════════════════════════════════════╗
║          SILA — Medical AI Assistant  v14.0                     ║
║          Railway-Hardened · 384-dim · Local Embeddings Only     ║
║                                                                  ║
║  Stack : FastAPI + Pinecone v3 + Gemini (google-genai SDK)      ║
║  Mode  : Strict RAG (local 384-dim) · Social chat               ║
║  v14.0 changes vs v13.2:                                        ║
║  • EMBEDDING_BACKEND forced to 'local' only (384-dim)           ║
║  • Removed all Gemini embedding paths + dead code               ║
║  • SentenceTransformer lazy-load hardened (CPU-only, no torch)  ║
║  • Pinecone expected_dim hardcoded to 384                        ║
║  • Removed verbose vector/tensor log dumps                      ║
║  • GeminiService cache changed to bounded deque-based LRU       ║
║  • IntentClassifier keyword fallback made primary + fast        ║
║  • Healthcheck always fast (<1s, no SDK touch)                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ── Unbuffered output first — Railway log visibility ──────────────
import os

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── Standard library ──────────────────────────────────────────────
import asyncio
import base64
import collections
import functools
import hashlib
import json
import logging
import re
import sys
import threading
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

sys.setrecursionlimit(1000)

# ── Lightweight third-party ───────────────────────────────────────
from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, model_validator

# Heavy SDKs (google.genai, pinecone, sentence_transformers) are
# lazy-imported inside their respective init functions so the process
# starts and passes Railway healthcheck before any SDK init happens.

load_dotenv()

# ──────────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
PINECONE_API_KEY  = os.getenv("PINECONE_API_KEY", "")
INDEX_NAME        = os.getenv("PINECONE_INDEX", "medical-index-arabicdata")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "")

MIN_CONFIDENCE    = float(os.getenv("SCORE_THRESHOLD", "0.45"))
MAX_QUERY_LENGTH  = int(os.getenv("MAX_QUERY_LENGTH", "500"))
MAX_IMAGE_MB      = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
MAX_IMAGE_BYTES   = MAX_IMAGE_MB * 1024 * 1024

TOP_K             = int(os.getenv("TOP_K", "7"))
MAX_RETRIES       = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY       = float(os.getenv("RETRY_DELAY", "1.5"))

# ── Embedding: local only, 384-dim ───────────────────────────────
# EMBEDDING_BACKEND is read but only 'local' is supported.
EMBED_MODEL       = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_DIM         = 384  # all-MiniLM-L6-v2 fixed output dimension

MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "20"))
EXTERNAL_CALL_TIMEOUT   = int(os.getenv("EXTERNAL_CALL_TIMEOUT", "25"))

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set.")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is not set.")

# ──────────────────────────────────────────────────────────────────
# Logging — compact, no vector dumps
# ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("sila")

# ──────────────────────────────────────────────────────────────────
# Concurrency gate
# ──────────────────────────────────────────────────────────────────
_request_semaphore: Optional[asyncio.Semaphore] = None

# ──────────────────────────────────────────────────────────────────
# Rate limiter
# ──────────────────────────────────────────────────────────────────
_rate_limit_store: Dict[str, list] = defaultdict(list)
_rate_limit_lock  = threading.Lock()
_RATE_LIMIT_REQUESTS = 10
_RATE_LIMIT_WINDOW   = 60  # seconds


def _check_rate_limit(ip: str) -> bool:
    now = time.time()
    with _rate_limit_lock:
        _rate_limit_store[ip] = [
            ts for ts in _rate_limit_store[ip] if now - ts < _RATE_LIMIT_WINDOW
        ]
        if len(_rate_limit_store[ip]) < _RATE_LIMIT_REQUESTS:
            _rate_limit_store[ip].append(now)
            return True
    return False


# ──────────────────────────────────────────────────────────────────
# Lazy SDK: Pinecone
# ──────────────────────────────────────────────────────────────────
_pinecone_index = None
_pinecone_lock  = threading.Lock()


def _init_pinecone():
    global _pinecone_index
    if _pinecone_index is None:
        with _pinecone_lock:
            if _pinecone_index is None:
                log.info("Initialising Pinecone…")
                from pinecone import Pinecone as _PC
                _pinecone_index = _PC(api_key=PINECONE_API_KEY).Index(INDEX_NAME)
                log.info("Pinecone ready.")
    return _pinecone_index


async def get_index():
    if _pinecone_index is None:
        await asyncio.to_thread(_init_pinecone)
    return _pinecone_index


# ──────────────────────────────────────────────────────────────────
# Lazy SDK: Gemini
# ──────────────────────────────────────────────────────────────────
_gemini_client = None
_gemini_lock   = threading.Lock()


def _init_gemini():
    global _gemini_client
    if _gemini_client is None:
        with _gemini_lock:
            if _gemini_client is None:
                log.info("Initialising Gemini client…")
                from google import genai as _genai
                _gemini_client = _genai.Client(api_key=GEMINI_API_KEY)
                log.info("Gemini client ready.")
    return _gemini_client


def _get_gemini_sync():
    return _init_gemini()


def _gemini_types():
    from google.genai import types
    return types


# ──────────────────────────────────────────────────────────────────
# Lazy SDK: SentenceTransformer (CPU-only, 384-dim)
# ──────────────────────────────────────────────────────────────────
_st_model       = None
_st_lock        = threading.Lock()
_st_load_error: Optional[str] = None


def _load_and_encode_sync(text: str) -> List[float]:
    global _st_model, _st_load_error
    if _st_model is None:
        with _st_lock:
            if _st_model is None:
                if _st_load_error:
                    raise RuntimeError(f"Embedder previously failed: {_st_load_error}")
                try:
                    log.info(f"Lazy-loading SentenceTransformer: {EMBED_MODEL}")
                    t0 = time.perf_counter()
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer(EMBED_MODEL, device="cpu")
                    # Disable any torch gradient / autocast overhead
                    try:
                        import torch as _torch
                        _torch.set_num_threads(1)
                        _torch.set_num_interop_threads(1)
                        _torch.set_grad_enabled(False)
                    except ImportError:
                        pass
                    log.info(f"SentenceTransformer ready in {time.perf_counter()-t0:.2f}s")
                    _st_model = model
                except Exception as exc:
                    _st_load_error = str(exc)
                    raise

    vec: List[float] = _st_model.encode(
        text,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).tolist()

    if len(vec) != EMBED_DIM:
        raise RuntimeError(
            f"Embedding dimension mismatch: got {len(vec)}, expected {EMBED_DIM}."
        )
    return vec


async def _encode_async(text: str) -> List[float]:
    try:
        return await asyncio.to_thread(_load_and_encode_sync, text)
    except Exception as exc:
        raise RuntimeError(f"Embedding unavailable: {exc}") from exc


# ──────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────
GEMINI_TEXT_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

GEMINI_VISION_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]

ALLOWED_IMAGE_TYPES = frozenset({
    "image/jpeg", "image/png", "image/webp", "image/heic", "image/heif",
})

MEDICAL_DISCLAIMER = (
    "⚠️ تنبيه: هذه المعلومات للتوجيه العام فقط "
    "ولا تُغني عن استشارة طبيب متخصص."
)

MEDICAL_KEYWORDS: frozenset = frozenset({
    "ألم","وجع","مرض","دواء","طبيب","مستشفى","أعراض","علاج",
    "صداع","حمى","سعال","ضغط","سكر","قلب","كلى","معدة",
    "عظام","جلد","عين","أذن","أنف","رئة","كبد","دم",
    "تعب","إرهاق","دوار","غثيان","إسهال","إمساك","حرقة",
    "الم","عندي","عندى","اشعر","احس","اعاني","يؤلم",
    "بوجعني","بتوجعني","حاسس","حاسه","حبوب","طفح","حكة",
    "عملية","جراحة","منظار","تحليل","أشعة","نتيجة","تقرير",
    "pain","ache","fever","cough","headache","nausea","dizzy",
    "vomit","diarrhea","symptom","disease","doctor","hospital",
    "medicine","drug","blood","heart","lung","kidney","liver",
    "diabetes","pressure","infection","allergy","rash","swelling",
    "fatigue","tired","breathe","chest","stomach","throat",
    "surgery","scan","test","result","report","prescription",
    "برد","انفلونزا","رشح","زكام","كحة","بلغم","حرارة",
    "cold","flu","runny","nose","sneeze","congestion",
})

# ──────────────────────────────────────────────────────────────────
# Domain Models
# ──────────────────────────────────────────────────────────────────

@dataclass
class KnowledgeMatch:
    question: str
    answer: str
    confidence: float
    category: Optional[str] = None

    @property
    def is_reliable(self) -> bool:
        return self.confidence >= MIN_CONFIDENCE


@dataclass
class QueryContext:
    raw_query: str
    language: str
    matches: List[KnowledgeMatch] = field(default_factory=list)

    @property
    def has_reliable_matches(self) -> bool:
        return any(m.is_reliable for m in self.matches)

    @property
    def best_confidence(self) -> float:
        return self.matches[0].confidence if self.matches else 0.0


# ──────────────────────────────────────────────────────────────────
# Pydantic Schemas
# ──────────────────────────────────────────────────────────────────

class MessageDto(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    role: str
    content: str


class AskRequest(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    question: Optional[str] = None
    text: Optional[str] = None
    history: Optional[List[MessageDto]] = None

    @property
    def query(self) -> str:
        return (self.question or self.text or "").strip()

    @model_validator(mode="after")
    def validate_query(self) -> "AskRequest":
        q = self.query
        if not q:
            raise ValueError("Request must include a non-empty 'question' or 'text' field.")
        if len(q) > MAX_QUERY_LENGTH:
            raise ValueError(f"Query exceeds {MAX_QUERY_LENGTH} characters.")
        return self


class MatchResult(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    question: str
    answer: str
    confidence: float
    category: Optional[str] = None


class AskResponse(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    query: str
    reply: str
    model_used: str
    matches: List[MatchResult]
    is_medical: bool
    found_in_database: bool
    low_confidence: bool
    language: str
    disclaimer: str


class ImageAnalysisResponse(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    status: str
    analysis: Optional[str] = None
    model_used: Optional[str] = None
    disclaimer: str = MEDICAL_DISCLAIMER


# ──────────────────────────────────────────────────────────────────
# Language Detection
# ──────────────────────────────────────────────────────────────────

class LanguageDetector:
    @staticmethod
    def detect(text: str) -> str:
        if not text:
            return "ar"
        arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
        return "ar" if arabic_chars / max(len(text.strip()), 1) > 0.25 else "en"


# ──────────────────────────────────────────────────────────────────
# Intent Classifier — keyword-first, Gemini as optional enhancer
# ──────────────────────────────────────────────────────────────────

class IntentClassifier:
    """
    Keyword check runs first (zero cost).
    Gemini is called only when keywords give no signal, to reduce latency/cost.
    """

    _PROMPT = (
        "Classify this message into exactly one category.\n\n"
        "Categories:\n"
        "- social  : greetings, thanks, casual conversation, non-medical\n"
        "- medical : symptoms, diseases, medications, body, pain, health\n\n"
        "Rules:\n"
        "- Reply with ONE word only: social OR medical\n"
        "- No punctuation, no explanation\n\n"
        "Message: {query}"
    )

    @classmethod
    def _classify_sync(cls, query: str) -> str:
        q_lower = query.lower()
        # Fast keyword path — if any medical keyword matches, skip Gemini call
        if any(kw in q_lower for kw in MEDICAL_KEYWORDS):
            return "medical"

        # Only call Gemini for ambiguous messages
        try:
            types = _gemini_types()
            resp = _get_gemini_sync().models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=cls._PROMPT.format(query=query),
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=5),
            )
            result = resp.text.strip().lower()
            if "medical" in result:
                return "medical"
        except Exception as exc:
            log.warning(f"IntentClassifier Gemini call failed: {exc}")

        return "social"

    @classmethod
    async def classify(cls, query: str) -> str:
        return await asyncio.to_thread(cls._classify_sync, query)


# ──────────────────────────────────────────────────────────────────
# Knowledge Base Service
# ──────────────────────────────────────────────────────────────────

class KnowledgeBaseService:

    async def search(self, query: str, top_k: int = TOP_K) -> List[KnowledgeMatch]:
        log.info(f"[KB] Search: '{query[:80]}'")

        try:
            vector = await _encode_async(query)
        except Exception as exc:
            log.error(f"[KB] Embedding failed: {exc}")
            return []

        index = await get_index()

        last_exc: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                _query_fn = functools.partial(
                    index.query,
                    vector=vector,
                    top_k=top_k,
                    include_metadata=True,
                    namespace=PINECONE_NAMESPACE or "",
                )
                results = await asyncio.to_thread(_query_fn)
                raw_scores = [round(float(m.score), 4) for m in results.matches if m.score is not None]
                log.info(
                    f"[KB] Pinecone scores: {raw_scores} "
                    f"| threshold={MIN_CONFIDENCE} | n={len(results.matches)}"
                )
                return self._parse_matches(results)
            except Exception as exc:
                last_exc = exc
                log.warning(f"[KB] Pinecone attempt {attempt}/{MAX_RETRIES} failed: {exc}")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY * attempt)

        log.error(f"[KB] All retries exhausted: {last_exc}")
        return []

    @staticmethod
    def _parse_matches(results: Any) -> List[KnowledgeMatch]:
        matches = []
        filtered = 0
        for m in results.matches:
            score = float(m.score) if m.score is not None else 0.0
            if score < MIN_CONFIDENCE:
                filtered += 1
                continue
            meta = m.metadata or {}
            matches.append(KnowledgeMatch(
                question=meta.get("question", ""),
                answer=meta.get("answer", ""),
                confidence=round(score, 4),
                category=meta.get("category", "General"),
            ))
        log.info(f"[KB] Kept={len(matches)} filtered={filtered}")
        return matches


# ──────────────────────────────────────────────────────────────────
# Prompt Builder
# ──────────────────────────────────────────────────────────────────

class PromptBuilder:
    _SYSTEM_AR = (
        "أنت 'سيلا'، مساعد طبي ذكي وموثوق.\n"
        "أسلوبك: دافئ واحترافي، كأنك طبيب خبير يشرح لمريضه بصدق واهتمام.\n\n"
        "قواعد صارمة لا استثناء فيها:\n"
        "١. استخدم فقط المعلومات المقدمة في قاعدة المعرفة أدناه.\n"
        "٢. لا تستخدم أي معرفة خارجية أو افتراضات شخصية تحت أي ظرف.\n"
        "٣. إذا كانت المعلومات غير كافية، قل بوضوح: "
        "'معلوماتي محدودة في هذه الحالة، أنصح بمراجعة طبيب متخصص.'\n"
        "٤. لا تُقدم تشخيصاً نهائياً أبداً — قدّم احتمالات فقط.\n"
        "٥. اذكر علامات الخطر التي تستدعي التدخل العاجل إن وُجدت.\n"
        "٦. اختم دائماً بالتوصية بمراجعة طبيب متخصص.\n"
        "٧. لغة الإجابة: عربية واضحة ومفهومة.\n"
        "٨. لا تُجيب إلا بناءً على السياق المسترجع من قاعدة المعرفة."
    )

    _SYSTEM_EN = (
        "You are 'Sila', a trusted and empathetic medical AI assistant.\n"
        "Tone: warm, calm, and professionally precise.\n\n"
        "Strict rules — no exceptions:\n"
        "1. Use ONLY the information provided in the knowledge base context below.\n"
        "2. NEVER use external knowledge or personal assumptions.\n"
        "3. If the data is insufficient, clearly state: "
        "'My knowledge is limited on this. Please consult a specialist.'\n"
        "4. NEVER provide a definitive diagnosis — suggest possibilities only.\n"
        "5. Flag any warning signs that require urgent care.\n"
        "6. Always close by recommending a specialist consultation.\n"
        "7. Only answer based on the retrieved context from the knowledge base."
    )

    _STRUCTURE_AR = (
        "رتّب إجابتك بهذا الشكل:\n\n"
        "🔍 الأسباب المحتملة:\n"
        "→ اذكر الأسباب الأكثر احتمالاً بناءً على السياق فقط\n\n"
        "💊 التوصيات والخطوات العملية:\n"
        "→ ما يمكن للمريض فعله الآن\n\n"
        "🏥 التخصص الطبي المناسب للمراجعة:\n"
        "→ اذكر التخصص المناسب\n\n"
        "⚠️ علامات الخطر التي تستدعي الطوارئ فوراً:\n"
        "→ اذكرها إن وُجدت، أو اكتب 'لا توجد علامات خطر واضحة في السياق المقدم'"
    )

    _STRUCTURE_EN = (
        "Structure your response as follows:\n\n"
        "🔍 Possible Causes:\n"
        "→ Based strictly on the provided context\n\n"
        "💊 Recommendations & Next Steps:\n"
        "→ Practical actions the patient can take\n\n"
        "🏥 Recommended Medical Specialty:\n"
        "→ Which specialist to consult\n\n"
        "⚠️ Warning Signs Requiring Immediate Emergency Care:\n"
        "→ List if present, or state "
        "'No critical warning signs identified in the provided context'"
    )

    _NO_DATA_AR = (
        "لا تتوفر لديّ معلومات كافية في قاعدة بياناتي للإجابة على هذا الاستفسار بدقة. "
        "أنصح بمراجعة طبيب متخصص للحصول على تقييم دقيق وآمن. "
        "صحتك أهم من أي إجابة سريعة. 🏥"
    )

    _NO_DATA_EN = (
        "I don't have sufficient information in my knowledge base to answer this accurately. "
        "I strongly recommend consulting a specialist for a proper and safe evaluation. "
        "Your health deserves accurate, professional care. 🏥"
    )

    def build(self, ctx: QueryContext) -> str:
        lang = ctx.language
        system    = self._SYSTEM_AR    if lang == "ar" else self._SYSTEM_EN
        structure = self._STRUCTURE_AR if lang == "ar" else self._STRUCTURE_EN
        sep = "━" * 50

        context_parts = []
        for i, m in enumerate(ctx.matches, start=1):
            reliability = (
                ("✅ موثوق" if m.is_reliable else "⚠️ ثقة منخفضة")
                if lang == "ar"
                else ("✅ Reliable" if m.is_reliable else "⚠️ Low confidence")
            )
            context_parts.append(
                f"[{i}] {reliability} — Score: {m.confidence:.0%}\n"
                f"[Specialty: {m.category or 'General'}]\n"
                f"Q: {m.question}\n"
                f"A: {m.answer}"
            )

        context_block   = f"\n\n{sep}\n".join(context_parts)
        label_context   = "📋 قاعدة المعرفة الطبية:" if lang == "ar" else "📋 Medical Knowledge Base:"
        label_question  = "🧑‍⚕️ سؤال المريض:"       if lang == "ar" else "🧑‍⚕️ Patient Question:"
        label_answer    = "الإجابة:"                 if lang == "ar" else "Answer:"

        return (
            f"{system}\n\n{sep}\n"
            f"{label_context}\n\n{context_block}\n\n{sep}\n"
            f"{label_question}\n{ctx.raw_query}\n\n"
            f"{structure}\n\n{label_answer}"
        )

    def no_data_response(self, language: str) -> str:
        return self._NO_DATA_AR if language == "ar" else self._NO_DATA_EN


# ──────────────────────────────────────────────────────────────────
# Gemini Service  (threadpool-offloaded, bounded LRU cache)
# ──────────────────────────────────────────────────────────────────

class _BoundedLRU:
    """Thread-safe LRU cache backed by an OrderedDict."""

    def __init__(self, maxsize: int = 200):
        self._cache: collections.OrderedDict = collections.OrderedDict()
        self._max = maxsize
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Tuple[str, str]]:
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def put(self, key: str, value: Tuple[str, str]) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self._max:
                self._cache.popitem(last=False)

    def __len__(self) -> int:
        return len(self._cache)


class GeminiService:

    def __init__(self) -> None:
        self._cache = _BoundedLRU(200)

    @staticmethod
    def _make_config(temperature: float = 0.2, max_tokens: int = 2048):
        types = _gemini_types()
        return types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    # ── Social replies ─────────────────────────────────────────────

    def _reply_social_sync(self, query: str, language: str) -> Tuple[str, str]:
        if language == "en":
            system = (
                "You are 'Sila', a friendly and warm medical AI assistant. "
                "Reply naturally in English. Keep it brief (1-2 sentences). "
                "If not medical, warmly mention your specialty and invite health questions."
            )
        else:
            system = (
                "أنت 'سيلا'، مساعد طبي ذكي وودود.\n"
                "رد بالعربية بشكل طبيعي ودافئ. الرد قصير (جملة أو اتنين بالكثير).\n"
                "لو الموضوع مش طبي، قول بلطف إنك متخصص في الاستشارات الطبية."
            )
        types = _gemini_types()
        for model_name in GEMINI_TEXT_MODELS:
            try:
                resp = _get_gemini_sync().models.generate_content(
                    model=model_name,
                    contents=query,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=0.75,
                        max_output_tokens=200,
                    ),
                )
                return resp.text.strip(), model_name
            except Exception as exc:
                log.warning(f"[Gemini] Social {model_name} failed: {exc}")

        fallback = (
            "Hello! 😊 I'm Sila, your medical AI. How can I help?"
            if language == "en"
            else "أهلاً! 😊 أنا سيلا، مساعدتك الطبية. كيف يمكنني مساعدتك؟"
        )
        return fallback, "fallback"

    async def reply_social(self, query: str, language: str) -> Tuple[str, str]:
        return await asyncio.to_thread(self._reply_social_sync, query, language)

    # ── RAG generation ─────────────────────────────────────────────

    def _generate_sync(self, prompt: str) -> Tuple[str, str]:
        cache_key = hashlib.sha256(prompt.encode()).hexdigest()
        cached = self._cache.get(cache_key)
        if cached:
            log.info("[Gemini] Cache hit.")
            return cached

        for model_name in GEMINI_TEXT_MODELS:
            try:
                log.info(f"[Gemini] RAG generate — model: {model_name}")
                resp = _get_gemini_sync().models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=self._make_config(),
                )
                result = (resp.text.strip(), model_name)
                self._cache.put(cache_key, result)
                return result
            except Exception as exc:
                log.warning(f"[Gemini] RAG {model_name} failed: {exc}")

        log.error("[Gemini] All text models exhausted.")
        return ("عذراً، حدث خطأ مؤقت في معالجة طلبك. يرجى المحاولة مرة أخرى.", "none")

    async def generate(self, prompt: str) -> Tuple[str, str]:
        return await asyncio.to_thread(self._generate_sync, prompt)

    # ── Image analysis ─────────────────────────────────────────────

    def _analyze_image_sync(self, image_bytes: bytes, mime_type: str) -> Tuple[str, str, str]:
        system_prompt = (
            "You are a specialized medical image analysis AI.\n\n"
            "You ONLY analyze medical images. Accepted types:\n"
            "  - Lab results / blood tests\n"
            "  - Prescriptions / medical reports\n"
            "  - X-rays, MRI, CT scans\n"
            "  - ECG / EKG strips\n"
            "  - Pathology slides\n"
            "  - Ultrasound images\n\n"
            "CRITICAL RULES:\n"
            "1. If NOT medical → respond with JSON ONLY:\n"
            '   {"status": "rejected", "analysis": "Not a medical image. '
            'Please upload a medical image such as a lab result, prescription, or scan."}\n\n'
            "2. If medical → respond with JSON ONLY:\n"
            '   {"status": "success", "analysis": "<structured analysis>"}\n\n'
            "3. Analysis must include: document type, key findings, abnormal values, "
            "next steps, urgent findings.\n"
            "4. NEVER provide a definitive diagnosis.\n"
            "5. Respond in the SAME language as image content.\n"
            "6. Output ONLY valid JSON — no markdown, no code fences."
        )

        b64_data = base64.b64encode(image_bytes).decode("utf-8")
        types    = _gemini_types()

        for model_name in GEMINI_VISION_MODELS:
            content_variants = [
                [
                    system_prompt,
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                ],
                [
                    {"text": system_prompt},
                    {"inline_data": {"mime_type": mime_type, "data": b64_data}},
                ],
            ]
            for contents in content_variants:
                try:
                    log.info(f"[Vision] model: {model_name}")
                    resp = _get_gemini_sync().models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=self._make_config(temperature=0.1, max_tokens=4096),
                    )
                    raw   = resp.text.strip()
                    clean = re.sub(
                        r"^\s*```+(?:json)?\s*|\s*```+\s*$", "", raw, flags=re.MULTILINE
                    ).strip()
                    brace = clean.find("{")
                    if brace > 0:
                        clean = clean[brace:]

                    try:
                        parsed   = json.loads(clean)
                        status   = str(parsed.get("status", "success"))
                        analysis = parsed.get("analysis", "")
                        if isinstance(analysis, dict):
                            analysis = json.dumps(analysis, ensure_ascii=False, indent=2)
                        if not isinstance(analysis, str):
                            analysis = json.dumps(analysis, ensure_ascii=False, indent=2)
                        log.info(f"[Vision] Success — {model_name} status={status}")
                        return status, analysis.strip(), model_name
                    except json.JSONDecodeError:
                        # Best-effort: scan for first complete JSON object
                        start = clean.find("{")
                        if start != -1:
                            depth = 0
                            in_str = esc = False
                            for i, ch in enumerate(clean[start:], start):
                                if esc:
                                    esc = False
                                elif ch == "\\":
                                    esc = True
                                elif ch == '"' and not esc:
                                    in_str = not in_str
                                elif not in_str:
                                    if ch == "{":
                                        depth += 1
                                    elif ch == "}":
                                        depth -= 1
                                        if depth == 0:
                                            try:
                                                parsed   = json.loads(clean[start: i + 1])
                                                status   = str(parsed.get("status", "success"))
                                                analysis = parsed.get("analysis", "")
                                                if not isinstance(analysis, str):
                                                    analysis = json.dumps(analysis, ensure_ascii=False)
                                                return status, analysis.strip(), model_name
                                            except Exception:
                                                break
                        log.warning(f"[Vision] {model_name}: non-JSON — using raw text.")
                        return "success", raw, model_name
                except Exception as exc:
                    log.warning(f"[Vision] {model_name} variant failed: {exc}")
                    continue

        log.error("[Vision] All models exhausted.")
        return ("error", "تعذّر تحليل الصورة مؤقتاً. يرجى المحاولة مرة أخرى لاحقاً.", "none")

    async def analyze_image(self, image_bytes: bytes, mime_type: str) -> Tuple[str, str, str]:
        return await asyncio.to_thread(self._analyze_image_sync, image_bytes, mime_type)

    @property
    def cache_size(self) -> int:
        return len(self._cache)


# ──────────────────────────────────────────────────────────────────
# Application State
# ──────────────────────────────────────────────────────────────────

class AppState:
    knowledge_base: Optional[KnowledgeBaseService] = None
    gemini: Optional[GeminiService]                = None
    prompt_builder: Optional[PromptBuilder]        = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _request_semaphore
    log.info("🚀 Sila v14.0 — zero-SDK boot.")
    _request_semaphore    = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    state.knowledge_base  = KnowledgeBaseService()
    state.gemini          = GeminiService()
    state.prompt_builder  = PromptBuilder()
    log.info(
        f"✅ Boot complete — embed=local/{EMBED_MODEL} dim={EMBED_DIM} "
        f"concurrency={MAX_CONCURRENT_REQUESTS} timeout={EXTERNAL_CALL_TIMEOUT}s "
        f"min_confidence={MIN_CONFIDENCE}"
    )
    yield
    log.info("🛑 Sila shutting down.")


# ──────────────────────────────────────────────────────────────────
# FastAPI Application
# ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Sila — Medical AI Assistant",
    description="مساعد طبي ذكي | Strict RAG + Gemini Vision + Arabic & English",
    version="14.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.error(f"Unhandled {type(exc).__name__} on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "An unexpected error occurred. Please try again later."},
    )


# ──────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "Sila — Medical AI Assistant",
        "version": "14.0.0",
        "status": "running",
        "endpoints": ["/ask", "/analyze-image", "/health", "/docs"],
    }


@app.get("/health")
def health():
    # Never touches any SDK — always fast for Railway healthcheck
    return {
        "status": "ok",
        "version": "14.0.0",
        "embed_model": EMBED_MODEL,
        "embed_dim": EMBED_DIM,
        "index": INDEX_NAME,
        "min_confidence": MIN_CONFIDENCE,
        "top_k": TOP_K,
    }


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, request: Request) -> AskResponse:
    if _request_semaphore is None:
        return JSONResponse(status_code=503, content={"error": "Server not ready yet."})

    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        lang = LanguageDetector.detect(req.query)
        msg  = (
            "لقد تجاوزت الحد المسموح من الطلبات. يرجى المحاولة بعد دقيقة."
            if lang == "ar"
            else "You have exceeded the rate limit. Please try again after a minute."
        )
        return JSONResponse(status_code=429, content={"error": msg})

    try:
        await _request_semaphore.acquire()
    except Exception:
        lang = LanguageDetector.detect(req.query)
        msg  = (
            "الخادم مشغول حالياً. يرجى المحاولة بعد لحظات."
            if lang == "ar"
            else "Server is busy. Please try again in a moment."
        )
        return JSONResponse(status_code=503, content={"error": msg})

    try:
        return await asyncio.wait_for(_ask_inner(req), timeout=EXTERNAL_CALL_TIMEOUT)
    except asyncio.TimeoutError:
        lang = LanguageDetector.detect(req.query)
        log.warning(f"[ASK] Timeout after {EXTERNAL_CALL_TIMEOUT}s: {req.query[:60]}")
        msg  = (
            "عذراً، استغرق الطلب وقتاً أطول من المتوقع. يرجى المحاولة مرة أخرى."
            if lang == "ar"
            else "Sorry, the request timed out. Please try again."
        )
        return AskResponse(
            query=req.query, reply=msg, model_used="none", matches=[],
            is_medical=True, found_in_database=False, low_confidence=True,
            language=lang, disclaimer=MEDICAL_DISCLAIMER,
        )
    finally:
        try:
            _request_semaphore.release()
        except Exception:
            pass


async def _ask_inner(req: AskRequest) -> AskResponse:
    q        = req.query
    language = LanguageDetector.detect(q)
    intent   = await IntentClassifier.classify(q)

    log.info(f"[ASK] lang={language} intent={intent} query='{q[:80]}'")

    # ── Social path ───────────────────────────────────────────────
    if intent == "social":
        reply, model_used = await state.gemini.reply_social(q, language)
        return AskResponse(
            query=q, reply=reply, model_used=model_used, matches=[],
            is_medical=False, found_in_database=False, low_confidence=False,
            language=language, disclaimer=MEDICAL_DISCLAIMER,
        )

    # ── Medical / RAG path ────────────────────────────────────────
    try:
        matches = await state.knowledge_base.search(q, top_k=TOP_K)
    except Exception as exc:
        log.error(f"[ASK] KB search error: {exc}")
        matches = []

    ctx = QueryContext(raw_query=q, language=language, matches=matches)

    if not ctx.has_reliable_matches:
        reply = state.prompt_builder.no_data_response(language)
        log.info(
            f"[ASK] No reliable matches — best={ctx.best_confidence:.4f} "
            f"total={len(matches)}"
        )
        return AskResponse(
            query=q, reply=reply, model_used="none", matches=[],
            is_medical=True, found_in_database=False, low_confidence=True,
            language=language, disclaimer=MEDICAL_DISCLAIMER,
        )

    try:
        prompt              = state.prompt_builder.build(ctx)
        reply, model_used   = await state.gemini.generate(prompt)
        log.info(f"[ASK] RAG ok — model={model_used} top={ctx.best_confidence:.4f}")
        return AskResponse(
            query=q, reply=reply, model_used=model_used,
            matches=[
                MatchResult(
                    question=m.question, answer=m.answer,
                    confidence=m.confidence, category=m.category,
                )
                for m in matches
            ],
            is_medical=True, found_in_database=True, low_confidence=False,
            language=language, disclaimer=MEDICAL_DISCLAIMER,
        )
    except Exception as exc:
        log.error(f"[ASK] Gemini generation failed: {exc}")
        reply = state.prompt_builder.no_data_response(language)
        return AskResponse(
            query=q, reply=reply, model_used="none",
            matches=[
                MatchResult(
                    question=m.question, answer=m.answer,
                    confidence=m.confidence, category=m.category,
                )
                for m in matches
            ],
            is_medical=True, found_in_database=False, low_confidence=True,
            language=language, disclaimer=MEDICAL_DISCLAIMER,
        )


@app.post("/analyze-image", response_model=ImageAnalysisResponse)
async def analyze_image(file: UploadFile = File(...)) -> JSONResponse:
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        log.warning(f"[IMAGE] Rejected: {file.content_type}")
        return JSONResponse(status_code=400, content={
            "status": "error",
            "analysis": (
                f"نوع الملف '{file.content_type}' غير مدعوم. "
                "الأنواع المقبولة: JPEG, PNG, WEBP, HEIC, HEIF."
            ),
            "model_used": "none",
            "disclaimer": MEDICAL_DISCLAIMER,
        })

    try:
        image_bytes = await file.read()
    except Exception as exc:
        log.error(f"[IMAGE] Read failed '{file.filename}': {exc}")
        return JSONResponse(status_code=400, content={
            "status": "error",
            "analysis": "فشل في قراءة الملف. تأكد من أن الصورة غير تالفة وحاول مرة أخرى.",
            "model_used": "none",
            "disclaimer": MEDICAL_DISCLAIMER,
        })

    if not image_bytes:
        return JSONResponse(status_code=400, content={
            "status": "error",
            "analysis": "الملف المرفوع فارغ. يرجى رفع صورة صحيحة.",
            "model_used": "none",
            "disclaimer": MEDICAL_DISCLAIMER,
        })

    if len(image_bytes) > MAX_IMAGE_BYTES:
        return JSONResponse(status_code=413, content={
            "status": "error",
            "analysis": (
                f"حجم الصورة يتجاوز الحد المسموح به ({MAX_IMAGE_MB}MB). "
                "يرجى ضغط الصورة وإعادة المحاولة."
            ),
            "model_used": "none",
            "disclaimer": MEDICAL_DISCLAIMER,
        })

    log.info(f"[IMAGE] Processing: '{file.filename}' {len(image_bytes)/1024:.1f}KB")

    try:
        status, analysis, model_used = await asyncio.wait_for(
            state.gemini.analyze_image(image_bytes, file.content_type),
            timeout=EXTERNAL_CALL_TIMEOUT,
        )
    except asyncio.TimeoutError:
        log.warning("[IMAGE] Timed out.")
        return JSONResponse(status_code=504, content={
            "status": "error",
            "analysis": "انتهت مهلة تحليل الصورة. يرجى المحاولة مرة أخرى.",
            "model_used": "none",
            "disclaimer": MEDICAL_DISCLAIMER,
        })

    http_status = 503 if status == "error" else 200
    log.info(f"[IMAGE] Done — status={status} model={model_used}")

    return JSONResponse(status_code=http_status, content={
        "status": status,
        "analysis": analysis,
        "model_used": model_used,
        "disclaimer": MEDICAL_DISCLAIMER,
    })