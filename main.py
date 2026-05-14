"""
╔══════════════════════════════════════════════════════════════════╗
║          SILA — Medical AI Assistant  v13.2                     ║
║          Production-Hardened · Async-Safe · Zero Blocking       ║
║                                                                  ║
║  Stack : FastAPI + Pinecone v3 + Gemini (google-genai SDK)      ║
║  Mode  : Strict RAG for medical · Social chat for greetings     ║
║  Vision: Medical image analysis via Gemini Vision               ║
║                                                                  ║
║  v13.2 vs v13.1  (Semantic retrieval fix):                    ║
║  • MIN_CONFIDENCE now uses SCORE_THRESHOLD env var (0.45)      ║
║  • Added TOP_K, MAX_RETRIES, RETRY_DELAY env vars             ║
║  • EMBEDDING_BACKEND default changed to 'local'                ║
║  • Strong semantic filter in _parse_matches to prevent         ║
║    genetic diseases/rare syndromes from matching simple        ║
║    symptoms like headache/pain/fever                            ║
║  • Query-side safety cleanup in _ask_inner to clear            ║
║    low-confidence or irrelevant matches                        ║
║  • Pinecone query always includes namespace (default "")      ║
║  • Enhanced logging: query text, embedding model, vector       ║
║    dimension, raw scores, filtered counts                      ║
║  • Fallback to no_data_response for irrelevant results         ║
╚══════════════════════════════════════════════════════════════════╝

Railway start command (recommended):
  gunicorn main:app -k uvicorn.workers.UvicornWorker \
    --workers 2 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT

Or single-worker uvicorn:
  uvicorn main:app --host 0.0.0.0 --port $PORT \
    --loop asyncio --http httptools
"""

# ── Unbuffered output must come first for Railway log visibility ──
import os
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── Standard library ──────────────────────────────────────────────
import asyncio
import base64
import hashlib
import json
import logging
import re
import sys
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

sys.setrecursionlimit(1000)

# ── Lightweight third-party (negligible boot cost) ────────────────
from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, model_validator

# ── Heavy SDKs (google.genai, pinecone) are NOT imported here ─────
# They are lazy-imported inside _init_gemini() / _init_pinecone()
# so the process starts and passes Railway healthcheck before any
# SDK initialisation happens.

load_dotenv()

# ──────────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
INDEX_NAME       = os.getenv("PINECONE_INDEX", "medical-index-arabicdata")

# v13.1: explicit namespace control — empty string = default namespace
# Set PINECONE_NAMESPACE in env if your vectors were upserted with a namespace.
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "")

# v13.2: Using SCORE_THRESHOLD env var for stricter filtering.
# Default 0.45 prevents semantically irrelevant matches (e.g., genetic diseases for simple symptoms).
MIN_CONFIDENCE   = float(os.getenv("SCORE_THRESHOLD", "0.45"))

MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "500"))
MAX_IMAGE_MB     = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
MAX_IMAGE_BYTES  = MAX_IMAGE_MB * 1024 * 1024

# v13.2: Pinecone query configuration
TOP_K             = int(os.getenv("TOP_K", "7"))
MAX_RETRIES       = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY       = float(os.getenv("RETRY_DELAY", "1.5"))

EMBEDDING_BACKEND  = os.getenv("EMBEDDING_BACKEND", "local").lower()
EMBED_MODEL        = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")

# How many /ask requests run concurrently before returning 503.
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "20"))

# Timeout (seconds) for any single /ask or /analyze-image call end-to-end.
EXTERNAL_CALL_TIMEOUT = int(os.getenv("EXTERNAL_CALL_TIMEOUT", "25"))

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set.")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is not set.")

# ──────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("sila")

# ──────────────────────────────────────────────────────────────────
# Concurrency gate  (initialised in lifespan, needs event loop)
# ──────────────────────────────────────────────────────────────────
_request_semaphore: Optional[asyncio.Semaphore] = None

# ──────────────────────────────────────────────────────────────────
# Simple in-memory IP-based rate limiter (v13.3)
# ──────────────────────────────────────────────────────────────────
from collections import defaultdict
import time as _time

_rate_limit_store: dict[str, list[float]] = defaultdict(list)
_rate_limit_lock = threading.Lock()
_RATE_LIMIT_REQUESTS = 10
_RATE_LIMIT_WINDOW = 60  # seconds

def _check_rate_limit(ip: str) -> bool:
    """Check if IP has exceeded rate limit. Returns True if allowed."""
    now = _time.time()
    with _rate_limit_lock:
        # Clean old requests outside the window
        _rate_limit_store[ip] = [
            ts for ts in _rate_limit_store[ip] if now - ts < _RATE_LIMIT_WINDOW
        ]
        # Check if under limit
        if len(_rate_limit_store[ip]) < _RATE_LIMIT_REQUESTS:
            _rate_limit_store[ip].append(now)
            return True
        return False

# ──────────────────────────────────────────────────────────────────
# Lazy SDK getters  (imports happen here, NEVER at module level)
# ──────────────────────────────────────────────────────────────────

_pinecone_index = None
_pinecone_lock  = threading.Lock()


def _init_pinecone():
    """Blocking init — always called via asyncio.to_thread."""
    global _pinecone_index
    if _pinecone_index is None:
        with _pinecone_lock:
            if _pinecone_index is None:
                log.info("Initializing Pinecone (lazy import)...")
                from pinecone import Pinecone as _PC  # noqa: PLC0415
                _pinecone_index = _PC(api_key=PINECONE_API_KEY).Index(INDEX_NAME)
                log.info("Pinecone ready.")
    return _pinecone_index


async def get_index():
    """Async-safe Pinecone getter."""
    if _pinecone_index is None:
        await asyncio.to_thread(_init_pinecone)
    return _pinecone_index


_gemini_client = None
_gemini_lock   = threading.Lock()


def _init_gemini():
    """Blocking init — always called via asyncio.to_thread or inside threadpool."""
    global _gemini_client
    if _gemini_client is None:
        with _gemini_lock:
            if _gemini_client is None:
                log.info("Initializing Gemini client (lazy import)...")
                from google import genai as _genai  # noqa: PLC0415
                _gemini_client = _genai.Client(api_key=GEMINI_API_KEY)
                log.info("Gemini client ready.")
    return _gemini_client


# Sync version for use inside threadpool workers
def _get_gemini_sync():
    return _init_gemini()


def _gemini_types():
    """Returns google.genai.types — lazy imported."""
    from google.genai import types  # noqa: PLC0415
    return types


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
    "ألم", "وجع", "مرض", "دواء", "طبيب", "مستشفى", "أعراض", "علاج",
    "صداع", "حمى", "سعال", "ضغط", "سكر", "قلب", "كلى", "معدة",
    "عظام", "جلد", "عين", "أذن", "أنف", "رئة", "كبد", "دم",
    "تعب", "إرهاق", "دوار", "غثيان", "إسهال", "إمساك", "حرقة",
    "الم", "عندي", "عندى", "اشعر", "احس", "اعاني", "يؤلم",
    "بوجعني", "بتوجعني", "حاسس", "حاسه", "حبوب", "طفح", "حكة",
    "عملية", "جراحة", "منظار", "تحليل", "أشعة", "نتيجة", "تقرير",
    "pain", "ache", "fever", "cough", "headache", "nausea", "dizzy",
    "vomit", "diarrhea", "symptom", "disease", "doctor", "hospital",
    "medicine", "drug", "blood", "heart", "lung", "kidney", "liver",
    "diabetes", "pressure", "infection", "allergy", "rash", "swelling",
    "fatigue", "tired", "breathe", "chest", "stomach", "throat",
    "surgery", "scan", "test", "result", "report", "prescription",
    # v13.1: added common cold / flu terms that were missing
    "برد", "انفلونزا", "رشح", "زكام", "كحة", "بلغم", "حرارة",
    "cold", "flu", "runny", "nose", "sneeze", "congestion",
})

# ──────────────────────────────────────────────────────────────────
# Embedding Backends  (all async-safe via threadpool)
# ──────────────────────────────────────────────────────────────────

class _GeminiEmbedder:
    # v13.1: Read dimension from env; default 768 matches text-embedding-004.
    # CRITICAL: this value MUST match the dimension used when upserting vectors.
    # If your index was built with a different model/dimension, set GEMINI_EMBED_DIM
    # in your environment to match. Mismatch causes silent wrong-result queries.
    _DIM: int = int(os.getenv("GEMINI_EMBED_DIM", "768"))

    @classmethod
    def _encode_sync(cls, text: str) -> List[float]:
        types = _gemini_types()
        result = _get_gemini_sync().models.embed_content(
            model=GEMINI_EMBED_MODEL,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=cls._DIM),
        )
        vec = list(result.embeddings[0].values)
        # v13.1: log dimension so mismatches are immediately visible in logs
        log.info(
            f"[Embed] model={GEMINI_EMBED_MODEL} dim={len(vec)} "
            f"configured_dim={cls._DIM} "
            f"first3={[round(v, 4) for v in vec[:3]]}"
        )
        return vec

    @classmethod
    async def encode(cls, text: str) -> List[float]:
        try:
            return await asyncio.to_thread(cls._encode_sync, text)
        except Exception as exc:
            log.error(f"[GeminiEmbedder] failed: {exc}")
            raise


class _LocalEmbedder:
    _instance: Any = None
    _lock           = threading.Lock()
    _load_error: Optional[str] = None

    @classmethod
    def _load_and_encode_sync(cls, text: str) -> List[float]:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    if cls._load_error:
                        raise RuntimeError(f"Local embedder previously failed: {cls._load_error}")
                    try:
                        log.info(f"⏳ Lazy-loading SentenceTransformer: {EMBED_MODEL}")
                        t0 = time.perf_counter()
                        from sentence_transformers import SentenceTransformer  # noqa
                        model = SentenceTransformer(EMBED_MODEL)
                        try:
                            import torch as _torch  # noqa
                            _torch.set_num_threads(2)
                            _torch.set_num_interop_threads(1)
                            model = model.to(_torch.device("cpu"))
                        except ImportError:
                            pass
                        log.info(f"✅ SentenceTransformer ready in {time.perf_counter()-t0:.2f}s")
                        cls._instance = model
                    except Exception as exc:
                        cls._load_error = str(exc)
                        raise
        return cls._instance.encode(text, show_progress_bar=False).tolist()

    @classmethod
    async def encode(cls, text: str) -> List[float]:
        return await asyncio.to_thread(cls._load_and_encode_sync, text)

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._instance is not None


class EmbeddingRouter:
    class EmbeddingUnavailableError(RuntimeError):
        pass

    @staticmethod
    async def encode(text: str) -> List[float]:
        """Fully async — never blocks the event loop."""
        if EMBEDDING_BACKEND == "gemini":
            try:
                return await _GeminiEmbedder.encode(text)
            except Exception as exc:
                raise EmbeddingRouter.EmbeddingUnavailableError(
                    "Gemini embedding service is currently unavailable."
                ) from exc

        if EMBEDDING_BACKEND == "local":
            try:
                return await _LocalEmbedder.encode(text)
            except Exception as exc:
                raise EmbeddingRouter.EmbeddingUnavailableError(
                    "Local embedding model is currently unavailable."
                ) from exc

        raise EmbeddingRouter.EmbeddingUnavailableError(
            "Embedding is disabled (EMBEDDING_BACKEND=none)."
        )


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

# ── v13.0: MessageDto mirrors .NET MessageDto exactly ─────────────
class MessageDto(BaseModel):
    """Single conversation turn — role is 'user' or 'assistant'."""
    model_config = {"arbitrary_types_allowed": True}
    role: str
    content: str


# ── v13.0: AskRequest fixed to accept `question` (from .NET) ──────
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
            raise ValueError(
                "Request must include a non-empty 'question' or 'text' field."
            )
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
        ratio = arabic_chars / max(len(text.strip()), 1)
        return "ar" if ratio > 0.25 else "en"


# ──────────────────────────────────────────────────────────────────
# Intent Classifier  (async-safe)
# ──────────────────────────────────────────────────────────────────

class IntentClassifier:
    _PROMPT = (
        "Classify this message into exactly one category.\n\n"
        "Categories:\n"
        "- social  : greetings, thanks, casual conversation, feelings, non-medical\n"
        "- medical : symptoms, diseases, medications, body parts, pain, health, lab results\n\n"
        "Rules:\n"
        "- Reply with ONE word only: social OR medical\n"
        "- No punctuation, no explanation\n\n"
        "Message: {query}"
    )

    @classmethod
    def _classify_sync(cls, query: str) -> str:
        try:
            types = _gemini_types()
            resp = _get_gemini_sync().models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=cls._PROMPT.format(query=query),
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=5),
            )
            result = resp.text.strip().lower()
            if "social" in result:
                return "social"
            if "medical" in result:
                return "medical"
        except Exception as exc:
            log.warning(f"IntentClassifier: Gemini unavailable, keyword fallback — {exc}")

        q_lower = query.lower()
        return "medical" if any(kw in q_lower for kw in MEDICAL_KEYWORDS) else "social"

    @classmethod
    async def classify(cls, query: str) -> str:
        return await asyncio.to_thread(cls._classify_sync, query)


# ──────────────────────────────────────────────────────────────────
# Knowledge Base Service  (fully async)
# ──────────────────────────────────────────────────────────────────

class KnowledgeBaseService:
    _MAX_RETRIES = MAX_RETRIES
    _RETRY_DELAY = RETRY_DELAY

    # v13.4: Simple keyword-based fallback retrieval
    @staticmethod
    def _keyword_fallback(query: str) -> List[KnowledgeMatch]:
        """Simple keyword-based fallback when vector search fails."""
        # This is a placeholder - in production, you'd implement actual keyword search
        # For now, return empty list to allow graceful degradation
        log.info("[KnowledgeBase] Using keyword fallback (not implemented yet)")
        return []

    async def search(self, query: str, top_k: int = None) -> List[KnowledgeMatch]:
        if top_k is None:
            top_k = TOP_K
        
        # v13.4: Log query text for debugging
        log.info(f"[KnowledgeBase] Search query: '{query[:100]}'")
        
        try:
            vector = await EmbeddingRouter.encode(query)
        except EmbeddingRouter.EmbeddingUnavailableError as exc:
            log.error(f"[KnowledgeBase] Embedding failed: {exc}")
            # v13.4: Return empty list instead of crashing
            return []
        
        index  = await get_index()

        # v13.4: Vector dimension safety check with graceful fallback
        expected_dim = 384 if EMBEDDING_BACKEND == "local" else 768
        if len(vector) != expected_dim:
            log.warning(
                f"[KnowledgeBase] Vector dimension mismatch: got {len(vector)}, expected {expected_dim}. "
                f"Proceeding with query anyway - may fail in Pinecone."
            )
            # v13.4: Don't crash, log and proceed

        # v13.4: log vector stats so dimension/value anomalies are visible
        log.info(
            f"[KnowledgeBase] Query vector — dim={len(vector)} "
            f"min={min(vector):.4f} max={max(vector):.4f} "
            f"index='{INDEX_NAME}' "
            f"namespace='{PINECONE_NAMESPACE or '<default>'}' "
            f"backend={EMBEDDING_BACKEND} "
            f"model={EMBED_MODEL if EMBEDDING_BACKEND=='local' else GEMINI_EMBED_MODEL}"
        )

        last_exc: Optional[Exception] = None
        start_time = time.perf_counter()
        
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                # v13.4: Always include namespace (default to empty string)
                query_kwargs: dict = dict(
                    vector=vector,
                    top_k=top_k,
                    include_metadata=True,
                    namespace=PINECONE_NAMESPACE or "",
                )

                results = await asyncio.to_thread(
                    index.query,
                    **query_kwargs,
                )

                # v13.4: log raw Pinecone scores BEFORE any filtering
                raw_scores = [round(m.score, 4) for m in results.matches]
                log.info(
                    f"[KnowledgeBase] Pinecone raw scores (top_{top_k}): {raw_scores} "
                    f"| MIN_CONFIDENCE={MIN_CONFIDENCE} "
                    f"| raw_count={len(results.matches)}"
                )

                matches = self._parse_matches(results, query)
                latency = time.perf_counter() - start_time
                log.info(
                    f"[KnowledgeBase] After filter: {len(matches)} matches kept "
                    f"(threshold≥{MIN_CONFIDENCE:.3f}) "
                    f"| latency={latency:.3f}s"
                )
                return matches

            except Exception as exc:
                last_exc = exc
                log.warning(
                    f"[KnowledgeBase] Pinecone attempt {attempt}/{self._MAX_RETRIES} failed: {exc}"
                )
                if attempt < self._MAX_RETRIES:
                    await asyncio.sleep(self._RETRY_DELAY * attempt)

        # v13.4: Return empty list instead of raising exception
        log.error(f"[KnowledgeBase] All retries exhausted: {last_exc}")
        return []

    @staticmethod
    def _parse_matches(results: Any, query: str) -> List[KnowledgeMatch]:
        # v13.4: Simplified filtering - use confidence threshold only
        # Removed over-aggressive keyword filtering that was removing valid matches
        pre_filter = MIN_CONFIDENCE
        
        matches = []
        filtered_count = 0
        
        for m in results.matches:
            if m.score < pre_filter:
                filtered_count += 1
                continue
            
            meta = m.metadata or {}
            matches.append(KnowledgeMatch(
                question=meta.get("question", ""),
                answer=meta.get("answer", ""),
                confidence=round(float(m.score), 4),
                category=meta.get("category", "General"),
            ))
        
        log.info(
            f"[KnowledgeBase] Filter summary: pre_filter={filtered_count} kept={len(matches)}"
        )
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
        "٨. استفد من التخصص الطبي المذكور في السياق لتحسين دقة الإجابة.\n"
        "٩. لا تُجيب إلا بناءً على السياق المسترجع من قاعدة المعرفة."
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
        "7. Leverage the medical specialty mentioned in the context to improve accuracy.\n"
        "8. Only answer based on the retrieved context from the knowledge base."
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
        lang      = ctx.language
        system    = self._SYSTEM_AR    if lang == "ar" else self._SYSTEM_EN
        structure = self._STRUCTURE_AR if lang == "ar" else self._STRUCTURE_EN

        context_parts = []
        for i, m in enumerate(ctx.matches, start=1):
            reliability = (
                ("✅ موثوق" if m.is_reliable else "⚠️ ثقة منخفضة")
                if lang == "ar"
                else ("✅ Reliable" if m.is_reliable else "⚠️ Low confidence")
            )
            category_label = m.category or "General"
            context_parts.append(
                f"[{i}] {reliability} — Score: {m.confidence:.0%}\n"
                f"[Specialty: {category_label}]\n"
                f"Q: {m.question}\n"
                f"A: {m.answer}"
            )

        sep            = "━" * 50
        context_block  = f"\n\n{sep}\n".join(context_parts)
        label_context  = "📋 قاعدة المعرفة الطبية:" if lang == "ar" else "📋 Medical Knowledge Base:"
        label_question = "🧑‍⚕️ سؤال المريض:"       if lang == "ar" else "🧑‍⚕️ Patient Question:"
        label_answer   = "الإجابة:"                  if lang == "ar" else "Answer:"

        return (
            f"{system}\n\n"
            f"{sep}\n"
            f"{label_context}\n\n"
            f"{context_block}\n\n"
            f"{sep}\n"
            f"{label_question}\n{ctx.raw_query}\n\n"
            f"{structure}\n\n"
            f"{label_answer}"
        )

    def no_data_response(self, language: str) -> str:
        return self._NO_DATA_AR if language == "ar" else self._NO_DATA_EN


# ──────────────────────────────────────────────────────────────────
# Gemini Service  (all calls offloaded to threadpool)
# ──────────────────────────────────────────────────────────────────

class GeminiService:
    """
    v13.1: All Gemini calls are offloaded to threadpool to avoid blocking the event loop.
    This is critical for Railway deployment where the process must stay responsive.
    """

    # v13.3: Bounded LRU cache (max 200 entries)
    _cache: dict[str, Tuple[str, str]] = {}
    _cache_lock = threading.Lock()
    _cache_max_size = 200
    _cache_access_order: list[str] = []

    def __init__(self) -> None:
        pass

    @staticmethod
    def _make_config(temperature: float = 0.2, max_tokens: int = 2048):
        types = _gemini_types()
        return types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    # ── Social replies ─────────────────────────────────────────────

    def _reply_social_sync(self, query: str, language: str) -> Tuple[str, str]:
        system = (
            "You are 'Sila', a friendly and warm medical AI assistant. "
            "Reply naturally in English. Keep it brief (1-2 sentences). "
            "If the topic is not medical, warmly mention that you specialize "
            "in medical consultations and invite them to ask health-related questions."
            if language == "en"
            else
            "أنت 'سيلا'، مساعد طبي ذكي وودود.\n"
            "رد بالعربية بشكل طبيعي ودافئ. الرد قصير (جملة أو اتنين بالكثير).\n"
            "لو الموضوع مش طبي، قول بلطف إنك متخصص في الاستشارات الطبية "
            "وادعوه يسأل أي سؤال صحي."
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
                log.warning(f"Social reply — {model_name} failed: {exc}")

        fallback = (
            "Hello! 😊 I'm Sila, your medical AI assistant. How can I help you today?"
            if language == "en"
            else "أهلاً! 😊 أنا سيلا، مساعدتك الطبية. كيف يمكنني مساعدتك؟"
        )
        return fallback, "fallback"

    async def reply_social(self, query: str, language: str) -> Tuple[str, str]:
        return await asyncio.to_thread(self._reply_social_sync, query, language)

    # ── RAG generation ─────────────────────────────────────────────

    def _generate_sync(self, prompt: str) -> Tuple[str, str]:
        cache_key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        with self._cache_lock:
            if cache_key in self._cache:
                log.info("Cache hit — reusing previous response.")
                # Update access order for LRU
                if cache_key in self._cache_access_order:
                    self._cache_access_order.remove(cache_key)
                self._cache_access_order.append(cache_key)
                return self._cache[cache_key]

        for model_name in GEMINI_TEXT_MODELS:
            try:
                log.info(f"RAG generate — trying model: {model_name}")
                resp = _get_gemini_sync().models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=self._make_config(),
                )
                text = resp.text.strip()
                with self._cache_lock:
                    # LRU eviction if cache is full
                    if len(self._cache) >= self._cache_max_size:
                        oldest_key = self._cache_access_order.pop(0)
                        if oldest_key in self._cache:
                            del self._cache[oldest_key]
                    self._cache[cache_key] = (text, model_name)
                    self._cache_access_order.append(cache_key)
                return text, model_name
            except Exception as exc:
                log.warning(f"RAG generate — {model_name} failed: {exc}")

        log.error("All text models exhausted.")
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
            "1. If the image is NOT medical → respond with this JSON ONLY:\n"
            '   {"status": "rejected", "analysis": "This image does not appear to be '
            'a medical document. Please upload a medical image such as a lab result, '
            'prescription, or scan."}\n\n'
            "2. If the image IS medical → respond with this JSON ONLY:\n"
            '   {"status": "success", "analysis": "<structured analysis>"}\n\n'
            "3. For medical images, the analysis must include:\n"
            "   - Document type identified\n"
            "   - Key findings\n"
            "   - Any abnormal or critical values\n"
            "   - Recommended next steps\n"
            "   - Any urgent findings requiring immediate attention\n\n"
            "4. NEVER provide a definitive diagnosis.\n"
            "5. Respond in the SAME language as the image content (Arabic or English).\n"
            "6. Output ONLY valid JSON — no markdown, no code fences, no preamble."
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
                    log.info(f"Vision analyze — trying model: {model_name}")
                    resp  = _get_gemini_sync().models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=self._make_config(temperature=0.1, max_tokens=4096),
                    )
                    raw   = resp.text.strip()

                    # ── Hardened JSON extraction ──────────────────
                    clean = re.sub(
                        r"^\s*```+(?:json)?\s*|\s*```+\s*$",
                        "",
                        raw,
                        flags=re.MULTILINE,
                    ).strip()

                    brace = clean.find("{")
                    if brace > 0:
                        clean = clean[brace:]

                    try:
                        parsed   = json.loads(clean)
                        status   = str(parsed.get("status", "success"))
                        analysis = parsed.get("analysis", "")
                        if isinstance(analysis, dict):
                            analysis = analysis.get("analysis") or json.dumps(
                                analysis, ensure_ascii=False, indent=2
                            )
                        if not isinstance(analysis, str):
                            analysis = json.dumps(analysis, ensure_ascii=False, indent=2)
                        log.info(f"Vision succeeded — model: {model_name}, status: {status}")
                        return status, analysis.strip(), model_name
                    except json.JSONDecodeError:
                        # v13.3: Try to extract first JSON object only
                        try:
                            import json as _json
                            # Find first complete JSON object
                            start = clean.find("{")
                            if start != -1:
                                brace_count = 0
                                in_string = False
                                escape_next = False
                                for i in range(start, len(clean)):
                                    char = clean[i]
                                    if escape_next:
                                        escape_next = False
                                    elif char == "\\":
                                        escape_next = True
                                    elif char == '"' and not escape_next:
                                        in_string = not in_string
                                    elif not in_string:
                                        if char == "{":
                                            brace_count += 1
                                        elif char == "}":
                                            brace_count -= 1
                                            if brace_count == 0:
                                                json_str = clean[start:i+1]
                                                parsed = _json.loads(json_str)
                                                status = str(parsed.get("status", "success"))
                                                analysis = parsed.get("analysis", "")
                                                if isinstance(analysis, dict):
                                                    analysis = analysis.get("analysis") or _json.dumps(
                                                        analysis, ensure_ascii=False, indent=2
                                                    )
                                                if not isinstance(analysis, str):
                                                    analysis = _json.dumps(analysis, ensure_ascii=False, indent=2)
                                                log.info(f"Vision succeeded (partial JSON) — model: {model_name}, status: {status}")
                                                return status, analysis.strip(), model_name
                        except Exception:
                            pass
                        # If all JSON parsing fails, use raw text
                        log.warning(f"Vision {model_name}: non-JSON response — using raw text.")
                        return "success", raw, model_name
                except Exception as exc:
                    log.warning(f"Vision {model_name} variant failed: {exc}")
                    continue

        log.error("All vision models exhausted.")
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
    gemini: Optional[GeminiService] = None
    prompt_builder: Optional[PromptBuilder] = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _request_semaphore
    log.info("🚀 Sila v13.1 — zero-SDK boot, async-safe runtime.")
    _request_semaphore   = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    state.knowledge_base = KnowledgeBaseService()
    state.gemini         = GeminiService()
    state.prompt_builder = PromptBuilder()
    log.info(
        f"✅ Boot complete — concurrency={MAX_CONCURRENT_REQUESTS} "
        f"timeout={EXTERNAL_CALL_TIMEOUT}s "
        f"min_confidence={MIN_CONFIDENCE} "
        f"namespace='{PINECONE_NAMESPACE or '<default>'}'"
    )
    yield
    log.info("🛑 Sila shutting down.")


# ──────────────────────────────────────────────────────────────────
# FastAPI Application
# ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Sila — Medical AI Assistant",
    description="مساعد طبي ذكي | Strict RAG + Gemini Vision + Arabic & English",
    version="13.2.0",
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
    log.error(f"Unhandled exception on {request.url.path}: {type(exc).__name__}: {exc}")
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
        "name":      "Sila — Medical AI Assistant",
        "version":   "13.2.0",
        "status":    "running",
        "endpoints": ["/ask", "/analyze-image", "/health", "/docs"],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "13.2.0",
        "embedding_backend": EMBEDDING_BACKEND,
        "index": INDEX_NAME,
        "min_confidence": MIN_CONFIDENCE,
        "top_k": TOP_K,
        "max_retries": MAX_RETRIES,
    }


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, request: Request) -> AskResponse:
    if _request_semaphore is None:
        return JSONResponse(status_code=503, content={"error": "Server not ready yet."})

    # v13.3: Rate limit check
    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        lang = LanguageDetector.detect(req.query)
        msg = (
            "لقد تجاوزت الحد المسموح من الطلبات. يرجى المحاولة بعد دقيقة."
            if lang == "ar"
            else "You have exceeded the rate limit. Please try again after a minute."
        )
        return JSONResponse(status_code=429, content={"error": msg})

    # v13.3: Simplified semaphore logic - use simple await acquire
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
        return await asyncio.wait_for(
            _ask_inner(req),
            timeout=EXTERNAL_CALL_TIMEOUT,
        )
    except asyncio.TimeoutError:
        lang = LanguageDetector.detect(req.query)
        log.warning(f"[ASK] Timed out after {EXTERNAL_CALL_TIMEOUT}s for query: {req.query[:60]}")
        msg = (
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
    history  = req.history or []
    language = LanguageDetector.detect(q)
    intent   = await IntentClassifier.classify(q)

    log.info(
        f"[ASK] query='{q[:80]}' lang={language} intent={intent} "
        f"history_turns={len(history)}"
    )

    # ── Social path ───────────────────────────────────────────────
    if intent == "social":
        reply, model_used = await state.gemini.reply_social(q, language)
        return AskResponse(
            query=q, reply=reply, model_used=model_used, matches=[],
            is_medical=False, found_in_database=False, low_confidence=False,
            language=language, disclaimer=MEDICAL_DISCLAIMER,
        )

    # ── Medical path ──────────────────────────────────────────────
    try:
        matches = await state.knowledge_base.search(q, top_k=TOP_K)
    except Exception as exc:
        # v13.4: Graceful fallback - don't return error, continue with empty matches
        log.error(f"[ASK] Pinecone search failed: {exc}")
        matches = []

    # v13.4: Removed query-side safety cleanup
    # System now degrades gracefully by using best available matches
    # No need to clear matches based on keywords since we removed over-aggressive filtering

    ctx = QueryContext(raw_query=q, language=language, matches=matches)

    if not ctx.has_reliable_matches:
        reply = state.prompt_builder.no_data_response(language)
        log.info(
            f"[ASK] No reliable matches above MIN_CONFIDENCE={MIN_CONFIDENCE} "
            f"— best score: {ctx.best_confidence:.4f} "
            f"total_returned: {len(matches)}"
        )
        return AskResponse(
            query=q, reply=reply, model_used="none", matches=[],
            is_medical=True, found_in_database=False, low_confidence=True,
            language=language, disclaimer=MEDICAL_DISCLAIMER,
        )

    try:
        prompt = state.prompt_builder.build(ctx)
        reply, model_used = await state.gemini.generate(prompt)
        log.info(f"[ASK] RAG success — model={model_used} top={ctx.best_confidence:.4f} n={len(matches)}")
        return AskResponse(
            query=q, reply=reply, model_used=model_used,
            matches=[MatchResult(question=m.question, answer=m.answer, confidence=m.confidence, category=m.category) for m in matches],
            is_medical=True, found_in_database=True, low_confidence=False,
            language=language, disclaimer=MEDICAL_DISCLAIMER,
        )
    except Exception as exc:
        # v13.4: Final safety net - if Gemini fails, return safe fallback
        log.error(f"[ASK] Gemini generation failed: {exc}")
        reply = state.prompt_builder.no_data_response(language)
        return AskResponse(
            query=q, reply=reply, model_used="none",
            matches=[MatchResult(question=m.question, answer=m.answer, confidence=m.confidence, category=m.category) for m in matches],
            is_medical=True, found_in_database=False, low_confidence=True,
            language=language, disclaimer=MEDICAL_DISCLAIMER,
        )


@app.post("/analyze-image", response_model=ImageAnalysisResponse)
async def analyze_image(file: UploadFile = File(...)) -> JSONResponse:
    """Medical image analysis via Gemini Vision — offloaded to threadpool."""

    if file.content_type not in ALLOWED_IMAGE_TYPES:
        log.warning(f"[IMAGE] Rejected unsupported type: {file.content_type}")
        return JSONResponse(status_code=400, content={
            "status": "error",
            "analysis": (
                f"نوع الملف '{file.content_type}' غير مدعوم. "
                "الأنواع المقبولة: JPEG, PNG, WEBP, HEIC, HEIF."
            ),
            "model_used": "none", "disclaimer": MEDICAL_DISCLAIMER,
        })

    try:
        image_bytes = await file.read()
    except Exception as exc:
        log.error(f"[IMAGE] Failed to read '{file.filename}': {exc}")
        return JSONResponse(status_code=400, content={
            "status": "error",
            "analysis": "فشل في قراءة الملف. تأكد من أن الصورة غير تالفة وحاول مرة أخرى.",
            "model_used": "none", "disclaimer": MEDICAL_DISCLAIMER,
        })

    if len(image_bytes) == 0:
        return JSONResponse(status_code=400, content={
            "status": "error", "analysis": "الملف المرفوع فارغ. يرجى رفع صورة صحيحة.",
            "model_used": "none", "disclaimer": MEDICAL_DISCLAIMER,
        })

    if len(image_bytes) > MAX_IMAGE_BYTES:
        return JSONResponse(status_code=413, content={
            "status": "error",
            "analysis": (
                f"حجم الصورة يتجاوز الحد المسموح به ({MAX_IMAGE_MB}MB). "
                "يرجى ضغط الصورة وإعادة المحاولة."
            ),
            "model_used": "none", "disclaimer": MEDICAL_DISCLAIMER,
        })

    size_kb = len(image_bytes) / 1024
    log.info(f"[IMAGE] Processing: '{file.filename}' {size_kb:.1f}KB {file.content_type}")

    try:
        status, analysis, model_used = await asyncio.wait_for(
            state.gemini.analyze_image(image_bytes, file.content_type),
            timeout=EXTERNAL_CALL_TIMEOUT,
        )
    except asyncio.TimeoutError:
        log.warning("[IMAGE] Analysis timed out.")
        return JSONResponse(status_code=504, content={
            "status": "error",
            "analysis": "انتهت مهلة تحليل الصورة. يرجى المحاولة مرة أخرى.",
            "model_used": "none", "disclaimer": MEDICAL_DISCLAIMER,
        })

    http_status = 503 if status == "error" else 200
    log.info(f"[IMAGE] Done — status={status} model={model_used}")

    return JSONResponse(status_code=http_status, content={
        "status": status, "analysis": analysis,
        "model_used": model_used, "disclaimer": MEDICAL_DISCLAIMER,
    })