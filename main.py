"""
╔══════════════════════════════════════════════════════════════════╗
║          SILA — Medical AI Assistant  v10.1                     ║
║          Railway-Safe · Memory-Optimized · Lazy Loading         ║
║                                                                  ║
║  Stack : FastAPI + Pinecone v3 + Gemini (google-genai SDK)      ║
║  Mode  : Strict RAG for medical · Social chat for greetings     ║
║  Vision: Medical image analysis via Gemini Vision               ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ──────────────────────────────────────────────────────────────────
# Standard Library
# ──────────────────────────────────────────────────────────────────
import base64
import hashlib
import json
import logging
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, List, Optional

# ──────────────────────────────────────────────────────────────────
# Third-Party
# ──────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
from pinecone import Pinecone as PineconeClient          # v3 safe alias
from pydantic import BaseModel, model_validator
from sentence_transformers import SentenceTransformer    # ✅ FIX 1: added import

# ──────────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────────
load_dotenv()

GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
INDEX_NAME       = os.getenv("PINECONE_INDEX", "sila-medical")
MIN_CONFIDENCE   = float(os.getenv("MIN_CONFIDENCE", "0.55"))
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "500"))
MAX_IMAGE_MB     = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
MAX_IMAGE_BYTES  = MAX_IMAGE_MB * 1024 * 1024
EMBED_MODEL      = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
GEMINI_TIMEOUT   = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "30"))

# ── CPU memory cap — critical for Railway free tier ────────────────
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

# ── Torch threads — only if torch is installed ────────────────────
try:
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
except ImportError:
    pass

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
# Gemini Client — lightweight HTTP client, zero RAM cost
# ──────────────────────────────────────────────────────────────────
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

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
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/heic",
    "image/heif",
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
})


# ──────────────────────────────────────────────────────────────────
# Lazy Embedding Model Singleton                        ✅ FIX 2
# ──────────────────────────────────────────────────────────────────

class _EmbedModelSingleton:
    """
    Thread-safe lazy singleton for SentenceTransformer.

    Loads ONCE on the first call to .get(), reused forever.
    Double-checked locking ensures only one load under concurrency.

    Why lazy:
    - Railway health-check fires immediately after container boot.
    - Loading model at startup spikes RAM before health-check passes
      → Railway kills the container → deploy fails.
    - Lazy loading lets /health respond in <50ms; model loads on the
      first real /ask request.
    """

    _instance: Optional[SentenceTransformer] = None
    _lock = threading.Lock()
    _load_error: Optional[str] = None

    @classmethod
    def get(cls) -> SentenceTransformer:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    if cls._load_error:
                        raise RuntimeError(
                            f"Embedding model failed previously: {cls._load_error}"
                        )
                    try:
                        log.info(f"⏳ Loading embedding model: {EMBED_MODEL}")
                        t0 = time.perf_counter()

                        model = SentenceTransformer(EMBED_MODEL)

                        try:
                            import torch as _torch
                            model = model.to(_torch.device("cpu"))
                        except ImportError:
                            pass

                        elapsed = time.perf_counter() - t0
                        log.info(f"✅ Embedding model ready in {elapsed:.2f}s")
                        cls._instance = model

                    except Exception as exc:
                        cls._load_error = str(exc)
                        log.error(f"❌ Failed to load embedding model: {exc}")
                        raise

        return cls._instance

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._instance is not None


# ──────────────────────────────────────────────────────────────────
# Domain Models
# ──────────────────────────────────────────────────────────────────

@dataclass
class KnowledgeMatch:
    question: str
    answer: str
    confidence: float

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

class AskRequest(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    text: Optional[str] = None
    question: Optional[str] = None

    @property
    def query(self) -> str:
        return (self.text or self.question or "").strip()

    @model_validator(mode="after")
    def validate_query(self) -> "AskRequest":
        q = self.query
        if not q:
            raise ValueError(
                "Request must include a non-empty 'text' or 'question' field."
            )
        if len(q) > MAX_QUERY_LENGTH:
            raise ValueError(f"Query exceeds {MAX_QUERY_LENGTH} characters.")
        return self


class MatchResult(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    question: str
    answer: str
    confidence: float


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
# Intent Classifier
# ──────────────────────────────────────────────────────────────────

class IntentClassifier:
    """
    Layer 1: Gemini LLM — zero local RAM, fast and accurate.
    Layer 2: Keyword fallback when Gemini is unavailable.
    """

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
    def classify(cls, query: str) -> str:
        try:
            resp = gemini_client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=cls._PROMPT.format(query=query),
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=5,
                ),
            )
            result = resp.text.strip().lower()
            if "social" in result:
                return "social"
            if "medical" in result:
                return "medical"
        except Exception as exc:
            log.warning(
                f"IntentClassifier: Gemini unavailable, using keyword fallback — {exc}"
            )

        q_lower = query.lower()
        return "medical" if any(kw in q_lower for kw in MEDICAL_KEYWORDS) else "social"


# ──────────────────────────────────────────────────────────────────
# Knowledge Base Service (Pinecone v3)
# ──────────────────────────────────────────────────────────────────

class KnowledgeBaseService:
    """
    Semantic search via Pinecone v3.
    Embedding model fetched per-call via lazy singleton.
    Includes retry logic for transient Pinecone errors.
    """

    _MAX_RETRIES = 3
    _RETRY_DELAY = 1.0  # seconds, multiplied by attempt number

    def __init__(self, index: Any) -> None:
        self._index = index

    def search(self, query: str, top_k: int = 5) -> List[KnowledgeMatch]:
        model  = _EmbedModelSingleton.get()
        vector = model.encode(query, show_progress_bar=False).tolist()

        last_exc: Optional[Exception] = None
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                results = self._index.query(
                    vector=vector,
                    top_k=top_k,
                    include_metadata=True,
                )
                return self._parse_matches(results)

            except Exception as exc:
                last_exc = exc
                log.warning(
                    f"[KnowledgeBase] Pinecone query attempt "
                    f"{attempt}/{self._MAX_RETRIES} failed: {exc}"
                )
                if attempt < self._MAX_RETRIES:
                    time.sleep(self._RETRY_DELAY * attempt)

        log.error(f"[KnowledgeBase] All Pinecone retries exhausted: {last_exc}")
        raise last_exc  # type: ignore[misc]

    @staticmethod
    def _parse_matches(results: Any) -> List[KnowledgeMatch]:
        matches: List[KnowledgeMatch] = []
        for m in results.matches:
            if m.score < MIN_CONFIDENCE * 0.70:
                continue
            meta = m.metadata or {}
            matches.append(KnowledgeMatch(
                question=meta.get("question", ""),
                answer=meta.get("answer", ""),
                confidence=round(float(m.score), 4),
            ))
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
        "٧. لغة الإجابة: عربية واضحة ومفهومة."
    )

    _SYSTEM_EN = (
        "You are 'Sila', a trusted and empathetic medical AI assistant.\n"
        "Tone: warm, calm, and professionally precise — like a knowledgeable doctor "
        "who takes time to explain clearly.\n\n"
        "Strict rules — no exceptions:\n"
        "1. Use ONLY the information provided in the knowledge base context below.\n"
        "2. NEVER use external knowledge or personal assumptions.\n"
        "3. If the data is insufficient, clearly state: "
        "'My knowledge is limited on this. Please consult a specialist.'\n"
        "4. NEVER provide a definitive diagnosis — suggest possibilities only.\n"
        "5. Flag any warning signs that require urgent care.\n"
        "6. Always close by recommending a specialist consultation."
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
            context_parts.append(
                f"[{i}] {reliability} — Score: {m.confidence:.0%}\n"
                f"Q: {m.question}\n"
                f"A: {m.answer}"
            )

        sep           = "━" * 50
        context_block = f"\n\n{sep}\n".join(context_parts)

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
# Gemini Service
# ──────────────────────────────────────────────────────────────────

class GeminiService:
    """
    Centralizes all Gemini interactions.
    In-memory response cache keyed by prompt hash (capped at 200).
    Falls back through model list on any failure.
    """

    def __init__(self) -> None:
        self._cache: dict[str, tuple[str, str]] = {}

    @staticmethod
    def _text_config(
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    # ── Social replies ─────────────────────────────────────────────

    def reply_social(self, query: str, language: str) -> tuple[str, str]:
        if language == "en":
            system = (
                "You are 'Sila', a friendly and warm medical AI assistant. "
                "Reply naturally in English. Keep it brief (1-2 sentences). "
                "If the topic is not medical, warmly mention that you specialize "
                "in medical consultations and invite them to ask any health-related questions."
            )
        else:
            system = (
                "أنت 'سيلا'، مساعد طبي ذكي وودود.\n"
                "رد بالعربية بشكل طبيعي ودافئ. الرد قصير (جملة أو اتنين بالكثير).\n"
                "لو الموضوع مش طبي، قول بلطف إنك متخصص في الاستشارات الطبية "
                "وادعوه يسأل أي سؤال صحي."
            )

        for model_name in GEMINI_TEXT_MODELS:
            try:
                resp = gemini_client.models.generate_content(
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

    # ── RAG generation ─────────────────────────────────────────────

    def generate(self, prompt: str) -> tuple[str, str]:
        cache_key = hashlib.md5(prompt.encode("utf-8")).hexdigest()
        if cache_key in self._cache:
            log.info("Cache hit — reusing previous response.")
            return self._cache[cache_key]

        for model_name in GEMINI_TEXT_MODELS:
            try:
                log.info(f"RAG generate — trying model: {model_name}")
                resp = gemini_client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=self._text_config(),
                )
                text = resp.text.strip()

                if len(self._cache) < 200:
                    self._cache[cache_key] = (text, model_name)

                return text, model_name

            except Exception as exc:
                log.warning(f"RAG generate — {model_name} failed: {exc}")

        log.error("All text models exhausted.")
        return (
            "عذراً، حدث خطأ مؤقت في معالجة طلبك. يرجى المحاولة مرة أخرى.",
            "none",
        )

    # ── Image analysis ─────────────────────────────────────────────

    def analyze_image(
        self,
        image_bytes: bytes,
        mime_type: str,
    ) -> tuple[str, str, str]:
        """Returns (status, analysis_text, model_name)."""

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
                    resp = gemini_client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=self._text_config(temperature=0.1, max_tokens=4096),
                    )
                    raw   = resp.text.strip()
                    clean = re.sub(
                        r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE
                    ).strip()

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
                    log.warning(f"Vision {model_name}: non-JSON — using raw text.")
                    return "success", raw, model_name

                except Exception as exc:
                    log.warning(f"Vision {model_name} variant failed: {exc}")
                    continue

        log.error("All vision models exhausted.")
        return (
            "error",
            "تعذّر تحليل الصورة مؤقتاً. يرجى المحاولة مرة أخرى لاحقاً.",
            "none",
        )

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
    """
    Lightweight startup — only creates stateless service objects
    and a Pinecone HTTP connection. Zero heavy models loaded here.
    SentenceTransformer loads lazily on the first /ask call.
    """
    log.info("🚀 Sila v10.1 — lightweight boot sequence starting")
    log.info(f"🔌 Connecting to Pinecone index: {INDEX_NAME}")

    pc    = PineconeClient(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    state.knowledge_base = KnowledgeBaseService(index)
    state.gemini         = GeminiService()
    state.prompt_builder = PromptBuilder()

    log.info("✅ Boot complete. Embedding model will load lazily on first /ask request.")
    yield
    log.info("🛑 Sila shutting down.")


# ──────────────────────────────────────────────────────────────────
# FastAPI Application
# ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Sila — Medical AI Assistant",
    description="مساعد طبي ذكي | Strict RAG + Gemini Vision + Arabic & English",
    version="10.1.0",
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
    log.error(
        f"Unhandled exception on {request.url.path}: {type(exc).__name__}: {exc}"
    )
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
        "version":   "10.1.0",
        "status":    "running",
        "endpoints": ["/ask", "/analyze-image", "/health", "/docs"],
    }


@app.get("/health")
def health():
    """Instant health-check — never triggers model loading."""
    return {
        "status":         "ok",
        "version":        "10.1.0",
        "model_loaded":   _EmbedModelSingleton.is_loaded(),
        "embed_model":    EMBED_MODEL,
        "cache_size":     state.gemini.cache_size if state.gemini else 0,
        "min_confidence": MIN_CONFIDENCE,
        "image_analysis": "enabled",
        "index":          INDEX_NAME,
    }


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    """
    Main Q&A endpoint.
    Social  → Gemini direct reply (no local model).
    Medical → lazy embed model → Pinecone → Gemini RAG.
    """
    q        = req.query
    language = LanguageDetector.detect(q)
    intent   = IntentClassifier.classify(q)

    log.info(f"[ASK] query='{q[:80]}' lang={language} intent={intent}")

    # ── Social path ───────────────────────────────────────────────
    if intent == "social":
        reply, model_used = state.gemini.reply_social(q, language)
        return AskResponse(
            query=q,
            reply=reply,
            model_used=model_used,
            matches=[],
            is_medical=False,
            found_in_database=False,
            low_confidence=False,
            language=language,
            disclaimer=MEDICAL_DISCLAIMER,
        )

    # ── Medical path ──────────────────────────────────────────────
    try:
        matches = state.knowledge_base.search(q, top_k=5)
    except Exception as exc:
        log.error(f"[ASK] Pinecone search failed after retries: {exc}")
        fallback_msg = (
            "عذراً، حدث خطأ في البحث. يرجى المحاولة مرة أخرى."
            if language == "ar"
            else "Sorry, search failed. Please try again."
        )
        return AskResponse(
            query=q,
            reply=fallback_msg,
            model_used="none",
            matches=[],
            is_medical=True,
            found_in_database=False,
            low_confidence=True,
            language=language,
            disclaimer=MEDICAL_DISCLAIMER,
        )

    ctx = QueryContext(raw_query=q, language=language, matches=matches)

    # ── Low-confidence path ───────────────────────────────────────
    if not ctx.has_reliable_matches:
        reply = state.prompt_builder.no_data_response(language)
        log.info(f"[ASK] No reliable matches — best score: {ctx.best_confidence:.2f}")
        return AskResponse(
            query=q,
            reply=reply,
            model_used="none",
            matches=[
                MatchResult(
                    question=m.question,
                    answer=m.answer,
                    confidence=m.confidence,
                )
                for m in matches
            ],
            is_medical=True,
            found_in_database=False,
            low_confidence=True,
            language=language,
            disclaimer=MEDICAL_DISCLAIMER,
        )

    # ── RAG generation ────────────────────────────────────────────
    prompt            = state.prompt_builder.build(ctx)
    reply, model_used = state.gemini.generate(prompt)

    log.info(
        f"[ASK] RAG success — model={model_used} "
        f"top_confidence={ctx.best_confidence:.2f} matches={len(matches)}"
    )

    return AskResponse(
        query=q,
        reply=reply,
        model_used=model_used,
        matches=[
            MatchResult(
                question=m.question,
                answer=m.answer,
                confidence=m.confidence,
            )
            for m in matches
        ],
        is_medical=True,
        found_in_database=True,
        low_confidence=False,
        language=language,
        disclaimer=MEDICAL_DISCLAIMER,
    )


@app.post("/analyze-image", response_model=ImageAnalysisResponse)
async def analyze_image(file: UploadFile = File(...)) -> JSONResponse:
    """Medical image analysis via Gemini Vision. No local model."""

    # ── File type validation ──────────────────────────────────────
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        log.warning(f"[IMAGE] Rejected unsupported type: {file.content_type}")
        return JSONResponse(
            status_code=400,
            content={
                "status":     "error",
                "analysis":   (
                    f"نوع الملف '{file.content_type}' غير مدعوم. "
                    "الأنواع المقبولة: JPEG, PNG, WEBP, HEIC, HEIF."
                ),
                "model_used": "none",
                "disclaimer": MEDICAL_DISCLAIMER,
            },
        )

    # ── File read ─────────────────────────────────────────────────
    try:
        image_bytes = await file.read()
    except Exception as exc:
        log.error(f"[IMAGE] Failed to read '{file.filename}': {exc}")
        return JSONResponse(
            status_code=400,
            content={
                "status":     "error",
                "analysis":   "فشل في قراءة الملف. تأكد من أن الصورة غير تالفة وحاول مرة أخرى.",
                "model_used": "none",
                "disclaimer": MEDICAL_DISCLAIMER,
            },
        )

    # ── Size validation ───────────────────────────────────────────
    if len(image_bytes) == 0:
        return JSONResponse(
            status_code=400,
            content={
                "status":     "error",
                "analysis":   "الملف المرفوع فارغ. يرجى رفع صورة صحيحة.",
                "model_used": "none",
                "disclaimer": MEDICAL_DISCLAIMER,
            },
        )

    if len(image_bytes) > MAX_IMAGE_BYTES:
        return JSONResponse(
            status_code=413,
            content={
                "status":     "error",
                "analysis":   (
                    f"حجم الصورة يتجاوز الحد المسموح به ({MAX_IMAGE_MB}MB). "
                    "يرجى ضغط الصورة وإعادة المحاولة."
                ),
                "model_used": "none",
                "disclaimer": MEDICAL_DISCLAIMER,
            },
        )

    size_kb = len(image_bytes) / 1024
    log.info(
        f"[IMAGE] Processing: '{file.filename}' {size_kb:.1f}KB {file.content_type}"
    )

    # ── Vision analysis ───────────────────────────────────────────
    status, analysis, model_used = state.gemini.analyze_image(
        image_bytes, file.content_type
    )

    http_status = 503 if status == "error" else 200
    log.info(f"[IMAGE] Done — status={status} model={model_used}")

    return JSONResponse(
        status_code=http_status,
        content={
            "status":     status,
            "analysis":   analysis,
            "model_used": model_used,
            "disclaimer": MEDICAL_DISCLAIMER,
        },
    )