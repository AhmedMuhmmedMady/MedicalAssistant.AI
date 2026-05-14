"""
╔══════════════════════════════════════════════════════════════════╗
║          SILA — Medical AI Assistant  v14.2                     ║
║          Railway-Hardened · 384-dim · Local Embeddings Only     ║
║                                                                  ║
║  Stack : FastAPI + Pinecone v3 + Gemini (google-genai SDK)      ║
║  Mode  : Strict RAG (local 384-dim) · Social chat               ║
║                                                                  ║
║  🎯 v14.2 Changes:                                              ║
║  • MIN_CONFIDENCE: 0.45 → 0.60                                  ║
║  • RAG_STRONG gap: +0.10 → +0.15 (score >= 0.75)                ║
║  • ✅ Added keyword-overlap relevance guard                     ║
║  • ✅ RAG_WEAK requires relevance_ok                            ║
║  • ✅ GEMINI_ONLY gives real medical answers                    ║
║  • ✅ Better category consistency tracking                      ║
║  • ✅ Improved error handling & logging                         ║
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
from collections import Counter, defaultdict
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
# Environment Configuration
# ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY", "")
PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY", "")
INDEX_NAME         = os.getenv("PINECONE_INDEX", "medical-index-arabicdata")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "")

MIN_CONFIDENCE   = float(os.getenv("SCORE_THRESHOLD", "0.60"))
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "500"))
MAX_IMAGE_MB     = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
MAX_IMAGE_BYTES  = MAX_IMAGE_MB * 1024 * 1024

TOP_K        = int(os.getenv("TOP_K", "7"))
MAX_RETRIES  = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY  = float(os.getenv("RETRY_DELAY", "1.5"))

EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_DIM   = 384  # all-MiniLM-L6-v2 fixed output dimension

MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "20"))
EXTERNAL_CALL_TIMEOUT   = int(os.getenv("EXTERNAL_CALL_TIMEOUT", "25"))

# Validation
if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY is not set.")
if not PINECONE_API_KEY:
    raise RuntimeError("❌ PINECONE_API_KEY is not set.")

# ──────────────────────────────────────────────────────────────────
# Logging Configuration
# ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("sila")

# ──────────────────────────────────────────────────────────────────
# Concurrency Semaphore
# ──────────────────────────────────────────────────────────────────
_request_semaphore: Optional[asyncio.Semaphore] = None

# ──────────────────────────────────────────────────────────────────
# Rate Limiter — In-Memory, Thread-Safe
# ──────────────────────────────────────────────────────────────────
_rate_limit_store: Dict[str, list] = defaultdict(list)
_rate_limit_lock  = threading.Lock()
_RATE_LIMIT_REQUESTS = 10
_RATE_LIMIT_WINDOW   = 60  # seconds


def _check_rate_limit(ip: str) -> bool:
    """Check if IP has exceeded rate limit (10 req/min)."""
    now = time.time()
    with _rate_limit_lock:
        # Remove old entries outside the window
        _rate_limit_store[ip] = [
            ts for ts in _rate_limit_store[ip] if now - ts < _RATE_LIMIT_WINDOW
        ]
        # Check and add new request
        if len(_rate_limit_store[ip]) < _RATE_LIMIT_REQUESTS:
            _rate_limit_store[ip].append(now)
            return True
    return False


# ──────────────────────────────────────────────────────────────────
# Lazy SDK: Pinecone v3
# ──────────────────────────────────────────────────────────────────
_pinecone_index = None
_pinecone_lock  = threading.Lock()


def _init_pinecone():
    global _pinecone_index
    if _pinecone_index is None:
        with _pinecone_lock:
            if _pinecone_index is None:
                log.info("🔄 Initialising Pinecone v3…")
                try:
                    from pinecone import Pinecone as _PC
                    _pinecone_index = _PC(api_key=PINECONE_API_KEY).Index(INDEX_NAME)
                    log.info(f"✅ Pinecone ready — index: {INDEX_NAME}")
                except Exception as exc:
                    log.error(f"❌ Pinecone init failed: {exc}")
                    raise
    return _pinecone_index


async def get_index():
    if _pinecone_index is None:
        await asyncio.to_thread(_init_pinecone)
    return _pinecone_index


# ──────────────────────────────────────────────────────────────────
# Lazy SDK: Google Gemini
# ──────────────────────────────────────────────────────────────────
_gemini_client = None
_gemini_lock   = threading.Lock()


def _init_gemini():
    global _gemini_client
    if _gemini_client is None:
        with _gemini_lock:
            if _gemini_client is None:
                log.info("🔄 Initialising Gemini client…")
                try:
                    from google import genai as _genai
                    _gemini_client = _genai.Client(api_key=GEMINI_API_KEY)
                    log.info("✅ Gemini client ready.")
                except Exception as exc:
                    log.error(f"❌ Gemini init failed: {exc}")
                    raise
    return _gemini_client


def _get_gemini_sync():
    """Get Gemini client (initialize if needed)."""
    return _init_gemini()


def _gemini_types():
    """Import Gemini types module."""
    from google.genai import types
    return types


# ──────────────────────────────────────────────────────────────────
# Lazy SDK: SentenceTransformer (CPU-only, 384-dim)
# ──────────────────────────────────────────────────────────────────
_st_model: Any = None
_st_lock        = threading.Lock()
_st_load_error: Optional[str] = None


def _load_and_encode_sync(text: str) -> List[float]:
    """Load SentenceTransformer model and encode text to 384-dim vector."""
    global _st_model, _st_load_error

    if _st_model is None:
        with _st_lock:
            if _st_model is None:
                if _st_load_error:
                    raise RuntimeError(f"❌ Embedder previously failed: {_st_load_error}")

                try:
                    log.info(f"🔄 Lazy-loading SentenceTransformer: {EMBED_MODEL}")
                    t0 = time.perf_counter()

                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer(EMBED_MODEL, device="cpu")

                    # Optimize PyTorch if available
                    try:
                        import torch as _torch
                        _torch.set_num_threads(1)
                        _torch.set_num_interop_threads(1)
                        _torch.set_grad_enabled(False)
                    except ImportError:
                        pass

                    elapsed = time.perf_counter() - t0
                    log.info(f"✅ SentenceTransformer ready in {elapsed:.2f}s")
                    _st_model = model

                except Exception as exc:
                    _st_load_error = str(exc)
                    log.error(f"❌ Failed to load SentenceTransformer: {exc}")
                    raise

    # Encode text
    try:
        vec: List[float] = _st_model.encode(
            text,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).tolist()

        if len(vec) != EMBED_DIM:
            raise RuntimeError(
                f"❌ Embedding dimension mismatch: got {len(vec)}, expected {EMBED_DIM}."
            )
        return vec
    except Exception as exc:
        log.error(f"❌ Encoding failed: {exc}")
        raise


async def _encode_async(text: str) -> List[float]:
    """Async wrapper for encoding."""
    try:
        return await asyncio.to_thread(_load_and_encode_sync, text)
    except Exception as exc:
        raise RuntimeError(f"❌ Embedding unavailable: {exc}") from exc


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
    "⚠️ تنبيه: هذه المعلومات للتوجيه العام فقط ولا تُغني عن استشارة طبيب متخصص."
)

# Medical keywords (expanded for better detection)
MEDICAL_KEYWORDS: frozenset = frozenset({
    # Arabic
    "ألم", "وجع", "مرض", "دواء", "طبيب", "مستشفى", "أعراض", "علاج",
    "صداع", "حمى", "سعال", "ضغط", "سكر", "قلب", "كلى", "معدة",
    "عظام", "جلد", "عين", "أذن", "أنف", "رئة", "كبد", "دم",
    "تعب", "إرهاق", "دوار", "غثيان", "إسهال", "إمساك", "حرقة",
    "الم", "عندي", "عندى", "اشعر", "احس", "اعاني", "يؤلم",
    "بوجعني", "بتوجعني", "حاسس", "حاسه", "حبوب", "طفح", "حكة",
    "عملية", "جراحة", "منظار", "تحليل", "أشعة", "نتيجة", "تقرير",
    "برد", "انفلونزا", "رشح", "زكام", "كحة", "بلغم", "حرارة",
    # English
    "pain", "ache", "fever", "cough", "headache", "nausea", "dizzy",
    "vomit", "diarrhea", "symptom", "disease", "doctor", "hospital",
    "medicine", "drug", "blood", "heart", "lung", "kidney", "liver",
    "diabetes", "pressure", "infection", "allergy", "rash", "swelling",
    "fatigue", "tired", "breathe", "chest", "stomach", "throat",
    "surgery", "scan", "test", "result", "report", "prescription",
    "cold", "flu", "runny", "nose", "sneeze", "congestion",
})

# Low-quality answer patterns (garbage KB responses)
LOW_QUALITY_PATTERNS: frozenset = frozenset({
    "طبيعي",
    "تم الاجابة",
    "كل شيء ممكن",
    "راجع الطبيب",
    "natural",
    "answered",
    "everything possible",
    "consult doctor",
})

# Emergency keywords (require immediate ER attention)
EMERGENCY_KEYWORDS_AR: frozenset = frozenset({
    "ألم صدر", "ضيق تنفس", "نوبة قلبية", "سكتة دماغية", "نزيف شديد",
    "إغماء", "فقدان وعي", "صدمة", "حروق شديدة", "كسر عظم",
    "ألم حاد", "طوارئ", "إسعاف", "علاج فوري", "خطر على الحياة",
    "ضربة شمس", "تسمم", "جرح عميق", "نزيف داخلي", "انفجار",
    "ألم بطن حاد", "صعوبة بلع", "خدر", "شلل", "تشنج",
    "انتحار", "أفكار انتحارية", "إيذاء النفس",
})

EMERGENCY_KEYWORDS_EN: frozenset = frozenset({
    "chest pain", "difficulty breathing", "heart attack", "stroke", "severe bleeding",
    "fainting", "loss of consciousness", "shock", "severe burns", "broken bone",
    "severe pain", "emergency", "ambulance", "immediate treatment", "life threatening",
    "heat stroke", "poisoning", "deep wound", "internal bleeding", "explosion",
    "severe abdominal pain", "difficulty swallowing", "numbness", "paralysis", "seizure",
    "suicide", "suicidal thoughts", "self harm",
})

# ──────────────────────────────────────────────────────────────────
# Domain Models
# ──────────────────────────────────────────────────────────────────

@dataclass
class KnowledgeMatch:
    """A single match from the knowledge base."""
    question: str
    answer: str
    confidence: float
    category: Optional[str] = None

    @property
    def is_reliable(self) -> bool:
        """Check if confidence meets minimum threshold."""
        return self.confidence >= MIN_CONFIDENCE

    @property
    def is_low_quality(self) -> bool:
        """Check if answer contains low-quality patterns."""
        answer_lower = self.answer.lower()
        return any(pattern.lower() in answer_lower for pattern in LOW_QUALITY_PATTERNS)


@dataclass
class QueryContext:
    """Context for a single query."""
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
# Pydantic Request/Response Models
# ──────────────────────────────────────────────────────────────────

class MessageDto(BaseModel):
    """Chat history message."""
    role: str
    content: str


class AskRequest(BaseModel):
    """Request to /ask endpoint."""
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
    """A knowledge base match result."""
    question: str
    answer: str
    confidence: float
    category: Optional[str] = None


class AskResponse(BaseModel):
    """Response from /ask endpoint."""
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
    """Response from /analyze-image endpoint."""
    status: str
    analysis: Optional[str] = None
    model_used: Optional[str] = None
    disclaimer: str = MEDICAL_DISCLAIMER


# ──────────────────────────────────────────────────────────────────
# Language Detection
# ──────────────────────────────────────────────────────────────────

class LanguageDetector:
    """Detect Arabic vs English."""

    @staticmethod
    def detect(text: str) -> str:
        if not text:
            return "ar"
        # Count Arabic Unicode characters
        arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
        ratio = arabic_chars / max(len(text.strip()), 1)
        return "ar" if ratio > 0.25 else "en"


# ──────────────────────────────────────────────────────────────────
# Emergency Detection
# ──────────────────────────────────────────────────────────────────

class EmergencyDetector:
    """Detect emergency medical queries requiring immediate attention."""

    @staticmethod
    def is_emergency(query: str, language: str) -> bool:
        """Check if query contains emergency keywords."""
        q_lower = query.lower()
        
        if language == "ar":
            return any(kw in q_lower for kw in EMERGENCY_KEYWORDS_AR)
        else:
            return any(kw in q_lower for kw in EMERGENCY_KEYWORDS_EN)


# ──────────────────────────────────────────────────────────────────
# Intent Classifier
# ──────────────────────────────────────────────────────────────────

class IntentClassifier:
    """
    Classify query as 'medical' or 'social'.
    
    Strategy:
    1. Check medical keywords first (zero cost)
    2. If no signal, call Gemini flash-lite for classification
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
        """Synchronous classification."""
        q_lower = query.lower()

        # Fast path: keyword check
        if any(kw in q_lower for kw in MEDICAL_KEYWORDS):
            log.info("[Intent] Medical keyword detected")
            return "medical"

        # Slow path: Gemini classification
        try:
            types = _gemini_types()
            resp = _get_gemini_sync().models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=cls._PROMPT.format(query=query),
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=5),
            )
            result = resp.text.strip().lower()
            if "medical" in result:
                log.info("[Intent] Gemini classified as: medical")
                return "medical"
            log.info("[Intent] Gemini classified as: social")
        except Exception as exc:
            log.warning(f"[Intent] Gemini fallback failed: {exc} — defaulting to social")

        return "social"

    @classmethod
    async def classify(cls, query: str) -> str:
        """Async wrapper for classification."""
        return await asyncio.to_thread(cls._classify_sync, query)


# ──────────────────────────────────────────────────────────────────
# Knowledge Base Service
# ──────────────────────────────────────────────────────────────────

class KnowledgeBaseService:
    """RAG pipeline: embedding, Pinecone search, relevance filtering."""

    @staticmethod
    def _category_consistency(matches: List[KnowledgeMatch]) -> float:
        """Calculate category consistency score (0.0-1.0)."""
        if not matches:
            return 0.0
        categories = [m.category for m in matches if m.category]
        if not categories:
            return 0.0
        counts = Counter(categories)
        dominant_count = counts.most_common(1)[0][1]
        return round(dominant_count / len(categories), 2)

    @staticmethod
    def _extract_medical_tokens(text: str) -> set[str]:
        """
        Extract medical tokens from text (Arabic + English support).
        
        Normalizes Arabic letters, removes diacritics, filters stop words,
        and keeps only medically relevant tokens (>= 3 chars).
        """
        # Normalize Arabic letters (أ -> ا, ة -> ه)
        arabic_normalization = str.maketrans({
            'أ': 'ا', 'إ': 'ا', 'آ': 'ا',
            'ة': 'ه',
            'ى': 'ي',
        })
        
        # Remove diacritics and normalize
        clean = re.sub(r"[\u064b-\u065f\u0670]", "", text.lower())
        clean = clean.translate(arabic_normalization)
        
        # Tokenize
        tokens = set(t for t in re.split(r"[\s\W]+", clean) if len(t) >= 3)
        
        # Filter for medical relevance (keep tokens that appear in medical keywords or are longer)
        medical_tokens = set()
        for token in tokens:
            # Keep if it's in medical keywords or is a substantive word
            if any(kw in token or token in kw for kw in MEDICAL_KEYWORDS):
                medical_tokens.add(token)
            elif len(token) >= 4:  # Keep longer tokens as potentially medical
                medical_tokens.add(token)
        
        return medical_tokens

    @staticmethod
    def _relevance_ok(query: str, matches: List[KnowledgeMatch], min_overlap: int = 1) -> bool:
        """
        Relevance Guard: Medical token overlap check to prevent Pinecone cosine drift.
        
        Uses medical token extraction for more accurate relevance detection.
        Checks that at least `min_overlap` medical query tokens appear in top-3 match texts.
        """
        if not matches:
            return False

        # Extract medical tokens from query
        query_tokens = KnowledgeBaseService._extract_medical_tokens(query)
        
        if not query_tokens:
            # Can't check empty token set — allow through
            log.info("[KB] Relevance: Query has no medical tokens — allowing")
            return True

        # Build token set from top-3 match texts
        combined = " ".join(
            f"{m.question} {m.answer}" for m in matches[:3]
        )
        match_tokens = KnowledgeBaseService._extract_medical_tokens(combined)

        # Check overlap
        overlap = len(query_tokens & match_tokens)
        ok = overlap >= min_overlap

        log.info(
            f"[KB] Relevance: query_tokens={len(query_tokens)} match_tokens={len(match_tokens)} "
            f"overlap={overlap} min_overlap={min_overlap} result={'✅' if ok else '❌'}"
        )

        return ok

    async def search(self, query: str, top_k: int = TOP_K) -> List[KnowledgeMatch]:
        """Search knowledge base and return top matches."""
        log.info(f"[KB] Searching: '{query[:80]}'")

        # Step 1: Encode query
        try:
            vector = await _encode_async(query)
        except Exception as exc:
            log.error(f"[KB] Encoding failed: {exc}")
            return []

        # Step 2: Get Pinecone index
        index = await get_index()

        # Step 3: Query with retries
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

                # Log scores
                raw_scores = [
                    round(float(m.score), 4)
                    for m in results.matches
                    if m.score is not None
                ]
                log.info(
                    f"[KB] Scores: {raw_scores} | threshold={MIN_CONFIDENCE} | count={len(results.matches)}"
                )

                return self._parse_matches(results)

            except Exception as exc:
                last_exc = exc
                log.warning(f"[KB] Attempt {attempt}/{MAX_RETRIES} failed: {exc}")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY * attempt)

        log.error(f"[KB] All {MAX_RETRIES} retries exhausted")
        return []

    @staticmethod
    def _relevance_ok(query: str, matches: List[KnowledgeMatch], min_overlap: int = 1) -> bool:
        """
        Relevance Guard: Keyword overlap check to prevent Pinecone cosine drift.
        
        Checks that at least `min_overlap` query tokens appear in top-3 match texts.
        
        Why? Pinecone similarity can give high scores for off-topic results
        (e.g., "heart disease" when searching for "headache pain").
        """
        if not matches:
            return False

        # Normalize and tokenize query
        q_clean = re.sub(r"[\u064b-\u065f]", "", query.lower())  # Remove Arabic diacritics
        q_tokens = set(t for t in re.split(r"[\s\W]+", q_clean) if len(t) >= 3)

        if not q_tokens:
            # Can't check empty token set — allow through
            log.info("[KB] Relevance: Query too short/empty — allowing")
            return True

        # Build token set from top-3 match texts
        combined = " ".join(
            f"{m.question} {m.answer}" for m in matches[:3]
        ).lower()
        combined_clean = re.sub(r"[\u064b-\u065f]", "", combined)
        match_tokens = set(t for t in re.split(r"[\s\W]+", combined_clean) if len(t) >= 3)

        # Check overlap
        overlap = len(q_tokens & match_tokens)
        ok = overlap >= min_overlap

        log.info(
            f"[KB] Relevance guard: query_tokens={len(q_tokens)} "
            f"match_tokens={len(match_tokens)} overlap={overlap} min_overlap={min_overlap} "
            f"result={'✅' if ok else '❌'}"
        )

        return ok

    @staticmethod
    def _parse_matches(results: Any) -> List[KnowledgeMatch]:
        """Parse Pinecone results into KnowledgeMatch objects."""
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

        log.info(f"[KB] Parsed: {len(matches)} kept, {filtered} filtered")
        return matches


# ──────────────────────────────────────────────────────────────────
# Prompt Builder
# ──────────────────────────────────────────────────────────────────

class PromptBuilder:
    """Build prompts for RAG_STRONG, RAG_WEAK, GEMINI_ONLY modes."""

    _SEP = "━" * 50

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
        "Tone: warm, calm, and professionally precise.\n\n"
        "Strict rules — no exceptions:\n"
        "1. Use ONLY the information provided in the knowledge base context below.\n"
        "2. NEVER use external knowledge or personal assumptions.\n"
        "3. If the data is insufficient, clearly state: "
        "'My knowledge is limited on this. Please consult a specialist.'\n"
        "4. NEVER provide a definitive diagnosis — suggest possibilities only.\n"
        "5. Flag any warning signs that require urgent care.\n"
        "6. Always close by recommending a specialist consultation.\n"
        "7. Respond in clear, professional English."
    )

    _GEMINI_ONLY_AR = (
        "أنت 'سيلا'، مساعد طبي ذكي وموثوق.\n"
        "أسلوبك: دافئ واحترافي، كأنك طبيب خبير يشرح لمريضه بصدق واهتمام.\n\n"
        "لم يتم العثور على معلومات مطابقة في قاعدة البيانات الطبية لهذا الاستفسار.\n"
        "استخدم معرفتك الطبية العامة الموثوقة للإجابة بشكل مفيد وشامل.\n\n"
        "قواعد صارمة:\n"
        "١. قدّم إجابة طبية مفيدة وحقيقية بناءً على معرفتك الطبية العامة.\n"
        "٢. لا تُقدم تشخيصاً نهائياً أبداً — قدّم احتمالات وأسباباً محتملة.\n"
        "٣. اذكر علامات الخطر التي تستدعي التدخل العاجل إن وُجدت.\n"
        "٤. اختم دائماً بالتوصية بمراجعة طبيب متخصص.\n"
        "٥. لغة الإجابة: عربية واضحة ومفهومة.\n"
        "٦. لا تكرر عبارة 'معلوماتي محدودة' — قدّم قيمة طبية حقيقية."
    )

    _GEMINI_ONLY_EN = (
        "You are 'Sila', a trusted and empathetic medical AI assistant.\n"
        "Tone: warm, calm, and professionally precise.\n\n"
        "No matching records were found in the medical knowledge base for this query.\n"
        "Use your reliable general medical knowledge to provide a genuinely helpful response.\n\n"
        "Strict rules:\n"
        "1. Give a real, helpful medical answer based on your general medical knowledge.\n"
        "2. NEVER provide a definitive diagnosis — suggest possibilities and likely causes.\n"
        "3. Flag any warning signs that require urgent care.\n"
        "4. Always close by recommending a specialist consultation.\n"
        "5. Do NOT repeatedly say 'my knowledge is limited' — provide genuine medical value."
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
        "→ List if present, or state 'No critical warning signs identified'"
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

    def _build_context_block(
        self,
        matches: List[KnowledgeMatch],
        language: str,
    ) -> str:
        """Format knowledge base matches into readable context block."""
        sep = self._SEP
        parts = []

        for i, m in enumerate(matches, start=1):
            reliability = (
                ("✅ موثوق" if m.is_reliable else "⚠️ ثقة منخفضة")
                if language == "ar"
                else ("✅ Reliable" if m.is_reliable else "⚠️ Low confidence")
            )
            parts.append(
                f"[{i}] {reliability} — Score: {m.confidence:.0%}\n"
                f"[Specialty: {m.category or 'General'}]\n"
                f"Q: {m.question}\n"
                f"A: {m.answer}"
            )

        return f"\n\n{sep}\n".join(parts)

    def build(self, ctx: QueryContext) -> str:
        """Build RAG_STRONG prompt (strict RAG mode)."""
        lang      = ctx.language
        system    = self._SYSTEM_AR    if lang == "ar" else self._SYSTEM_EN
        structure = self._STRUCTURE_AR if lang == "ar" else self._STRUCTURE_EN
        sep       = self._SEP

        context_block  = self._build_context_block(ctx.matches, lang)
        label_context  = "📋 قاعدة المعرفة الطبية:"       if lang == "ar" else "📋 Medical Knowledge Base:"
        label_question = "🧑‍⚕️ سؤال المريض:"              if lang == "ar" else "🧑‍⚕️ Patient Question:"
        label_answer   = "الإجابة:"                        if lang == "ar" else "Answer:"

        return (
            f"{system}\n\n{sep}\n"
            f"{label_context}\n\n{context_block}\n\n{sep}\n"
            f"{label_question}\n{ctx.raw_query}\n\n"
            f"{structure}\n\n{label_answer}"
        )

    def build_rag_weak(self, ctx: QueryContext) -> str:
        """Build RAG_WEAK prompt (hints only, Gemini can extend)."""
        lang      = ctx.language
        system    = self._SYSTEM_AR    if lang == "ar" else self._SYSTEM_EN
        structure = self._STRUCTURE_AR if lang == "ar" else self._STRUCTURE_EN
        sep       = self._SEP

        weak_warning = (
            "⚠️ ملاحظة: السياق المسترجع قد يحتوي على معلومات طبية ضعيفة أو غير دقيقة. "
            "استخدمه كإشارات داعمة فقط، واعتمد على معرفتك الطبية العامة."
            if lang == "ar"
            else "⚠️ Note: Retrieved context may contain weak or inaccurate information. "
            "Use it only as supporting hints and rely on your general medical knowledge."
        )

        context_block  = self._build_context_block(ctx.matches, lang)
        label_context  = (
            "📋 قاعدة المعرفة الطبية (إشارات داعمة):"
            if lang == "ar"
            else "📋 Medical Knowledge Base (supporting hints):"
        )
        label_question = "🧑‍⚕️ سؤال المريض:" if lang == "ar" else "🧑‍⚕️ Patient Question:"
        label_answer   = "الإجابة:"            if lang == "ar" else "Answer:"

        return (
            f"{system}\n\n{weak_warning}\n\n{sep}\n"
            f"{label_context}\n\n{context_block}\n\n{sep}\n"
            f"{label_question}\n{ctx.raw_query}\n\n"
            f"{structure}\n\n{label_answer}"
        )

    def build_gemini_only(self, query: str, language: str) -> str:
        """Build GEMINI_ONLY prompt (pure Gemini reasoning, no RAG)."""
        system         = self._GEMINI_ONLY_AR if language == "ar" else self._GEMINI_ONLY_EN
        structure      = self._STRUCTURE_AR   if language == "ar" else self._STRUCTURE_EN
        label_question = "🧑‍⚕️ سؤال المريض:"  if language == "ar" else "🧑‍⚕️ Patient Question:"
        label_answer   = "الإجابة:"            if language == "ar" else "Answer:"
        sep            = self._SEP

        return (
            f"{system}\n\n{sep}\n"
            f"{label_question}\n{query}\n\n"
            f"{structure}\n\n{label_answer}"
        )

    def no_data_response(self, language: str) -> str:
        """Fallback response when nothing works."""
        return self._NO_DATA_AR if language == "ar" else self._NO_DATA_EN


# ──────────────────────────────────────────────────────────────────
# LRU Cache — Thread-Safe, Bounded (200 entries)
# ──────────────────────────────────────────────────────────────────

class _BoundedLRU:
    """Simple LRU cache for prompt responses."""

    def __init__(self, maxsize: int = 200):
        self._cache: collections.OrderedDict = collections.OrderedDict()
        self._max  = maxsize
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


# ──────────────────────────────────────────────────────────────────
# Gemini Service (Text + Vision)
# ──────────────────────────────────────────────────────────────────

class GeminiService:
    """Wrapper for Gemini API calls (text generation + image analysis)."""

    def __init__(self) -> None:
        self._cache = _BoundedLRU(200)

    @staticmethod
    def _make_config(temperature: float = 0.2, max_tokens: int = 2048):
        """Create Gemini generation config."""
        types = _gemini_types()
        return types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    # ── Social Replies ─────────────────────────────────────────────

    def _reply_social_sync(self, query: str, language: str) -> Tuple[str, str]:
        """Generate social chat response."""
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
                log.info(f"[Social] {model_name}")
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
                log.warning(f"[Social] {model_name} failed: {exc}")

        # Fallback
        fallback = (
            "Hello! 😊 I'm Sila, your medical AI. How can I help?"
            if language == "en"
            else "أهلاً! 😊 أنا سيلا، مساعدتك الطبية. كيف يمكنني مساعدتك؟"
        )
        return fallback, "fallback"

    async def reply_social(self, query: str, language: str) -> Tuple[str, str]:
        return await asyncio.to_thread(self._reply_social_sync, query, language)

    # ── RAG Generation ─────────────────────────────────────────────

    def _generate_sync(self, prompt: str) -> Tuple[str, str]:
        """Generate medical response with caching."""
        cache_key = hashlib.sha256(prompt.encode()).hexdigest()
        cached = self._cache.get(cache_key)

        if cached:
            log.info("[Gemini] ✅ Cache hit")
            return cached

        for model_name in GEMINI_TEXT_MODELS:
            try:
                log.info(f"[Gemini] Generate — {model_name}")
                resp = _get_gemini_sync().models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=self._make_config(),
                )
                result = (resp.text.strip(), model_name)
                self._cache.put(cache_key, result)
                return result
            except Exception as exc:
                log.warning(f"[Gemini] {model_name} failed: {exc}")

        log.error("[Gemini] ❌ All text models exhausted")
        return (
            "عذراً، حدث خطأ مؤقت في معالجة طلبك. يرجى المحاولة مرة أخرى.",
            "none",
        )

    async def generate(self, prompt: str) -> Tuple[str, str]:
        return await asyncio.to_thread(self._generate_sync, prompt)

    # ── Image Analysis ─────────────────────────────────────────────

    def _analyze_image_sync(
        self, image_bytes: bytes, mime_type: str
    ) -> Tuple[str, str, str]:
        """Analyze medical image and return JSON response."""
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
                    log.info(f"[Vision] {model_name}")
                    resp = _get_gemini_sync().models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=self._make_config(temperature=0.1, max_tokens=4096),
                    )

                    raw   = resp.text.strip()
                    clean = re.sub(
                        r"^\s*```+(?:json)?\s*|\s*```+\s*$", "", raw, flags=re.MULTILINE
                    ).strip()

                    # Extract JSON if it's wrapped
                    brace = clean.find("{")
                    if brace > 0:
                        clean = clean[brace:]

                    try:
                        parsed   = json.loads(clean)
                        status   = str(parsed.get("status", "success"))
                        analysis = parsed.get("analysis", "")

                        if not isinstance(analysis, str):
                            analysis = json.dumps(analysis, ensure_ascii=False, indent=2)

                        log.info(f"[Vision] ✅ {model_name} status={status}")
                        return status, analysis.strip(), model_name

                    except json.JSONDecodeError:
                        # Best-effort JSON extraction
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
                                                parsed   = json.loads(clean[start:i + 1])
                                                status   = str(parsed.get("status", "success"))
                                                analysis = parsed.get("analysis", "")
                                                if not isinstance(analysis, str):
                                                    analysis = json.dumps(
                                                        analysis, ensure_ascii=False
                                                    )
                                                return status, analysis.strip(), model_name
                                            except Exception:
                                                break

                        log.warning(f"[Vision] {model_name} — JSON parse fallback")
                        return "success", raw, model_name

                except Exception as exc:
                    log.warning(f"[Vision] {model_name} variant failed: {exc}")
                    continue

        log.error("[Vision] ❌ All models exhausted")
        return (
            "error",
            "تعذّر تحليل الصورة مؤقتاً. يرجى المحاولة مرة أخرى لاحقاً.",
            "none",
        )

    async def analyze_image(
        self, image_bytes: bytes, mime_type: str
    ) -> Tuple[str, str, str]:
        return await asyncio.to_thread(self._analyze_image_sync, image_bytes, mime_type)

    @property
    def cache_size(self) -> int:
        return len(self._cache)


# ──────────────────────────────────────────────────────────────────
# Application State
# ──────────────────────────────────────────────────────────────────

class AppState:
    """Singleton for app services."""
    knowledge_base: Optional[KnowledgeBaseService] = None
    gemini: Optional[GeminiService]                = None
    prompt_builder: Optional[PromptBuilder]        = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown handler."""
    global _request_semaphore
    log.info("🚀 SILA v14.2 starting — zero-SDK boot")

    _request_semaphore   = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    state.knowledge_base = KnowledgeBaseService()
    state.gemini         = GeminiService()
    state.prompt_builder = PromptBuilder()

    log.info(
        f"✅ Boot complete:\n"
        f"   embed={EMBED_MODEL} (dim={EMBED_DIM})\n"
        f"   min_confidence={MIN_CONFIDENCE}\n"
        f"   rag_strong_threshold={MIN_CONFIDENCE + 0.15}\n"
        f"   concurrency={MAX_CONCURRENT_REQUESTS}\n"
        f"   timeout={EXTERNAL_CALL_TIMEOUT}s\n"
        f"   index={INDEX_NAME}"
    )

    yield

    log.info("🛑 SILA shutting down")


# ──────────────────────────────────────────────────────────────────
# FastAPI Application
# ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Sila — Medical AI Assistant v14.2",
    description="مساعد طبي ذكي | Strict RAG + Gemini Vision + Arabic & English",
    version="14.2.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.error(f"❌ Unhandled {type(exc).__name__} on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "An unexpected error occurred. Please try again later."},
    )


# ──────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Root endpoint — info."""
    return {
        "name": "Sila — Medical AI Assistant",
        "version": "14.2.0",
        "status": "running ✅",
        "endpoints": ["/ask", "/analyze-image", "/health", "/docs"],
        "docs": "https://sila-medical.docs.io",
    }


@app.get("/health")
def health():
    """Health check — ultra-fast, no SDK calls."""
    return {
        "status": "ok",
        "version": "14.2.0",
        "uptime": "healthy",
        "config": {
            "embed_model": EMBED_MODEL,
            "embed_dim": EMBED_DIM,
            "index": INDEX_NAME,
            "min_confidence": MIN_CONFIDENCE,
            "top_k": TOP_K,
            "rag_strong_threshold": MIN_CONFIDENCE + 0.15,
        }
    }


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, request: Request) -> AskResponse:
    """Main medical query endpoint."""
    if _request_semaphore is None:
        return JSONResponse(status_code=503, content={"error": "Server not ready yet"})

    # Rate limiting
    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        lang = LanguageDetector.detect(req.query)
        msg = (
            "لقد تجاوزت الحد المسموح من الطلبات. يرجى المحاولة بعد دقيقة."
            if lang == "ar"
            else "You have exceeded the rate limit. Please try again after a minute."
        )
        log.warning(f"[ASK] Rate limit exceeded for {client_ip}")
        return JSONResponse(status_code=429, content={"error": msg})

    # Semaphore
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
        log.warning(f"[ASK] ❌ Timeout after {EXTERNAL_CALL_TIMEOUT}s")
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
    """Inner logic for /ask endpoint."""
    q        = req.query
    language = LanguageDetector.detect(q)
    intent   = await IntentClassifier.classify(q)

    log.info(f"[ASK] Query: '{q[:80]}' | lang={language} | intent={intent}")

    # ── SOCIAL INTENT ─────────────────────────────────────────────
    if intent == "social":
        reply, model_used = await state.gemini.reply_social(q, language)
        log.info(f"[ASK] ✅ Social response — {model_used}")
        return AskResponse(
            query=q, reply=reply, model_used=model_used, matches=[],
            is_medical=False, found_in_database=False, low_confidence=False,
            language=language, disclaimer=MEDICAL_DISCLAIMER,
        )

    # ── EMERGENCY DETECTION ───────────────────────────────────────
    is_emergency = EmergencyDetector.is_emergency(q, language)
    if is_emergency:
        log.warning(f"[ASK] 🚨 EMERGENCY_QUERY=True")

    # ── MEDICAL INTENT: RETRIEVE FROM KB ──────────────────────────
    try:
        matches = await state.knowledge_base.search(q, top_k=TOP_K)
    except Exception as exc:
        log.error(f"[ASK] KB search failed: {exc}")
        matches = []

    # ── 3-TIER DECISION ENGINE ────────────────────────────────────
    top_score            = matches[0].confidence if matches else 0.0
    category_consistency = KnowledgeBaseService._category_consistency(matches)
    relevance_ok         = KnowledgeBaseService._relevance_ok(q, matches)

    # Check for low-quality matches (garbage KB responses)
    low_quality_count = sum(1 for m in matches if m.is_low_quality)
    low_quality_ratio = low_quality_count / len(matches) if matches else 0.0

    # Determine RAG mode
    if not matches or top_score < MIN_CONFIDENCE:
        rag_mode = "GEMINI_ONLY"
        reason   = f"no_matches_or_low_score(score={top_score:.3f})"
    elif not relevance_ok:
        rag_mode = "GEMINI_ONLY"
        reason   = "relevance_guard_failed"
    elif low_quality_ratio > 0.6:
        # More than 60% low-quality matches → downgrade to GEMINI_ONLY
        rag_mode = "GEMINI_ONLY"
        reason   = f"low_quality_ratio={low_quality_ratio:.2f}>0.6"
    elif top_score >= MIN_CONFIDENCE + 0.15 and category_consistency >= 0.7:
        rag_mode = "RAG_STRONG"
        reason   = f"high_confidence_{top_score:.3f}_consistent_categories"
    elif top_score >= MIN_CONFIDENCE:
        rag_mode = "RAG_WEAK"
        reason   = f"medium_confidence_{top_score:.3f}"
    else:
        rag_mode = "GEMINI_ONLY"
        reason   = "fallback"

    # Get category distribution
    category_dist: Dict[str, int] = {}
    if matches:
        categories = [m.category for m in matches if m.category]
        category_dist = dict(Counter(categories))

    log.info(
        f"[ASK] 🎯 {rag_mode} — {reason} "
        f"| top_score={top_score:.4f} | matches={len(matches)} "
        f"| relevance_ok={relevance_ok} | category_consistency={category_consistency:.2f} "
        f"| low_quality_ratio={low_quality_ratio:.2f}"
    )

    # Convert to response format
    match_results = [
        MatchResult(
            question=m.question,
            answer=m.answer,
            confidence=m.confidence,
            category=m.category,
        )
        for m in matches
    ]

    # ── RAG_STRONG ────────────────────────────────────────────────
    if rag_mode == "RAG_STRONG":
        ctx = QueryContext(raw_query=q, language=language, matches=matches)
        try:
            prompt            = state.prompt_builder.build(ctx)
            reply, model_used = await state.gemini.generate(prompt)
            log.info(f"[ASK] ✅ RAG_STRONG — {model_used}")
            return AskResponse(
                query=q, reply=reply, model_used=model_used,
                matches=match_results,
                is_medical=True, found_in_database=True, low_confidence=False,
                language=language, disclaimer=MEDICAL_DISCLAIMER,
            )
        except Exception as exc:
            log.error(f"[ASK] RAG_STRONG Gemini failed: {exc} — fallback to GEMINI_ONLY")
            rag_mode = "GEMINI_ONLY"

    # ── RAG_WEAK ──────────────────────────────────────────────────
    if rag_mode == "RAG_WEAK":
        ctx = QueryContext(raw_query=q, language=language, matches=matches)
        try:
            prompt            = state.prompt_builder.build_rag_weak(ctx)
            reply, model_used = await state.gemini.generate(prompt)
            log.info(f"[ASK] ✅ RAG_WEAK — {model_used}")
            return AskResponse(
                query=q, reply=reply, model_used=model_used,
                matches=match_results,
                is_medical=True, found_in_database=True, low_confidence=True,
                language=language, disclaimer=MEDICAL_DISCLAIMER,
            )
        except Exception as exc:
            log.error(f"[ASK] RAG_WEAK Gemini failed: {exc} — fallback to GEMINI_ONLY")
            rag_mode = "GEMINI_ONLY"

    # ── GEMINI_ONLY ───────────────────────────────────────────────
    try:
        prompt            = state.prompt_builder.build_gemini_only(q, language)
        reply, model_used = await state.gemini.generate(prompt)
        log.info(f"[ASK] ✅ GEMINI_ONLY — {model_used}")
        return AskResponse(
            query=q, reply=reply, model_used=model_used,
            matches=match_results,
            is_medical=True, found_in_database=False, low_confidence=True,
            language=language, disclaimer=MEDICAL_DISCLAIMER,
        )
    except Exception as exc:
        log.error(f"[ASK] ❌ GEMINI_ONLY failed: {exc}")
        reply = state.prompt_builder.no_data_response(language)
        return AskResponse(
            query=q, reply=reply, model_used="none",
            matches=match_results,
            is_medical=True, found_in_database=False, low_confidence=True,
            language=language, disclaimer=MEDICAL_DISCLAIMER,
        )


@app.post("/analyze-image", response_model=ImageAnalysisResponse)
async def analyze_image(file: UploadFile = File(...)) -> JSONResponse:
    """Image analysis endpoint — medical images only."""

    # Validate MIME type
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        log.warning(f"[IMAGE] ❌ Rejected MIME type: {file.content_type}")
        return JSONResponse(status_code=400, content={
            "status": "error",
            "analysis": (
                f"نوع الملف '{file.content_type}' غير مدعوم. "
                "الأنواع المقبولة: JPEG, PNG, WEBP, HEIC, HEIF."
            ),
            "model_used": "none",
            "disclaimer": MEDICAL_DISCLAIMER,
        })

    # Read file
    try:
        image_bytes = await file.read()
    except Exception as exc:
        log.error(f"[IMAGE] ❌ Failed to read '{file.filename}': {exc}")
        return JSONResponse(status_code=400, content={
            "status": "error",
            "analysis": "فشل في قراءة الملف. تأكد من أن الصورة غير تالفة وحاول مرة أخرى.",
            "model_used": "none",
            "disclaimer": MEDICAL_DISCLAIMER,
        })

    # Validate not empty
    if not image_bytes:
        return JSONResponse(status_code=400, content={
            "status": "error",
            "analysis": "الملف المرفوع فارغ. يرجى رفع صورة صحيحة.",
            "model_used": "none",
            "disclaimer": MEDICAL_DISCLAIMER,
        })

    # Validate size
    if len(image_bytes) > MAX_IMAGE_BYTES:
        log.warning(f"[IMAGE] ❌ File too large: {len(image_bytes) / 1024 / 1024:.1f}MB")
        return JSONResponse(status_code=413, content={
            "status": "error",
            "analysis": (
                f"حجم الصورة يتجاوز الحد المسموح به ({MAX_IMAGE_MB}MB). "
                "يرجى ضغط الصورة وإعادة المحاولة."
            ),
            "model_used": "none",
            "disclaimer": MEDICAL_DISCLAIMER,
        })

    log.info(f"[IMAGE] Processing: '{file.filename}' ({len(image_bytes) / 1024:.1f}KB)")

    # Analyze
    try:
        status, analysis, model_used = await asyncio.wait_for(
            state.gemini.analyze_image(image_bytes, file.content_type),
            timeout=EXTERNAL_CALL_TIMEOUT,
        )
    except asyncio.TimeoutError:
        log.warning("[IMAGE] ❌ Timeout")
        return JSONResponse(status_code=504, content={
            "status": "error",
            "analysis": "انتهت مهلة تحليل الصورة. يرجى المحاولة مرة أخرى.",
            "model_used": "none",
            "disclaimer": MEDICAL_DISCLAIMER,
        })

    http_status = 200 if status == "success" else 503 if status == "error" else 200
    log.info(f"[IMAGE] ✅ Done — status={status} model={model_used}")

    return JSONResponse(status_code=http_status, content={
        "status": status,
        "analysis": analysis,
        "model_used": model_used,
        "disclaimer": MEDICAL_DISCLAIMER,
    })


# ──────────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    log.info("🚀 Starting SILA v14.2 server…")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
    )
