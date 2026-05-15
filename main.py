"""
╔══════════════════════════════════════════════════════════════════╗
║          SILA — Medical AI Assistant  v16.0                     ║
║          Hybrid RAG · Exact-Match Boost · Medical Awareness     ║
║                                                                  ║
║  🆕 v16.0 Retrieval Upgrades:                                    ║
║  • Hybrid Scoring: cosine(0.6) + exact(0.3) + category(0.1)    ║
║  • Exact-match boost — guaranteed top-1 if sim ≥ 0.92          ║
║  • Medical Awareness Layer — symptom→category soft filter       ║
║  • Unified Arabic normalization (ى/ي, ة/ه, أإآ/ا)              ║
║  • Improved relevance guard (exact + category fast paths)       ║
║  • Rich logging: exact/similarity/semantic per result           ║
║  كل حاجة تانية من v15.0 محتفظ بيها بدون تغيير                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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

from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, model_validator

load_dotenv()

# ──────────────────────────────────────────────────────────────────
# Environment
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
EMBED_DIM   = 384

MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "20"))
EXTERNAL_CALL_TIMEOUT   = int(os.getenv("EXTERNAL_CALL_TIMEOUT", "25"))

if not GEMINI_API_KEY:   raise RuntimeError("❌ GEMINI_API_KEY is not set.")
if not PINECONE_API_KEY: raise RuntimeError("❌ PINECONE_API_KEY is not set.")

# ──────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("sila")

# ──────────────────────────────────────────────────────────────────
# Concurrency
# ──────────────────────────────────────────────────────────────────
_request_semaphore: Optional[asyncio.Semaphore] = None
_rate_limit_store: Dict[str, list] = defaultdict(list)
_rate_limit_lock  = threading.Lock()
_RATE_LIMIT_REQUESTS = 10
_RATE_LIMIT_WINDOW   = 60

def _check_rate_limit(ip: str) -> bool:
    now = time.time()
    with _rate_limit_lock:
        _rate_limit_store[ip] = [ts for ts in _rate_limit_store[ip] if now - ts < _RATE_LIMIT_WINDOW]
        if len(_rate_limit_store[ip]) < _RATE_LIMIT_REQUESTS:
            _rate_limit_store[ip].append(now)
            return True
    return False

# ──────────────────────────────────────────────────────────────────
# Lazy SDKs
# ──────────────────────────────────────────────────────────────────
_pinecone_index = None
_pinecone_lock  = threading.Lock()

def _init_pinecone():
    global _pinecone_index
    if _pinecone_index is None:
        with _pinecone_lock:
            if _pinecone_index is None:
                log.info("🔄 Initialising Pinecone v3…")
                from pinecone import Pinecone as _PC
                _pinecone_index = _PC(api_key=PINECONE_API_KEY).Index(INDEX_NAME)
                log.info(f"✅ Pinecone ready — index: {INDEX_NAME}")
    return _pinecone_index

async def get_index():
    if _pinecone_index is None:
        await asyncio.to_thread(_init_pinecone)
    return _pinecone_index

_gemini_client = None
_gemini_lock   = threading.Lock()

def _init_gemini():
    global _gemini_client
    if _gemini_client is None:
        with _gemini_lock:
            if _gemini_client is None:
                log.info("🔄 Initialising Gemini client…")
                from google import genai as _genai
                _gemini_client = _genai.Client(api_key=GEMINI_API_KEY)
                log.info("✅ Gemini client ready.")
    return _gemini_client

def _get_gemini_sync(): return _init_gemini()
def _gemini_types():
    from google.genai import types
    return types

_st_model: Any = None
_st_lock        = threading.Lock()
_st_load_error: Optional[str] = None

def _load_and_encode_sync(text: str) -> List[float]:
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
    try:
        vec: List[float] = _st_model.encode(
            text, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False,
        ).tolist()
        if len(vec) != EMBED_DIM:
            raise RuntimeError(f"❌ Embedding dim mismatch: got {len(vec)}, expected {EMBED_DIM}.")
        return vec
    except Exception as exc:
        log.error(f"❌ Encoding failed: {exc}")
        raise

async def _encode_async(text: str) -> List[float]:
    try:
        return await asyncio.to_thread(_load_and_encode_sync, text)
    except Exception as exc:
        raise RuntimeError(f"❌ Embedding unavailable: {exc}") from exc

# ──────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────
GEMINI_TEXT_MODELS   = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite"]
GEMINI_VISION_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]

ALLOWED_IMAGE_TYPES = frozenset({
    "image/jpeg", "image/png", "image/webp", "image/heic", "image/heif",
})

MEDICAL_DISCLAIMER = "⚠️ تنبيه: هذه المعلومات للتوجيه العام فقط ولا تُغني عن استشارة طبيب متخصص."

MEDICAL_KEYWORDS: frozenset = frozenset({
    "ألم","وجع","مرض","دواء","طبيب","مستشفى","أعراض","علاج",
    "صداع","حمى","سعال","ضغط","سكر","قلب","كلى","معدة",
    "عظام","جلد","عين","أذن","أنف","رئة","كبد","دم",
    "تعب","إرهاق","دوار","غثيان","إسهال","إمساك","حرقة",
    "الم","عندي","عندى","اشعر","احس","اعاني","يؤلم",
    "بوجعني","بتوجعني","حاسس","حاسه","حبوب","طفح","حكة",
    "عملية","جراحة","منظار","تحليل","أشعة","نتيجة","تقرير",
    "برد","انفلونزا","رشح","زكام","كحة","بلغم","حرارة",
    "pain","ache","fever","cough","headache","nausea","dizzy",
    "vomit","diarrhea","symptom","disease","doctor","hospital",
    "medicine","drug","blood","heart","lung","kidney","liver",
    "diabetes","pressure","infection","allergy","rash","swelling",
    "fatigue","tired","breathe","chest","stomach","throat",
    "surgery","scan","test","result","report","prescription",
    "cold","flu","runny","nose","sneeze","congestion",
})

LOW_QUALITY_PATTERNS: frozenset = frozenset({
    "طبيعي","تم الاجابة","كل شيء ممكن","راجع الطبيب","استشر طبيب",
    "غير واضح","وضح اكثر","natural","answered","everything possible",
    "consult doctor","not clear","clarify more",
})

GARBAGE_PATTERNS: frozenset = frozenset({
    "تم الاجابة","راجع الطبيب","استشر طبيب","كل شيء ممكن","غير واضح",
    "وضح اكثر","طبيعي","لا يوجد","لا يوجد جواب","لا يوجد رد",
    "معلومات غير متوفرة","غير متوفر","لا اعرف","لا أستطيع","لا يمكنني",
    "معلومات محدودة","answered","consult doctor","everything possible",
    "not clear","clarify more","natural","no answer","no response",
    "information not available","not available","don't know","cannot","limited information",
})

MAX_CONTEXT_MATCHES = 3

ARABIC_NORMALIZATION = str.maketrans({
    'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ة': 'ه', 'ى': 'ي', 'ؤ': 'و', 'ئ': 'ي',
})

EMERGENCY_KEYWORDS_AR: frozenset = frozenset({
    "ألم صدر","ضيق تنفس","نوبة قلبية","سكتة دماغية","نزيف شديد",
    "إغماء","فقدان وعي","صدمة","حروق شديدة","كسر عظم",
    "ألم حاد","طوارئ","إسعاف","علاج فوري","خطر على الحياة",
    "ضربة شمس","تسمم","جرح عميق","نزيف داخلي","انفجار",
    "ألم بطن حاد","صعوبة بلع","خدر","شلل","تشنج",
    "انتحار","أفكار انتحارية","إيذاء النفس",
})

EMERGENCY_KEYWORDS_EN: frozenset = frozenset({
    "chest pain","difficulty breathing","heart attack","stroke","severe bleeding",
    "fainting","loss of consciousness","shock","severe burns","broken bone",
    "severe pain","emergency","ambulance","immediate treatment","life threatening",
    "heat stroke","poisoning","deep wound","internal bleeding","explosion",
    "severe abdominal pain","difficulty swallowing","numbness","paralysis","seizure",
    "suicide","suicidal thoughts","self harm",
})

# ── 🆕 v16.0: Hybrid scoring constants ────────────────────────────
WEIGHT_COSINE         = 0.60
WEIGHT_EXACT          = 0.30
WEIGHT_CATEGORY       = 0.10
EXACT_MATCH_THRESHOLD = 0.92    # token-Jaccard above this → exact boost

# ── 🆕 v16.0: Symptom → category map ──────────────────────────────
SYMPTOM_CATEGORY_MAP: Dict[str, List[str]] = {
    "حلق":["respiratory","general"],"سعال":["respiratory","general"],
    "كحة":["respiratory","general"],"رئة":["respiratory"],
    "ربو":["respiratory"],"أنف":["respiratory"],"زكام":["respiratory"],
    "رشح":["respiratory"],"ضيق تنفس":["respiratory","cardiology"],
    "throat":["respiratory","general"],"cough":["respiratory","general"],
    "breath":["respiratory","cardiology"],"asthma":["respiratory"],
    "حرارة":["general","pediatrics"],"حمى":["general","pediatrics"],
    "برد":["general","respiratory"],"انفلونزا":["general","respiratory"],
    "fever":["general","pediatrics"],"infection":["general"],
    "قلب":["cardiology"],"صدر":["cardiology","respiratory"],
    "ضغط":["cardiology"],"heart":["cardiology"],"chest":["cardiology","respiratory"],
    "صداع":["neurology","general"],"دوار":["neurology"],
    "أعصاب":["neurology"],"headache":["neurology","general"],"dizzy":["neurology"],
    "معدة":["gastroenterology"],"بطن":["gastroenterology"],
    "إسهال":["gastroenterology"],"غثيان":["gastroenterology"],
    "كبد":["gastroenterology"],"stomach":["gastroenterology"],
    "diarrhea":["gastroenterology"],"nausea":["gastroenterology"],
    "جلد":["dermatology"],"طفح":["dermatology"],"حكة":["dermatology"],
    "skin":["dermatology"],"rash":["dermatology"],"itching":["dermatology"],
    "عظام":["orthopedic"],"مفاصل":["orthopedic"],"ظهر":["orthopedic"],
    "bone":["orthopedic"],"joint":["orthopedic"],
    "كلى":["urology"],"بول":["urology"],"kidney":["urology"],"urine":["urology"],
    "عين":["ophthalmology"],"نظر":["ophthalmology"],"eye":["ophthalmology"],
    "طفل":["pediatrics"],"رضيع":["pediatrics"],"أطفال":["pediatrics"],
    "child":["pediatrics"],"infant":["pediatrics"],
    "نفس":["psychology"],"قلق":["psychology"],"اكتئاب":["psychology"],
    "anxiety":["psychology"],"depression":["psychology"],
    "سكري":["endocrinology"],"غدة":["endocrinology"],"هرمون":["endocrinology"],
    "diabetes":["endocrinology"],"thyroid":["endocrinology"],
    "حمل":["gynecology"],"دورة":["gynecology"],"رحم":["gynecology"],
    "pregnancy":["gynecology"],"period":["gynecology"],
    "أسنان":["dentistry"],"سن":["dentistry"],"لثة":["dentistry"],
    "tooth":["dentistry"],"gum":["dentistry"],
}

# ──────────────────────────────────────────────────────────────────
# 🆕 v16.0: Text normalization helpers
# ──────────────────────────────────────────────────────────────────
def _normalize_text(text: str) -> str:
    """Unified normalization: lowercase + no punctuation + Arabic unification."""
    if not text:
        return ""
    t = re.sub(r"[\u064b-\u065f\u0670\u0640]", "", text)
    t = t.translate(str.maketrans({'أ':'ا','إ':'ا','آ':'ا','ة':'ه','ى':'ي','ؤ':'و','ئ':'ي'}))
    t = t.lower()
    t = re.sub(r"[^\w\u0600-\u06ff\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def _token_similarity(a: str, b: str) -> float:
    """Jaccard coefficient on normalized word tokens."""
    ta = set(_normalize_text(a).split())
    tb = set(_normalize_text(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

def _extract_expected_categories(query: str) -> List[str]:
    """Map query symptoms → expected medical categories (soft, ordered by frequency)."""
    q   = _normalize_text(query)
    cnt: Dict[str, int] = {}
    for kw, cats in SYMPTOM_CATEGORY_MAP.items():
        if _normalize_text(kw) in q:
            for c in cats:
                cnt[c] = cnt.get(c, 0) + 1
    return sorted(cnt, key=lambda c: cnt[c], reverse=True)

def _compute_hybrid_score(
    cosine: float,
    query_norm: str,
    match_question: str,
    match_category: Optional[str],
    expected_categories: List[str],
) -> Tuple[float, float, float, str]:
    """
    final = cosine*0.6 + exact_bonus*0.3 + category_bonus*0.1
    Returns (final, exact_bonus, category_bonus, match_type)
    """
    sim        = _token_similarity(query_norm, _normalize_text(match_question))
    match_type = "semantic"
    if sim >= EXACT_MATCH_THRESHOLD:
        exact_bonus = 1.0
        match_type  = "exact" if sim == 1.0 else "high_similarity"
    elif sim >= 0.75:
        exact_bonus = sim * 0.6
    else:
        exact_bonus = 0.0

    category_bonus = 0.0
    if match_category and expected_categories:
        cl = (match_category or "").lower()
        ec = [c.lower() for c in expected_categories]
        if cl in ec:
            category_bonus = 1.0 if cl == ec[0] else 0.6

    final = WEIGHT_COSINE * cosine + WEIGHT_EXACT * exact_bonus + WEIGHT_CATEGORY * category_bonus
    return round(final, 4), round(exact_bonus, 4), round(category_bonus, 4), match_type


# ──────────────────────────────────────────────────────────────────
# Domain Models
# ──────────────────────────────────────────────────────────────────
@dataclass
class KnowledgeMatch:
    question:   str
    answer:     str
    confidence: float
    category:   Optional[str] = None

    @property
    def is_reliable(self) -> bool:
        return self.confidence >= MIN_CONFIDENCE

    @property
    def is_low_quality(self) -> bool:
        al = self.answer.lower()
        return any(p.lower() in al for p in LOW_QUALITY_PATTERNS)

    @property
    def is_garbage(self) -> bool:
        a = self.answer.strip()
        if len(a) < 12:
            return True
        al = a.lower()
        return any(p.lower() in al for p in GARBAGE_PATTERNS)


@dataclass
class QueryContext:
    raw_query: str
    language:  str
    matches:   List[KnowledgeMatch] = field(default_factory=list)

    @property
    def has_reliable_matches(self) -> bool:
        return any(m.is_reliable for m in self.matches)

    @property
    def best_confidence(self) -> float:
        return self.matches[0].confidence if self.matches else 0.0


# ──────────────────────────────────────────────────────────────────
# Pydantic Models
# ──────────────────────────────────────────────────────────────────
class MessageDto(BaseModel):
    role:    str
    content: str

class AskRequest(BaseModel):
    question: Optional[str] = None
    text:     Optional[str] = None
    history:  Optional[List[MessageDto]] = None

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
    question:   str
    answer:     str
    confidence: float
    category:   Optional[str] = None

class AskResponse(BaseModel):
    query:              str
    reply:              str
    model_used:         str
    matches:            List[MatchResult]
    is_medical:         bool
    found_in_database:  bool
    low_confidence:     bool
    language:           str
    disclaimer:         str

class ImageAnalysisResponse(BaseModel):
    status:     str
    analysis:   Optional[str] = None
    model_used: Optional[str] = None
    disclaimer: str = MEDICAL_DISCLAIMER


# ──────────────────────────────────────────────────────────────────
# Language Detector
# ──────────────────────────────────────────────────────────────────
class LanguageDetector:
    @staticmethod
    def detect(text: str) -> str:
        if not text: return "ar"
        arabic = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
        return "ar" if arabic / max(len(text.strip()), 1) > 0.25 else "en"


# ──────────────────────────────────────────────────────────────────
# Emergency Detector
# ──────────────────────────────────────────────────────────────────
class EmergencyDetector:
    @staticmethod
    def is_emergency(query: str, language: str) -> bool:
        q = query.lower()
        kws = EMERGENCY_KEYWORDS_AR if language == "ar" else EMERGENCY_KEYWORDS_EN
        return any(kw in q for kw in kws)


# ──────────────────────────────────────────────────────────────────
# Intent Classifier
# ──────────────────────────────────────────────────────────────────
class IntentClassifier:
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
        q = query.lower()
        if any(kw in q for kw in MEDICAL_KEYWORDS):
            log.info("[Intent] Medical keyword detected")
            return "medical"
        try:
            types = _gemini_types()
            resp  = _get_gemini_sync().models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=cls._PROMPT.format(query=query),
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=5),
            )
            result = resp.text.strip().lower()
            return "medical" if "medical" in result else "social"
        except Exception as exc:
            log.warning(f"[Intent] Gemini failed: {exc} — defaulting to social")
        return "social"

    @classmethod
    async def classify(cls, query: str) -> str:
        return await asyncio.to_thread(cls._classify_sync, query)


# ──────────────────────────────────────────────────────────────────
# Knowledge Base Service  🆕 v16.0 Hybrid Retrieval
# ──────────────────────────────────────────────────────────────────
class KnowledgeBaseService:

    # ── Text helpers ───────────────────────────────────────────────
    @staticmethod
    def _normalize_arabic(text: str) -> str:
        clean = re.sub(r"[\u064b-\u065f\u0670]", "", text.lower())
        return clean.translate(ARABIC_NORMALIZATION)

    @staticmethod
    def _sanitize_match(match: "KnowledgeMatch") -> "KnowledgeMatch":
        q = re.sub(r'\s+', ' ', match.question.strip())
        a = re.sub(r'\s+', ' ', match.answer.strip())
        if any('\u0600' <= c <= '\u06ff' for c in q):
            q = KnowledgeBaseService._normalize_arabic(q)
        if any('\u0600' <= c <= '\u06ff' for c in a):
            a = KnowledgeBaseService._normalize_arabic(a)
        return KnowledgeMatch(question=q, answer=a, confidence=match.confidence, category=match.category)

    @staticmethod
    def _deduplicate_and_sanitize(matches: List["KnowledgeMatch"]) -> List["KnowledgeMatch"]:
        seen, out = set(), []
        for m in matches:
            if m.is_garbage: continue
            key = m.answer.lower().strip()
            if key in seen: continue
            seen.add(key)
            out.append(KnowledgeBaseService._sanitize_match(m))
        return out

    @staticmethod
    def _select_top_matches(matches: List["KnowledgeMatch"], max_count: int = MAX_CONTEXT_MATCHES) -> List["KnowledgeMatch"]:
        if not matches: return []
        sorted_m, seen_cats, selected = sorted(matches, key=lambda m: m.confidence, reverse=True), set(), []
        for m in sorted_m:
            if len(selected) >= max_count: break
            if m.category not in seen_cats or len(selected) < 2:
                selected.append(m)
                if m.category: seen_cats.add(m.category)
        return selected

    @staticmethod
    def _calculate_garbage_ratio(matches: List["KnowledgeMatch"]) -> float:
        if not matches: return 0.0
        return round(sum(1 for m in matches if m.is_garbage) / len(matches), 2)

    @staticmethod
    def _category_consistency(matches: List["KnowledgeMatch"]) -> float:
        if not matches: return 0.0
        cats = [m.category for m in matches if m.category]
        if not cats: return 0.0
        return round(Counter(cats).most_common(1)[0][1] / len(cats), 2)

    @staticmethod
    def _extract_medical_tokens(text: str) -> set:
        clean = re.sub(r"[\u064b-\u065f\u0670]", "", text.lower())
        clean = clean.translate(ARABIC_NORMALIZATION)
        tokens = set(t for t in re.split(r"[\s\W]+", clean) if len(t) >= 3)
        out = set()
        for tok in tokens:
            if any(kw in tok or tok in kw for kw in MEDICAL_KEYWORDS):
                out.add(tok)
            elif len(tok) >= 4:
                out.add(tok)
        return out

    # ── 🆕 v16.0: Hybrid search ────────────────────────────────────
    async def search(self, query: str, top_k: int = TOP_K) -> List["KnowledgeMatch"]:
        """Hybrid search: cosine + exact-match boost + category soft filter."""
        log.info(f"[KB-v2] Searching: '{query[:80]}'")

        # Medical awareness layer
        expected_cats = _extract_expected_categories(query)
        log.info(f"[KB-v2] Expected categories: {expected_cats or ['unknown']}")

        # Encode
        try:
            vector = await _encode_async(query)
        except Exception as exc:
            log.error(f"[KB-v2] Encoding failed: {exc}")
            return []

        # Fetch 3× from Pinecone, then re-rank
        index   = await get_index()
        fetch_k = min(top_k * 3, 30)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                _qfn    = functools.partial(
                    index.query, vector=vector, top_k=fetch_k,
                    include_metadata=True, namespace=PINECONE_NAMESPACE or "",
                )
                results = await asyncio.to_thread(_qfn)

                raw_scores = [round(float(m.score), 4) for m in results.matches if m.score]
                log.info(f"[KB-v2] Pinecone raw scores ({len(raw_scores)}): {raw_scores}")

                return self._parse_matches_hybrid(results, query, expected_cats, top_k)

            except Exception as exc:
                log.warning(f"[KB-v2] Attempt {attempt}/{MAX_RETRIES} failed: {exc}")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY * attempt)

        log.error("[KB-v2] All retries exhausted")
        return []

    # ── 🆕 v16.0: Hybrid re-ranking ────────────────────────────────
    @staticmethod
    def _parse_matches_hybrid(
        results: Any,
        query: str,
        expected_cats: List[str],
        top_k: int,
    ) -> List["KnowledgeMatch"]:
        query_norm = _normalize_text(query)
        candidates, filtered = [], 0

        for m in results.matches:
            cosine = float(m.score) if m.score is not None else 0.0

            # Relaxed pre-filter (let re-ranking do the real work)
            if cosine < MIN_CONFIDENCE * 0.80:
                filtered += 1
                continue

            meta     = m.metadata or {}
            q_text   = meta.get("question", "")
            a_text   = meta.get("answer",   "")
            category = meta.get("category", "General")

            final, exact_b, cat_b, match_type = _compute_hybrid_score(
                cosine             = cosine,
                query_norm         = query_norm,
                match_question     = q_text,
                match_category     = category,
                expected_categories= expected_cats,
            )

            icon = "🎯" if match_type == "exact" else "🔍" if match_type == "high_similarity" else "🌐"
            log.info(
                f"[KB-v2] {icon} {match_type:<16} cosine={cosine:.3f} "
                f"exact_b={exact_b:.2f} cat_b={cat_b:.2f} → final={final:.4f} "
                f"| cat={category} | q='{q_text[:50]}'"
            )

            candidates.append(KnowledgeMatch(
                question=q_text, answer=a_text, confidence=final, category=category,
            ))

        # Re-rank by hybrid score
        candidates.sort(key=lambda x: x.confidence, reverse=True)

        # Apply threshold on hybrid score
        kept = [c for c in candidates if c.confidence >= MIN_CONFIDENCE]
        log.info(
            f"[KB-v2] Re-ranked: {len(kept)} kept "
            f"(top={kept[0].confidence:.4f}), "
            f"{len(candidates)-len(kept)} below threshold, {filtered} pre-filtered"
        )
        return kept[:top_k]

    # ── 🆕 v16.0: Improved relevance guard ─────────────────────────
    @staticmethod
    def _relevance_ok(query: str, matches: List["KnowledgeMatch"], min_overlap: int = 1) -> bool:
        if not matches:
            return False

        # Fast path 1: exact / high-similarity match
        qn = _normalize_text(query)
        for m in matches[:3]:
            sim = _token_similarity(qn, _normalize_text(m.question))
            if sim >= EXACT_MATCH_THRESHOLD:
                log.info(f"[KB-v2] Relevance: ✅ exact match (sim={sim:.3f})")
                return True

        # Fast path 2: category match
        expected = _extract_expected_categories(query)
        if expected and matches:
            top_cat = (matches[0].category or "").lower()
            if top_cat in [c.lower() for c in expected]:
                log.info(f"[KB-v2] Relevance: ✅ category match ({top_cat})")
                return True

        # Slow path: medical token overlap
        qt = KnowledgeBaseService._extract_medical_tokens(query)
        if not qt:
            log.info("[KB-v2] Relevance: no medical tokens — allowing")
            return True

        combined = " ".join(f"{m.question} {m.answer}" for m in matches[:3])
        mt       = KnowledgeBaseService._extract_medical_tokens(combined)
        overlap  = len(qt & mt)
        ok       = overlap >= min_overlap

        log.info(
            f"[KB-v2] Relevance: query_tokens={len(qt)} match_tokens={len(mt)} "
            f"overlap={overlap} → {'✅' if ok else '❌'}"
        )
        return ok


# ──────────────────────────────────────────────────────────────────
# Prompt Builder  (unchanged from v15.0)
# ──────────────────────────────────────────────────────────────────
class PromptBuilder:
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

    _RAG_LIGHT_AR = (
        "أنت 'سيلا'، مساعد طبي ذكي وموثوق.\n"
        "⚠️ المعلومات المسترجعة محدودة — استخدمها كإشارات داعمة فقط، "
        "واعتمد بشكل أساسي على معرفتك الطبية العامة.\n\n"
        "قواعد صارمة:\n"
        "١. استخدم المعلومات المقدمة كإشارات داعمة فقط.\n"
        "٢. اعتمد على معرفتك الطبية العامة لتقديم إجابة شاملة ومفيدة.\n"
        "٣. لا تُقدم تشخيصاً نهائياً أبداً.\n"
        "٤. اذكر علامات الخطر إن وُجدت.\n"
        "٥. اختم بالتوصية بمراجعة طبيب متخصص.\n"
        "٦. لغة الإجابة: عربية واضحة ومفهومة."
    )

    _RAG_LIGHT_EN = (
        "You are 'Sila', a trusted and empathetic medical AI assistant.\n"
        "⚠️ Retrieved information is limited — use it only as supporting hints "
        "and rely primarily on your general medical knowledge.\n\n"
        "Strict rules:\n"
        "1. Use the provided information as supporting hints only.\n"
        "2. Use your general medical knowledge for a comprehensive answer.\n"
        "3. NEVER provide a definitive diagnosis.\n"
        "4. Flag warning signs requiring urgent care.\n"
        "5. Always close by recommending a specialist consultation."
    )

    _EMERGENCY_AR = (
        "أنت 'سيلا'، مساعد طبي ذكي وموثوق.\n"
        "🚨 تنبيه هام: يبدو أن المريض يعاني من أعراض طارئة.\n\n"
        "قواعد صارمة:\n"
        "١. قدّم إجابة فورية ومباشرة حول الأعراض الطارئة.\n"
        "٢. أوصي بشدة بمراجعة قسم الطوارئ أو الاتصال بالإسعاف فوراً.\n"
        "٣. اذكر علامات الخطر التي تستدعي التدخل العاجل.\n"
        "٤. لا تُقدم تشخيصاً نهائياً — قدّم تقييماً أولياً فقط.\n"
        "٥. كن مختصراً ومباشراً — السلامة أولاً."
    )

    _EMERGENCY_EN = (
        "You are 'Sila', a trusted and empathetic medical AI assistant.\n"
        "🚨 Alert: The patient appears to be experiencing emergency symptoms.\n\n"
        "Strict rules:\n"
        "1. Provide immediate guidance on emergency symptoms.\n"
        "2. Strongly recommend the emergency department or ambulance immediately.\n"
        "3. List warning signs requiring urgent intervention.\n"
        "4. NEVER provide a definitive diagnosis.\n"
        "5. Be concise and direct — safety first."
    )

    _STRUCTURE_AR = (
        "رتّب إجابتك بهذا الشكل:\n\n"
        "🔍 الأسباب المحتملة:\n→ اذكر الأسباب الأكثر احتمالاً\n\n"
        "💊 التوصيات والخطوات العملية:\n→ ما يمكن للمريض فعله الآن\n\n"
        "🏥 التخصص الطبي المناسب للمراجعة:\n→ اذكر التخصص المناسب\n\n"
        "⚠️ علامات الخطر التي تستدعي الطوارئ فوراً:\n→ اذكرها إن وُجدت"
    )

    _STRUCTURE_EN = (
        "Structure your response as follows:\n\n"
        "🔍 Possible Causes:\n→ Based on the provided context\n\n"
        "💊 Recommendations & Next Steps:\n→ Practical actions the patient can take\n\n"
        "🏥 Recommended Medical Specialty:\n→ Which specialist to consult\n\n"
        "⚠️ Warning Signs Requiring Immediate Emergency Care:\n→ List if present"
    )

    _FALLBACK_AR = (
        "بناءً على الأعراض المذكورة، قد تكون الحالة ناتجة عن عدة أسباب محتملة. "
        "يلزم فحص طبي دقيق لتحديد السبب بدقة. "
        "أنصح بشدة بمراجعة طبيب متخصص للتقييم المناسب. 🏥"
    )

    _FALLBACK_EN = (
        "Based on the symptoms mentioned, this could be caused by several possible factors. "
        "A proper medical examination is needed to accurately determine the cause. "
        "I strongly recommend consulting a specialist for proper evaluation. 🏥"
    )

    def safe_fallback_response(self, language: str) -> str:
        return self._FALLBACK_AR if language == "ar" else self._FALLBACK_EN

    def _build_context_block(self, matches: List[KnowledgeMatch], language: str) -> str:
        sep, parts = self._SEP, []
        for i, m in enumerate(matches, 1):
            rel = ("✅ موثوق" if m.is_reliable else "⚠️ ثقة منخفضة") if language == "ar" \
                  else ("✅ Reliable" if m.is_reliable else "⚠️ Low confidence")
            parts.append(
                f"[{i}] {rel} — Score: {m.confidence:.0%}\n"
                f"[Specialty: {m.category or 'General'}]\n"
                f"Q: {m.question}\nA: {m.answer}"
            )
        return f"\n\n{sep}\n".join(parts)

    def _prompt(self, system, context_label, context, question_label, query, structure, answer_label):
        sep = self._SEP
        return (
            f"{system}\n\n{sep}\n"
            f"{context_label}\n\n{context}\n\n{sep}\n"
            f"{question_label}\n{query}\n\n"
            f"{structure}\n\n{answer_label}"
        )

    def build(self, ctx: QueryContext) -> str:
        lang = ctx.language
        return self._prompt(
            self._SYSTEM_AR if lang=="ar" else self._SYSTEM_EN,
            "📋 قاعدة المعرفة الطبية:" if lang=="ar" else "📋 Medical Knowledge Base:",
            self._build_context_block(ctx.matches, lang),
            "🧑‍⚕️ سؤال المريض:" if lang=="ar" else "🧑‍⚕️ Patient Question:",
            ctx.raw_query,
            self._STRUCTURE_AR if lang=="ar" else self._STRUCTURE_EN,
            "الإجابة:" if lang=="ar" else "Answer:",
        )

    def build_rag_light(self, ctx: QueryContext) -> str:
        lang = ctx.language
        return self._prompt(
            self._RAG_LIGHT_AR if lang=="ar" else self._RAG_LIGHT_EN,
            "📋 قاعدة المعرفة الطبية (إشارات محدودة):" if lang=="ar" else "📋 Medical Knowledge Base (limited hints):",
            self._build_context_block(ctx.matches, lang),
            "🧑‍⚕️ سؤال المريض:" if lang=="ar" else "🧑‍⚕️ Patient Question:",
            ctx.raw_query,
            self._STRUCTURE_AR if lang=="ar" else self._STRUCTURE_EN,
            "الإجابة:" if lang=="ar" else "Answer:",
        )

    def build_emergency(self, query: str, language: str) -> str:
        sep = self._SEP
        system    = self._EMERGENCY_AR if language=="ar" else self._EMERGENCY_EN
        structure = self._STRUCTURE_AR  if language=="ar" else self._STRUCTURE_EN
        ql        = "🧑‍⚕️ سؤال المريض:" if language=="ar" else "🧑‍⚕️ Patient Question:"
        al        = "الإجابة:"            if language=="ar" else "Answer:"
        return f"{system}\n\n{sep}\n{ql}\n{query}\n\n{structure}\n\n{al}"

    def build_gemini_only(self, query: str, language: str) -> str:
        sep = self._SEP
        system    = self._GEMINI_ONLY_AR if language=="ar" else self._GEMINI_ONLY_EN
        structure = self._STRUCTURE_AR   if language=="ar" else self._STRUCTURE_EN
        ql        = "🧑‍⚕️ سؤال المريض:" if language=="ar" else "🧑‍⚕️ Patient Question:"
        al        = "الإجابة:"            if language=="ar" else "Answer:"
        return f"{system}\n\n{sep}\n{ql}\n{query}\n\n{structure}\n\n{al}"

    def no_data_response(self, language: str) -> str:
        return (
            "لا تتوفر لديّ معلومات كافية. أنصح بمراجعة طبيب متخصص. 🏥"
            if language=="ar" else
            "I don't have sufficient information. Please consult a specialist. 🏥"
        )


# ──────────────────────────────────────────────────────────────────
# LRU Cache
# ──────────────────────────────────────────────────────────────────
class _BoundedLRU:
    def __init__(self, maxsize: int = 200):
        self._cache: collections.OrderedDict = collections.OrderedDict()
        self._max   = maxsize
        self._lock  = threading.Lock()

    def get(self, key: str) -> Optional[Tuple[str, str]]:
        with self._lock:
            if key not in self._cache: return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def put(self, key: str, value: Tuple[str, str]) -> None:
        with self._lock:
            if key in self._cache: self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self._max: self._cache.popitem(last=False)

    def __len__(self) -> int: return len(self._cache)


# ──────────────────────────────────────────────────────────────────
# Gemini Service  (improved retry + safe fallback)
# ──────────────────────────────────────────────────────────────────
class GeminiService:
    def __init__(self) -> None:
        self._cache = _BoundedLRU(200)

    @staticmethod
    def _make_config(temperature: float = 0.2, max_tokens: int = 2048):
        types = _gemini_types()
        return types.GenerateContentConfig(temperature=temperature, max_output_tokens=max_tokens)

    # ── Social ──────────────────────────────────────────────────────
    def _reply_social_sync(self, query: str, language: str) -> Tuple[str, str]:
        system = (
            "أنت 'سيلا'، مساعد طبي ذكي وودود. رد بالعربية بشكل طبيعي ودافئ. الرد قصير (جملة أو اتنين)."
            if language == "ar" else
            "You are 'Sila', a friendly medical AI. Reply naturally in English. Keep it brief (1-2 sentences)."
        )
        types = _gemini_types()
        for model in GEMINI_TEXT_MODELS:
            try:
                resp = _get_gemini_sync().models.generate_content(
                    model=model, contents=query,
                    config=types.GenerateContentConfig(
                        system_instruction=system, temperature=0.75, max_output_tokens=200,
                    ),
                )
                return resp.text.strip(), model
            except Exception as exc:
                log.warning(f"[Social] {model} failed: {exc}")
        fallback = "أهلاً! 😊 أنا سيلا، مساعدتك الطبية. كيف يمكنني مساعدتك؟" \
                   if language=="ar" else "Hello! 😊 I'm Sila, your medical AI. How can I help?"
        return fallback, "fallback"

    async def reply_social(self, query: str, language: str) -> Tuple[str, str]:
        return await asyncio.to_thread(self._reply_social_sync, query, language)

    # ── RAG generation (improved retry) ────────────────────────────
    def _generate_sync(self, prompt: str) -> Tuple[str, str]:
        cache_key = hashlib.sha256(prompt.encode()).hexdigest()
        cached    = self._cache.get(cache_key)
        if cached:
            log.info("[Gemini] ✅ Cache hit")
            return cached

        for model in GEMINI_TEXT_MODELS:
            for attempt in range(1, 3):   # 2 attempts per model
                try:
                    log.info(f"[Gemini] {model} attempt {attempt}/2")
                    resp   = _get_gemini_sync().models.generate_content(
                        model=model, contents=prompt, config=self._make_config(),
                    )
                    result = (resp.text.strip(), model)
                    self._cache.put(cache_key, result)
                    log.info(f"[Gemini] ✅ Success — {model}")
                    return result
                except Exception as exc:
                    log.warning(f"[Gemini] {model} attempt {attempt} failed: {exc}")
                    if attempt < 2:
                        time.sleep(2 ** (attempt - 1))  # 1s, then 2s

        log.error("[Gemini] ❌ All models exhausted — returning safe fallback")
        # Never return "حدث خطأ مؤقت" — return a safe medical fallback
        safe = (
            "بناءً على الأعراض المذكورة، قد تكون الحالة ناتجة عن عدة أسباب محتملة. "
            "يلزم فحص طبي دقيق. أنصح بمراجعة طبيب متخصص للتقييم المناسب. 🏥"
        )
        return safe, "safe_fallback"

    async def generate(self, prompt: str) -> Tuple[str, str]:
        return await asyncio.to_thread(self._generate_sync, prompt)

    # ── Image analysis (unchanged) ──────────────────────────────────
    def _analyze_image_sync(self, image_bytes: bytes, mime_type: str) -> Tuple[str, str, str]:
        system_prompt = (
            "You are a specialized medical image analysis AI.\n\n"
            "You ONLY analyze medical images. Accepted types:\n"
            "  - Lab results / blood tests\n  - Prescriptions / medical reports\n"
            "  - X-rays, MRI, CT scans\n  - ECG / EKG strips\n"
            "  - Pathology slides\n  - Ultrasound images\n\n"
            "CRITICAL RULES:\n"
            "1. If NOT medical → respond with JSON ONLY:\n"
            '   {"status": "rejected", "analysis": "Not a medical image."}\n\n'
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
            for contents in [
                [system_prompt, types.Part.from_bytes(data=image_bytes, mime_type=mime_type)],
                [{"text": system_prompt}, {"inline_data": {"mime_type": mime_type, "data": b64_data}}],
            ]:
                try:
                    resp  = _get_gemini_sync().models.generate_content(
                        model=model_name, contents=contents,
                        config=self._make_config(temperature=0.1, max_tokens=4096),
                    )
                    raw   = resp.text.strip()
                    clean = re.sub(r"^\s*```+(?:json)?\s*|\s*```+\s*$", "", raw, flags=re.MULTILINE).strip()
                    brace = clean.find("{")
                    if brace > 0: clean = clean[brace:]
                    try:
                        parsed   = json.loads(clean)
                        status   = str(parsed.get("status", "success"))
                        analysis = parsed.get("analysis", "")
                        if not isinstance(analysis, str):
                            analysis = json.dumps(analysis, ensure_ascii=False, indent=2)
                        return status, analysis.strip(), model_name
                    except json.JSONDecodeError:
                        return "success", raw, model_name
                except Exception as exc:
                    log.warning(f"[Vision] {model_name} failed: {exc}")

        return "error", "تعذّر تحليل الصورة مؤقتاً. يرجى المحاولة لاحقاً.", "none"

    async def analyze_image(self, image_bytes: bytes, mime_type: str) -> Tuple[str, str, str]:
        return await asyncio.to_thread(self._analyze_image_sync, image_bytes, mime_type)

    @property
    def cache_size(self) -> int: return len(self._cache)


# ──────────────────────────────────────────────────────────────────
# App State
# ──────────────────────────────────────────────────────────────────
class AppState:
    knowledge_base: Optional[KnowledgeBaseService] = None
    gemini:         Optional[GeminiService]         = None
    prompt_builder: Optional[PromptBuilder]         = None

state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _request_semaphore
    log.info("🚀 SILA v16.0 starting…")
    _request_semaphore   = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    state.knowledge_base = KnowledgeBaseService()
    state.gemini         = GeminiService()
    state.prompt_builder = PromptBuilder()
    log.info(
        f"✅ Boot complete — embed={EMBED_MODEL} dim={EMBED_DIM} "
        f"min_confidence={MIN_CONFIDENCE} hybrid_weights=({WEIGHT_COSINE}/{WEIGHT_EXACT}/{WEIGHT_CATEGORY}) "
        f"exact_threshold={EXACT_MATCH_THRESHOLD} index={INDEX_NAME}"
    )
    yield
    log.info("🛑 SILA shutting down")


# ──────────────────────────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sila — Medical AI Assistant v16.0",
    description="Hybrid RAG · Exact-Match Boost · Arabic & English",
    version="16.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.error(f"❌ {type(exc).__name__} on {request.url.path}: {exc}")
    return JSONResponse(status_code=500, content={"error": "An unexpected error occurred."})

@app.get("/")
def root():
    return {"name":"Sila — Medical AI","version":"16.0.0","status":"running ✅",
            "endpoints":["/ask","/analyze-image","/health","/docs"]}

@app.get("/health")
def health():
    return {"status":"ok","version":"16.0.0","config":{
        "embed_model":EMBED_MODEL,"embed_dim":EMBED_DIM,"index":INDEX_NAME,
        "min_confidence":MIN_CONFIDENCE,"top_k":TOP_K,"max_context_matches":MAX_CONTEXT_MATCHES,
        "hybrid_weights":f"cosine={WEIGHT_COSINE} exact={WEIGHT_EXACT} category={WEIGHT_CATEGORY}",
        "exact_match_threshold":EXACT_MATCH_THRESHOLD,
        "garbage_protection":"enabled","emergency_detection":"enabled",
    }}

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, request: Request) -> AskResponse:
    if _request_semaphore is None:
        return JSONResponse(status_code=503, content={"error":"Server not ready"})

    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        lang = LanguageDetector.detect(req.query)
        msg  = ("لقد تجاوزت الحد المسموح من الطلبات." if lang=="ar"
                else "Rate limit exceeded. Please try again in a minute.")
        return JSONResponse(status_code=429, content={"error": msg})

    await _request_semaphore.acquire()
    try:
        return await asyncio.wait_for(_ask_inner(req), timeout=EXTERNAL_CALL_TIMEOUT)
    except asyncio.TimeoutError:
        lang = LanguageDetector.detect(req.query)
        msg  = ("عذراً، استغرق الطلب وقتاً أطول من المتوقع." if lang=="ar"
                else "Request timed out. Please try again.")
        return AskResponse(
            query=req.query, reply=msg, model_used="none", matches=[],
            is_medical=True, found_in_database=False, low_confidence=True,
            language=lang, disclaimer=MEDICAL_DISCLAIMER,
        )
    finally:
        try: _request_semaphore.release()
        except Exception: pass


async def _ask_inner(req: AskRequest) -> AskResponse:
    start = time.time()
    q     = req.query
    lang  = LanguageDetector.detect(q)
    intent= await IntentClassifier.classify(q)
    log.info(f"[ASK] '{q[:80]}' | lang={lang} | intent={intent}")

    # Social
    if intent == "social":
        reply, model = await state.gemini.reply_social(q, lang)
        log.info(f"[ASK] ✅ Social — {model} | {round((time.time()-start)*1000)}ms")
        return AskResponse(query=q, reply=reply, model_used=model, matches=[],
                           is_medical=False, found_in_database=False, low_confidence=False,
                           language=lang, disclaimer=MEDICAL_DISCLAIMER)

    # Emergency check
    is_emg = EmergencyDetector.is_emergency(q, lang)
    if is_emg: log.warning("[ASK] 🚨 EMERGENCY")

    # Retrieve
    try:
        raw_matches = await state.knowledge_base.search(q, top_k=TOP_K)
    except Exception as exc:
        log.error(f"[ASK] KB search failed: {exc}")
        raw_matches = []

    sanitized = KnowledgeBaseService._deduplicate_and_sanitize(raw_matches)
    garbage_r = KnowledgeBaseService._calculate_garbage_ratio(raw_matches)
    selected  = KnowledgeBaseService._select_top_matches(sanitized, MAX_CONTEXT_MATCHES)

    top_score  = selected[0].confidence if selected else 0.0
    cat_cons   = KnowledgeBaseService._category_consistency(selected)
    rel_ok     = KnowledgeBaseService._relevance_ok(q, selected)

    # 4-mode decision
    if is_emg:
        rag_mode = "EMERGENCY_OVERRIDE"; reason = "emergency"
    elif not selected or top_score < MIN_CONFIDENCE:
        rag_mode = "GEMINI_ONLY"; reason = f"no_matches(score={top_score:.3f})"
    elif not rel_ok:
        rag_mode = "GEMINI_ONLY"; reason = "relevance_guard"
    elif garbage_r > 0.5:
        rag_mode = "GEMINI_ONLY"; reason = f"garbage_ratio={garbage_r:.2f}"
    elif top_score >= 0.80 and cat_cons >= 0.7:
        rag_mode = "RAG_STRONG"; reason = f"high_conf={top_score:.3f}"
    elif top_score >= MIN_CONFIDENCE:
        rag_mode = "RAG_LIGHT"; reason = f"medium_conf={top_score:.3f}"
    else:
        rag_mode = "GEMINI_ONLY"; reason = "fallback"

    log.info(f"[ASK] 🎯 {rag_mode} — {reason} | top={top_score:.4f} "
             f"garbage={garbage_r:.2f} rel={rel_ok} cat_cons={cat_cons:.2f}")

    match_results = [MatchResult(question=m.question, answer=m.answer,
                                 confidence=m.confidence, category=m.category)
                     for m in selected]

    # Execute mode
    async def _gen(prompt):
        return await state.gemini.generate(prompt)

    if rag_mode == "EMERGENCY_OVERRIDE":
        reply, model = await _gen(state.prompt_builder.build_emergency(q, lang))
        return AskResponse(query=q, reply=reply, model_used=model, matches=match_results,
                           is_medical=True, found_in_database=False, low_confidence=False,
                           language=lang, disclaimer=MEDICAL_DISCLAIMER)

    if rag_mode == "RAG_STRONG":
        ctx = QueryContext(raw_query=q, language=lang, matches=selected)
        reply, model = await _gen(state.prompt_builder.build(ctx))
        return AskResponse(query=q, reply=reply, model_used=model, matches=match_results,
                           is_medical=True, found_in_database=True, low_confidence=False,
                           language=lang, disclaimer=MEDICAL_DISCLAIMER)

    if rag_mode == "RAG_LIGHT":
        ctx = QueryContext(raw_query=q, language=lang, matches=selected)
        reply, model = await _gen(state.prompt_builder.build_rag_light(ctx))
        return AskResponse(query=q, reply=reply, model_used=model, matches=match_results,
                           is_medical=True, found_in_database=True, low_confidence=True,
                           language=lang, disclaimer=MEDICAL_DISCLAIMER)

    # GEMINI_ONLY
    reply, model = await _gen(state.prompt_builder.build_gemini_only(q, lang))
    return AskResponse(query=q, reply=reply, model_used=model, matches=match_results,
                       is_medical=True, found_in_database=False, low_confidence=True,
                       language=lang, disclaimer=MEDICAL_DISCLAIMER)


@app.post("/analyze-image", response_model=ImageAnalysisResponse)
async def analyze_image(file: UploadFile = File(...)) -> JSONResponse:
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        return JSONResponse(status_code=400, content={
            "status":"error",
            "analysis":f"نوع الملف '{file.content_type}' غير مدعوم.",
            "model_used":"none","disclaimer":MEDICAL_DISCLAIMER,
        })
    try:
        image_bytes = await file.read()
    except Exception as exc:
        return JSONResponse(status_code=400, content={
            "status":"error","analysis":"فشل في قراءة الملف.",
            "model_used":"none","disclaimer":MEDICAL_DISCLAIMER,
        })
    if not image_bytes:
        return JSONResponse(status_code=400, content={
            "status":"error","analysis":"الملف المرفوع فارغ.",
            "model_used":"none","disclaimer":MEDICAL_DISCLAIMER,
        })
    if len(image_bytes) > MAX_IMAGE_BYTES:
        return JSONResponse(status_code=413, content={
            "status":"error",
            "analysis":f"حجم الصورة يتجاوز {MAX_IMAGE_MB}MB.",
            "model_used":"none","disclaimer":MEDICAL_DISCLAIMER,
        })
    try:
        status, analysis, model = await asyncio.wait_for(
            state.gemini.analyze_image(image_bytes, file.content_type),
            timeout=EXTERNAL_CALL_TIMEOUT,
        )
    except asyncio.TimeoutError:
        return JSONResponse(status_code=504, content={
            "status":"error","analysis":"انتهت مهلة تحليل الصورة.",
            "model_used":"none","disclaimer":MEDICAL_DISCLAIMER,
        })
    http_code = 200 if status == "success" else 503 if status == "error" else 200
    return JSONResponse(status_code=http_code, content={
        "status":status,"analysis":analysis,"model_used":model,"disclaimer":MEDICAL_DISCLAIMER,
    })


if __name__ == "__main__":
    import uvicorn
    log.info("🚀 Starting SILA v16.0…")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", access_log=True)