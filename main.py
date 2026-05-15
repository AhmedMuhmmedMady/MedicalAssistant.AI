"""
╔══════════════════════════════════════════════════════════════════╗
║  SILA — Medical AI Assistant  v17.1                             ║
║  Model : paraphrase-multilingual-MiniLM-L12-v2  (Arabic+EN)    ║
║  Fixes : Intent bug · History · Image lang · Retry logic       ║
╚══════════════════════════════════════════════════════════════════╝
"""
import os
os.environ["PYTHONUNBUFFERED"]    = "1"
os.environ.setdefault("OMP_NUM_THREADS",        "1")
os.environ.setdefault("MKL_NUM_THREADS",        "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import asyncio, base64, collections, functools, hashlib
import json, logging, re, sys, threading, time
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

sys.setrecursionlimit(1000)

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, model_validator

load_dotenv()

# ── Env ───────────────────────────────────────────────────────────
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY", "")
PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY", "")
INDEX_NAME         = os.getenv("PINECONE_INDEX", "medical-index-arabicdata")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "")

MIN_CONFIDENCE          = float(os.getenv("SCORE_THRESHOLD",        "0.45"))
MAX_QUERY_LENGTH        = int(os.getenv("MAX_QUERY_LENGTH",         "500"))
MAX_IMAGE_MB            = int(os.getenv("MAX_IMAGE_SIZE_MB",        "10"))
MAX_IMAGE_BYTES         = MAX_IMAGE_MB * 1024 * 1024
TOP_K                   = int(os.getenv("TOP_K",                    "7"))
MAX_RETRIES             = int(os.getenv("MAX_RETRIES",              "3"))
RETRY_DELAY             = float(os.getenv("RETRY_DELAY",            "1.5"))
EMBED_MODEL             = os.getenv("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
EMBED_DIM               = 384
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS",  "10"))
EXTERNAL_CALL_TIMEOUT   = int(os.getenv("EXTERNAL_CALL_TIMEOUT",    "45"))

if not GEMINI_API_KEY:   raise RuntimeError("❌ GEMINI_API_KEY missing")
if not PINECONE_API_KEY: raise RuntimeError("❌ PINECONE_API_KEY missing")

# ── Logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("sila")

# ── Concurrency ───────────────────────────────────────────────────
_request_semaphore: Optional[asyncio.Semaphore] = None
_rate_limit_store: Dict[str, list] = defaultdict(list)
_rate_limit_lock  = threading.Lock()

def _check_rate_limit(ip: str) -> bool:
    now = time.time()
    with _rate_limit_lock:
        _rate_limit_store[ip] = [t for t in _rate_limit_store[ip] if now - t < 60]
        if len(_rate_limit_store[ip]) < 10:
            _rate_limit_store[ip].append(now)
            return True
    return False

# ── Lazy: Pinecone ────────────────────────────────────────────────
_pinecone_index = None
_pinecone_lock  = threading.Lock()

def _init_pinecone():
    global _pinecone_index
    if _pinecone_index is None:
        with _pinecone_lock:
            if _pinecone_index is None:
                from pinecone import Pinecone as _PC
                _pinecone_index = _PC(api_key=PINECONE_API_KEY).Index(INDEX_NAME)
                log.info(f"✅ Pinecone ready — {INDEX_NAME}")
    return _pinecone_index

async def get_index():
    if _pinecone_index is None:
        await asyncio.to_thread(_init_pinecone)
    return _pinecone_index

# ── Lazy: Gemini ──────────────────────────────────────────────────
_gemini_client = None
_gemini_lock   = threading.Lock()

def _get_gemini():
    global _gemini_client
    if _gemini_client is None:
        with _gemini_lock:
            if _gemini_client is None:
                from google import genai as _g
                _gemini_client = _g.Client(api_key=GEMINI_API_KEY)
                log.info("✅ Gemini ready")
    return _gemini_client

def _gtypes():
    from google.genai import types
    return types

# ── Lazy: SentenceTransformer ─────────────────────────────────────
_st_model      = None
_st_lock       = threading.Lock()
_st_load_error = None

def _encode_sync(text: str) -> List[float]:
    global _st_model, _st_load_error
    if _st_model is None:
        with _st_lock:
            if _st_model is None:
                if _st_load_error:
                    raise RuntimeError(_st_load_error)
                try:
                    from sentence_transformers import SentenceTransformer
                    t0 = time.perf_counter()
                    m  = SentenceTransformer(EMBED_MODEL, device="cpu")
                    try:
                        import torch
                        torch.set_num_threads(1)
                        torch.set_grad_enabled(False)
                    except ImportError:
                        pass
                    _st_model = m
                    log.info(f"✅ Embedder ready in {time.perf_counter()-t0:.1f}s — {EMBED_MODEL}")
                except Exception as e:
                    _st_load_error = str(e)
                    raise
    vec = _st_model.encode(
        text, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    ).tolist()
    if len(vec) != EMBED_DIM:
        raise RuntimeError(f"Dim mismatch: {len(vec)} vs {EMBED_DIM}")
    return vec

async def _encode(text: str) -> List[float]:
    return await asyncio.to_thread(_encode_sync, text)

# ── Constants ─────────────────────────────────────────────────────
GEMINI_MODELS        = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.0-flash-lite"]
GEMINI_VISION_MODELS = ["gemini-2.0-flash", "gemini-1.5-flash"]
ALLOWED_IMAGE_TYPES  = frozenset({"image/jpeg","image/png","image/webp","image/heic","image/heif"})
MEDICAL_DISCLAIMER   = "⚠️ هذه المعلومات للتوجيه العام فقط ولا تُغني عن استشارة طبيب متخصص."

WEIGHT_COSINE   = 0.60
WEIGHT_EXACT    = 0.30
WEIGHT_CATEGORY = 0.10
EXACT_THRESH    = 0.85

MEDICAL_KEYWORDS = frozenset({
    "ألم","وجع","مرض","دواء","طبيب","مستشفى","أعراض","علاج","صداع","حمى",
    "سعال","ضغط","سكر","قلب","كلى","معدة","عظام","جلد","عين","رئة","كبد",
    "دم","تعب","دوار","غثيان","إسهال","إمساك","حرقة","حرارة","التهاب",
    "برد","انفلونزا","زكام","كحة","حبوب","طفح","حكة","جراحة","تحليل","أشعة",
    "pain","fever","cough","headache","nausea","dizzy","vomit","diarrhea",
    "symptom","disease","doctor","medicine","blood","heart","lung","kidney",
    "liver","diabetes","pressure","infection","allergy","rash","fatigue",
    "breathe","chest","stomach","throat","surgery","scan","cold","flu",
    "بول","تبول","إفرازات","نزيف","جرح","كسر","خلع","ورم","سرطان","غدة",
    "ضيق","تنفس","قصور","فشل","خدر","تنميل","رجفة","شلل","صرع","دوخة",
    "urine","bleeding","wound","fracture","tumor","cancer","swelling","numbness",
    "seizure","paralysis","tremor","discharge","abscess","cyst","inflammation",
})

EMERGENCY_AR = frozenset({
    "ألم صدر","ضيق تنفس","نوبة قلبية","سكتة دماغية","نزيف شديد","إغماء",
    "فقدان وعي","صدمة","حروق شديدة","ألم حاد","طوارئ","إسعاف","تسمم",
    "جرح عميق","ألم بطن حاد","شلل","تشنج","انتحار","إيذاء النفس",
    "لا يتنفس","توقف القلب","ضربة شمس حادة","حساسية حادة",
})
EMERGENCY_EN = frozenset({
    "chest pain","difficulty breathing","heart attack","stroke","severe bleeding",
    "fainting","loss of consciousness","shock","severe burns","severe pain",
    "emergency","ambulance","poisoning","deep wound","paralysis","seizure",
    "suicide","self harm","not breathing","cardiac arrest","anaphylaxis",
})

GARBAGE_PATTERNS = frozenset({
    "تم الاجابة","راجع الطبيب","استشر طبيب","كل شيء ممكن","غير واضح",
    "طبيعي","لا يوجد","لا اعرف","لا أستطيع","معلومات محدودة",
    "answered","consult doctor","not clear","not available","don't know","cannot",
})

SYMPTOM_CATEGORY_MAP: Dict[str, List[str]] = {
    "حلق":["respiratory","general"],"سعال":["respiratory","general"],
    "كحة":["respiratory","general"],"رئة":["respiratory"],
    "ربو":["respiratory"],"زكام":["respiratory"],"ضيق تنفس":["respiratory","cardiology"],
    "حرارة":["general","pediatrics"],"حمى":["general","pediatrics"],
    "برد":["general","respiratory"],"انفلونزا":["general","respiratory"],
    "قلب":["cardiology"],"صدر":["cardiology","respiratory"],"ضغط":["cardiology"],
    "صداع":["neurology","general"],"دوار":["neurology"],
    "معدة":["gastroenterology"],"بطن":["gastroenterology"],
    "إسهال":["gastroenterology"],"غثيان":["gastroenterology"],"كبد":["gastroenterology"],
    "جلد":["dermatology"],"طفح":["dermatology"],"حكة":["dermatology"],
    "عظام":["orthopedic"],"مفاصل":["orthopedic"],"ظهر":["orthopedic"],
    "كلى":["urology"],"بول":["urology"],
    "عين":["ophthalmology"],"نظر":["ophthalmology"],
    "طفل":["pediatrics"],"رضيع":["pediatrics"],"أطفال":["pediatrics"],
    "قلق":["psychology"],"اكتئاب":["psychology"],
    "سكري":["endocrinology"],"غدة":["endocrinology"],
    "حمل":["gynecology"],"دورة":["gynecology"],
    "أسنان":["dentistry"],"لثة":["dentistry"],
    "throat":["respiratory"],"cough":["respiratory"],"fever":["general","pediatrics"],
    "heart":["cardiology"],"chest":["cardiology","respiratory"],
    "headache":["neurology"],"stomach":["gastroenterology"],
    "skin":["dermatology"],"rash":["dermatology"],
    "bone":["orthopedic"],"kidney":["urology"],"eye":["ophthalmology"],
    "child":["pediatrics"],"anxiety":["psychology"],"diabetes":["endocrinology"],
    "pregnancy":["gynecology"],"tooth":["dentistry"],
}

MAX_CONTEXT = 3
SEP         = "━" * 48

ARABIC_NORM = str.maketrans({'أ':'ا','إ':'ا','آ':'ا','ة':'ه','ى':'ي','ؤ':'و','ئ':'ي'})

# ── Text helpers ──────────────────────────────────────────────────
def _norm(text: str) -> str:
    t = re.sub(r"[\u064b-\u065f\u0670\u0640]", "", text)
    t = t.translate(ARABIC_NORM).lower()
    t = re.sub(r"[^\w\u0600-\u06ff\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def _jaccard(a: str, b: str) -> float:
    ta, tb = set(_norm(a).split()), set(_norm(b).split())
    if not ta or not tb: return 0.0
    return len(ta & tb) / len(ta | tb)

def _expected_cats(query: str) -> List[str]:
    q, cnt = _norm(query), {}
    for kw, cats in SYMPTOM_CATEGORY_MAP.items():
        if _norm(kw) in q:
            for c in cats: cnt[c] = cnt.get(c, 0) + 1
    return sorted(cnt, key=lambda c: cnt[c], reverse=True)

def _hybrid_score(cosine, qn, mq, mcat, ecats):
    sim = _jaccard(qn, mq)
    if sim >= EXACT_THRESH:
        eb, mt = 1.0, ("exact" if sim == 1.0 else "high_sim")
    elif sim >= 0.65:
        eb, mt = sim * 0.6, "partial"
    else:
        eb, mt = 0.0, "semantic"
    cb = 0.0
    if mcat and ecats:
        cl = mcat.lower()
        ec = [c.lower() for c in ecats]
        if cl in ec: cb = 1.0 if cl == ec[0] else 0.6
    final = WEIGHT_COSINE*cosine + WEIGHT_EXACT*eb + WEIGHT_CATEGORY*cb
    return round(final, 4), mt

def _is_arabic(text: str) -> bool:
    ar = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
    return ar / max(len(text.strip()), 1) > 0.25

def _medical_tokens(text: str) -> set:
    t = re.sub(r"[\u064b-\u065f\u0670]", "", text.lower()).translate(ARABIC_NORM)
    tokens = set(x for x in re.split(r"[\s\W]+", t) if len(x) >= 3)
    return {tok for tok in tokens if any(kw in tok or tok in kw for kw in MEDICAL_KEYWORDS) or len(tok) >= 4}

# ── Domain models ─────────────────────────────────────────────────
@dataclass
class KBMatch:
    question:   str
    answer:     str
    confidence: float
    category:   Optional[str] = None

    @property
    def is_garbage(self) -> bool:
        a = self.answer.strip()
        if len(a) < 12: return True
        al = a.lower()
        return any(p.lower() in al for p in GARBAGE_PATTERNS)

@dataclass
class QCtx:
    query:    str
    language: str
    matches:  List[KBMatch] = field(default_factory=list)
    history:  List[Dict]    = field(default_factory=list)

# ── Pydantic ──────────────────────────────────────────────────────
class MsgDto(BaseModel):
    role: str; content: str

class AskReq(BaseModel):
    question: Optional[str] = None
    text:     Optional[str] = None
    history:  Optional[List[MsgDto]] = None

    @property
    def query(self) -> str: return (self.question or self.text or "").strip()

    @model_validator(mode="after")
    def _chk(self):
        if not self.query: raise ValueError("Empty query")
        if len(self.query) > MAX_QUERY_LENGTH: raise ValueError("Query too long")
        return self

class MatchOut(BaseModel):
    question: str; answer: str; confidence: float; category: Optional[str] = None

class AskResp(BaseModel):
    query: str; reply: str; model_used: str
    matches: List[MatchOut]
    is_medical: bool; found_in_database: bool
    low_confidence: bool; language: str; disclaimer: str

# ── KB Service ────────────────────────────────────────────────────
class KBService:

    @staticmethod
    def _clean(matches: List[KBMatch]) -> List[KBMatch]:
        seen, out = set(), []
        for m in matches:
            if m.is_garbage: continue
            k = m.answer.lower().strip()
            if k in seen: continue
            seen.add(k)
            out.append(m)
        return out

    @staticmethod
    def _top(matches: List[KBMatch], n=MAX_CONTEXT) -> List[KBMatch]:
        s, cats, sel = sorted(matches, key=lambda m: m.confidence, reverse=True), set(), []
        for m in s:
            if len(sel) >= n: break
            if m.category not in cats or len(sel) < 2:
                sel.append(m)
                if m.category: cats.add(m.category)
        return sel

    @staticmethod
    def _relevance_ok(query: str, matches: List[KBMatch]) -> bool:
        if not matches: return False
        qn = _norm(query)
        for m in matches[:3]:
            if _jaccard(qn, _norm(m.question)) >= EXACT_THRESH:
                return True
        ecats = _expected_cats(query)
        if ecats and (matches[0].category or "").lower() in [c.lower() for c in ecats]:
            return True
        qt = _medical_tokens(query)
        if not qt: return True
        mt = _medical_tokens(" ".join(f"{m.question} {m.answer}" for m in matches[:3]))
        return len(qt & mt) >= 1

    async def search(self, query: str, top_k: int = TOP_K) -> List[KBMatch]:
        ecats = _expected_cats(query)
        log.info(f"[KB] query='{query[:60]}' expected_cats={ecats}")
        try:
            vec = await _encode(query)
        except Exception as e:
            log.error(f"[KB] encode failed: {e}")
            return []

        idx     = await get_index()
        fetch_k = min(top_k * 3, 30)
        qn      = _norm(query)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                fn  = functools.partial(
                    idx.query, vector=vec, top_k=fetch_k,
                    include_metadata=True,
                    namespace=PINECONE_NAMESPACE if PINECONE_NAMESPACE else "",
                )
                res = await asyncio.to_thread(fn)
                cands = []

                for m in res.matches:
                    cosine = float(m.score or 0)
                    if cosine < MIN_CONFIDENCE * 0.75: continue
                    meta = m.metadata or {}
                    qt   = meta.get("question", "")
                    at   = meta.get("answer",   "")
                    cat  = meta.get("category", "general")
                    fs, mt = _hybrid_score(cosine, qn, qt, cat, ecats)
                    icon = "🎯" if mt=="exact" else "🔍" if mt=="high_sim" else "🌐"
                    log.info(f"[KB] {icon} {mt:10s} cos={cosine:.3f} → final={fs:.4f} cat={cat} q='{qt[:45]}'")
                    cands.append(KBMatch(question=qt, answer=at, confidence=fs, category=cat))

                cands.sort(key=lambda x: x.confidence, reverse=True)
                kept = [c for c in cands if c.confidence >= MIN_CONFIDENCE]
                log.info(f"[KB] kept={len(kept)} total_cands={len(cands)}")
                return kept[:top_k]

            except Exception as e:
                log.warning(f"[KB] attempt {attempt} failed: {e}")
                if attempt < MAX_RETRIES: await asyncio.sleep(RETRY_DELAY * attempt)
        return []

# ── Gemini Service ────────────────────────────────────────────────
class GeminiSvc:
    def __init__(self):
        self._cache: collections.OrderedDict = collections.OrderedDict()
        self._lock  = threading.Lock()

    def _cache_get(self, k):
        with self._lock:
            if k not in self._cache: return None
            self._cache.move_to_end(k)
            return self._cache[k]

    def _cache_put(self, k, v):
        with self._lock:
            self._cache[k] = v
            self._cache.move_to_end(k)
            if len(self._cache) > 150: self._cache.popitem(last=False)

    def _gen_sync(self, prompt: str) -> Tuple[str, str]:
        ck = hashlib.sha256(prompt.encode()).hexdigest()
        cv = self._cache_get(ck)
        if cv: return cv

        types = _gtypes()
        cfg   = types.GenerateContentConfig(temperature=0.2, max_output_tokens=2048)

        for model in GEMINI_MODELS:
            for attempt in range(1, 3):
                try:
                    log.info(f"[Gemini] {model} attempt {attempt}")
                    r = _get_gemini().models.generate_content(
                        model=model,
                        contents=prompt,
                        config=cfg,
                    )
                    result = (r.text.strip(), model)
                    self._cache_put(ck, result)
                    return result
                except Exception as e:
                    log.warning(f"[Gemini] {model} attempt {attempt} failed: {e}")
                    if attempt < 2: time.sleep(2 ** (attempt - 1))

        log.error("[Gemini] all models failed — safe fallback")
        safe = (
            "بناءً على الأعراض، قد تكون الحالة ناتجة عن عدة أسباب. "
            "أنصح بمراجعة طبيب متخصص للتقييم الدقيق. 🏥"
        )
        return safe, "safe_fallback"

    async def generate(self, prompt: str) -> Tuple[str, str]:
        return await asyncio.to_thread(self._gen_sync, prompt)

    def _social_sync(self, query: str, lang: str) -> Tuple[str, str]:
        if lang == "ar":
            sys_inst = (
                "أنت 'سيلا'، مساعد طبي ذكي وودود. "
                "رد بالعربية بشكل طبيعي ودافئ في جملة أو اثنتين. "
                "لو السؤال غير طبي تماماً فرد بشكل اجتماعي مناسب."
            )
        else:
            sys_inst = (
                "You are 'Sila', a friendly medical AI assistant. "
                "Reply naturally in English in 1-2 sentences. "
                "If the message is social/greeting, respond warmly."
            )

        types = _gtypes()
        for model in GEMINI_MODELS:
            try:
                r = _get_gemini().models.generate_content(
                    model=model,
                    contents=query,
                    config=types.GenerateContentConfig(
                        system_instruction=sys_inst,
                        temperature=0.7,
                        max_output_tokens=200,
                    ),
                )
                return r.text.strip(), model
            except Exception as e:
                log.warning(f"[Social] {model}: {e}")
        fb = "أهلاً! أنا سيلا، مساعدتك الطبية. 😊" if lang=="ar" else "Hello! I'm Sila, your medical AI. 😊"
        return fb, "fallback"

    async def reply_social(self, query: str, lang: str) -> Tuple[str, str]:
        return await asyncio.to_thread(self._social_sync, query, lang)

    # ── Image analysis ────────────────────────────────────────────
    def _image_sync(self, image_bytes: bytes, mime_type: str, lang: str) -> Tuple[str, str, str]:
        """
        Analyze medical image.
        - Non-medical → polite rejection in the user's language
        - Medical → structured bilingual analysis (Arabic + English medical terms)
        """
        sys_prompt = """You are a specialized medical image analysis AI.

ACCEPTED: lab results, blood tests, prescriptions, X-rays, MRI, CT scans, ECG, ultrasound, pathology slides, medical reports.

STEP 1 — Decide if medical:
- If NOT a medical image → return ONLY: {"status": "rejected", "analysis": "not_medical"}
- If medical → continue to step 2

STEP 2 — Analyze and return JSON:
{"status": "success", "analysis": "<your analysis>"}

FORMAT for analysis (Arabic explanation + English medical terms):
📋 نوع الفحص / Document Type: [English term]

🔍 النتائج الرئيسية / Key Findings:
→ [Arabic sentence] [English values/units, e.g. Hemoglobin: 11.2 g/dL]

⚠️ قيم خارج المعدل / Abnormal Values:
→ [Arabic explanation] [Reference range in English, e.g. Normal: 13.5–17.5 g/dL]

💊 التوصيات / Recommendations:
→ [Arabic + relevant English specialist names]

🚨 نتائج عاجلة / Urgent Findings:
→ [If none: لا توجد نتائج عاجلة واضحة / No urgent findings]

RULES:
- NEVER give a definitive diagnosis
- Keep medical values and terms in English, explanations in Arabic
- Return ONLY valid JSON — no markdown, no extra text"""

        types    = _gtypes()
        b64_data = base64.b64encode(image_bytes).decode("utf-8")

        # Two content formats to try for compatibility
        content_formats = [
            # Format 1: Part objects
            lambda: [
                sys_prompt,
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
            # Format 2: dict-based (fallback)
            lambda: [
                {"text": sys_prompt},
                {"inline_data": {"mime_type": mime_type, "data": b64_data}},
            ],
        ]

        for model in GEMINI_VISION_MODELS:
            for fmt_fn in content_formats:
                try:
                    contents = fmt_fn()
                    log.info(f"[Vision] trying {model}")
                    r = _get_gemini().models.generate_content(
                        model=model,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            temperature=0.1,
                            max_output_tokens=3000,
                        ),
                    )
                    raw   = r.text.strip()
                    clean = re.sub(r"^\s*```+(?:json)?\s*|\s*```+\s*$", "", raw, flags=re.MULTILINE).strip()
                    brace = clean.find("{")
                    if brace > 0: clean = clean[brace:]

                    try:
                        parsed   = json.loads(clean)
                        status   = str(parsed.get("status", "success"))
                        analysis = parsed.get("analysis", raw)

                        if status == "rejected" or analysis == "not_medical":
                            msg = (
                                "هذه الصورة لا تبدو صورة طبية. يرجى رفع صورة طبية مثل نتيجة تحليل، وصفة دواء، أو أشعة."
                                if lang == "ar" else
                                "This doesn't appear to be a medical image. Please upload a medical image such as a lab result, prescription, or X-ray."
                            )
                            return "rejected", msg, model

                        if not isinstance(analysis, str):
                            analysis = json.dumps(analysis, ensure_ascii=False, indent=2)

                        disclaimer = (
                            "\n\n⚠️ تنبيه: هذا التحليل للاسترشاد فقط ولا يُغني عن استشارة طبيب متخصص."
                            if lang == "ar" else
                            "\n\n⚠️ Note: This analysis is for guidance only and does not replace professional medical advice."
                        )
                        return "success", analysis.strip() + disclaimer, model

                    except json.JSONDecodeError:
                        # Best-effort: if raw is substantial, treat as success
                        if len(raw) > 100:
                            disclaimer = (
                                "\n\n⚠️ تنبيه: هذا التحليل للاسترشاد فقط."
                                if lang == "ar" else
                                "\n\n⚠️ For guidance only."
                            )
                            return "success", raw + disclaimer, model

                except Exception as e:
                    log.warning(f"[Vision] {model} format failed: {e}")
                    continue  # try next format/model

        err = (
            "عذراً، تعذّر تحليل الصورة مؤقتاً. يرجى المحاولة مرة أخرى."
            if lang == "ar" else
            "Sorry, image analysis is temporarily unavailable. Please try again."
        )
        return "error", err, "none"

    async def analyze_image(self, image_bytes: bytes, mime_type: str, lang: str = "ar") -> Tuple[str, str, str]:
        return await asyncio.to_thread(self._image_sync, image_bytes, mime_type, lang)

# ── Prompt Builder ────────────────────────────────────────────────
class Prompts:

    @staticmethod
    def _ctx_block(matches: List[KBMatch], lang: str) -> str:
        parts = []
        for i, m in enumerate(matches, 1):
            rel = "✅ موثوق" if m.confidence >= MIN_CONFIDENCE else "⚠️ ثقة منخفضة"
            if lang == "en":
                rel = "✅ Reliable" if m.confidence >= MIN_CONFIDENCE else "⚠️ Low confidence"
            parts.append(
                f"[{i}] {rel} ({m.confidence:.0%}) | Specialty: {m.category or 'General'}\n"
                f"Q: {m.question}\nA: {m.answer}"
            )
        return f"\n\n{SEP}\n".join(parts)

    @staticmethod
    def _history_block(history: List[Dict], lang: str) -> str:
        if not history: return ""
        label = "📜 سياق المحادثة السابقة:" if lang == "ar" else "📜 Previous conversation:"
        lines = [label]
        for h in history[-4:]:  # last 4 turns max
            role = "المريض" if h.get("role") == "user" else "سيلا"
            if lang == "en":
                role = "Patient" if h.get("role") == "user" else "Sila"
            lines.append(f"{role}: {h.get('content', '')[:200]}")
        return "\n".join(lines) + f"\n{SEP}"

    @staticmethod
    def _sys(lang: str) -> str:
        if lang == "ar":
            return (
                "أنت 'سيلا'، مساعد طبي ذكي وموثوق. أسلوبك دافئ واحترافي.\n\n"
                "قواعد:\n"
                "١. استخدم المعلومات المقدمة كمرجع أساسي.\n"
                "٢. لا تُقدم تشخيصاً نهائياً — قدّم احتمالات فقط.\n"
                "٣. اذكر علامات الخطر إن وُجدت.\n"
                "٤. اختم بالتوصية بمراجعة طبيب متخصص.\n"
                "٥. الرد بالعربية دائماً."
            )
        return (
            "You are 'Sila', a trusted medical AI assistant. Warm and professional tone.\n\n"
            "Rules:\n"
            "1. Use provided context as primary reference.\n"
            "2. NEVER give definitive diagnosis — suggest possibilities.\n"
            "3. Flag warning signs if present.\n"
            "4. Always recommend consulting a specialist.\n"
            "5. Reply in English always."
        )

    @staticmethod
    def _struct(lang: str) -> str:
        if lang == "ar":
            return (
                "🔍 الأسباب المحتملة:\n→ ...\n\n"
                "💊 التوصيات:\n→ ...\n\n"
                "🏥 التخصص المناسب:\n→ ...\n\n"
                "⚠️ علامات الخطر:\n→ ..."
            )
        return (
            "🔍 Possible Causes:\n→ ...\n\n"
            "💊 Recommendations:\n→ ...\n\n"
            "🏥 Recommended Specialist:\n→ ...\n\n"
            "⚠️ Warning Signs:\n→ ..."
        )

    def rag(self, ctx: QCtx) -> str:
        lang     = ctx.language
        hist     = self._history_block(ctx.history, lang)
        kb_label = "📋 قاعدة المعرفة الطبية:" if lang=="ar" else "📋 Medical Knowledge Base:"
        q_label  = "🧑‍⚕️ سؤال المريض:" if lang=="ar" else "🧑‍⚕️ Patient Question:"
        a_label  = "الإجابة:" if lang=="ar" else "Answer:"
        return (
            f"{self._sys(lang)}\n\n{SEP}\n"
            f"{hist}\n" if hist else ""
            f"{kb_label}\n\n{self._ctx_block(ctx.matches, lang)}\n\n{SEP}\n"
            f"{q_label}\n{ctx.query}\n\n{self._struct(lang)}\n\n{a_label}"
        )

    def gemini_only(self, query: str, lang: str, history: Optional[List[Dict]] = None) -> str:
        hist = self._history_block(history or [], lang)
        sys_ = (
            "أنت 'سيلا'، مساعد طبي ذكي. لا يوجد سياق من قاعدة البيانات.\n"
            "قدّم إجابة طبية مفيدة بناءً على معرفتك العامة.\n"
            "لا تُقدم تشخيصاً نهائياً. اختم بالتوصية بمراجعة طبيب. الرد بالعربية."
            if lang == "ar" else
            "You are 'Sila', a trusted medical AI. No database context available.\n"
            "Provide a helpful medical response based on general knowledge.\n"
            "Never diagnose definitively. Always recommend a specialist. Reply in English."
        )
        q_label = "🧑‍⚕️ سؤال المريض:" if lang=="ar" else "🧑‍⚕️ Patient Question:"
        a_label = "الإجابة:" if lang=="ar" else "Answer:"
        hist_section = f"{hist}\n" if hist else ""
        return f"{sys_}\n\n{SEP}\n{hist_section}{q_label}\n{query}\n\n{self._struct(lang)}\n\n{a_label}"

    def emergency(self, query: str, lang: str) -> str:
        sys_ = (
            "أنت 'سيلا'. 🚨 المريض يعاني من أعراض طارئة.\n"
            "قدّم توجيهاً عاجلاً. أوصِ بالطوارئ أو الإسعاف فوراً. كن مختصراً وواضحاً."
            if lang == "ar" else
            "You are 'Sila'. 🚨 Patient has emergency symptoms.\n"
            "Give immediate guidance. Strongly recommend ER or ambulance. Be concise and clear."
        )
        q_label = "🧑‍⚕️ سؤال المريض:" if lang=="ar" else "🧑‍⚕️ Patient Question:"
        a_label = "الإجابة:" if lang=="ar" else "Answer:"
        return f"{sys_}\n\n{SEP}\n{q_label}\n{query}\n\n{self._struct(lang)}\n\n{a_label}"

    @staticmethod
    def safe_fallback(lang: str) -> str:
        return (
            "بناءً على الأعراض المذكورة، قد تكون الحالة ناتجة عن عدة أسباب. "
            "أنصح بمراجعة طبيب متخصص للتقييم الدقيق. 🏥"
            if lang == "ar" else
            "Based on the symptoms, this could have several causes. "
            "Please consult a specialist for proper evaluation. 🏥"
        )

# ── Intent classifier ─────────────────────────────────────────────
def _is_medical_kw(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in MEDICAL_KEYWORDS)

async def _classify_intent(query: str) -> str:
    """
    Fast keyword check first, then Gemini only if uncertain.
    FIX: generate_content takes model + contents, not 3 positional args.
    """
    if _is_medical_kw(query):
        return "medical"
    # Short greetings / clearly social
    social_ar = {"مرحبا","اهلا","هلا","السلام","صباح","مساء","شكرا","شكراً","كيف حالك"}
    social_en = {"hi","hello","hey","thanks","thank","how are","good morning","good evening"}
    q_lower = query.lower().strip()
    if any(q_lower.startswith(s) for s in social_ar | social_en):
        return "social"
    if len(query.split()) <= 4:
        # Short query that doesn't match medical — probably social
        return "social"
    # Gemini classification for edge cases
    try:
        types = _gtypes()
        r = await asyncio.to_thread(
            lambda: _get_gemini().models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=f"Is this message medical or social? Reply ONE word only: medical or social\nMessage: {query}",
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=5),
            )
        )
        return "medical" if "medical" in r.text.lower() else "social"
    except Exception as e:
        log.warning(f"[Intent] classification failed: {e} — defaulting social")
        return "social"

def _is_emergency(query: str, lang: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in (EMERGENCY_AR if lang=="ar" else EMERGENCY_EN))

def _detect_lang_from_request(request: Optional[Request], query: str = "") -> str:
    """Detect language from query text first, then Accept-Language header."""
    if query and _is_arabic(query):
        return "ar"
    if request:
        al = request.headers.get("Accept-Language", "")
        if al.lower().startswith("ar"):
            return "ar"
    return "en" if query else "ar"  # default Arabic

# ── App state ─────────────────────────────────────────────────────
class _State:
    kb:      Optional[KBService]   = None
    gemini:  Optional[GeminiSvc]   = None
    prompts: Optional[Prompts]     = None

_state = _State()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _request_semaphore
    _request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    _state.kb      = KBService()
    _state.gemini  = GeminiSvc()
    _state.prompts = Prompts()
    log.info(
        f"🚀 SILA v17.1 ready | model={EMBED_MODEL} | "
        f"dim={EMBED_DIM} | min_conf={MIN_CONFIDENCE} | index={INDEX_NAME}"
    )
    yield
    log.info("🛑 SILA shutting down")

# ── FastAPI ───────────────────────────────────────────────────────
app = FastAPI(
    title="Sila — Medical AI v17.1",
    description="Hybrid RAG · Multilingual · Arabic-Aware · Railway-Safe",
    version="17.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def _err(req: Request, exc: Exception):
    log.error(f"❌ {type(exc).__name__}: {exc}")
    return JSONResponse(status_code=500, content={"error": "Unexpected error. Please try again."})

@app.get("/")
def root():
    return {
        "name":    "Sila Medical AI",
        "version": "17.1.0",
        "status":  "running ✅",
        "embed":   EMBED_MODEL,
    }

@app.get("/health")
def health():
    return {
        "status":         "ok",
        "version":        "17.1.0",
        "embed_model":    EMBED_MODEL,
        "embed_dim":      EMBED_DIM,
        "index":          INDEX_NAME,
        "min_confidence": MIN_CONFIDENCE,
    }

# ── /ask ──────────────────────────────────────────────────────────
@app.post("/ask", response_model=AskResp)
async def ask(req: AskReq, request: Request):
    if _request_semaphore is None:
        return JSONResponse(status_code=503, content={"error": "Service not ready"})

    # Language: detect from query text first
    lang = "ar" if _is_arabic(req.query) else "en"
    ip   = request.client.host if request.client else "unknown"

    if not _check_rate_limit(ip):
        msg = "تجاوزت الحد المسموح به. حاول بعد دقيقة." if lang=="ar" else "Rate limit exceeded. Try again in a minute."
        return JSONResponse(status_code=429, content={"error": msg})

    await _request_semaphore.acquire()
    try:
        return await asyncio.wait_for(_ask_inner(req, lang), timeout=EXTERNAL_CALL_TIMEOUT)
    except asyncio.TimeoutError:
        msg = "انتهت مهلة الطلب. يرجى المحاولة مجدداً." if lang=="ar" else "Request timed out. Please try again."
        return AskResp(
            query=req.query, reply=msg, model_used="none", matches=[],
            is_medical=True, found_in_database=False, low_confidence=True,
            language=lang, disclaimer=MEDICAL_DISCLAIMER,
        )
    finally:
        try: _request_semaphore.release()
        except: pass

async def _ask_inner(req: AskReq, lang: str) -> AskResp:
    t0      = time.time()
    q       = req.query
    history = [h.model_dump() for h in (req.history or [])]
    intent  = await _classify_intent(q)
    log.info(f"[ASK] '{q[:70]}' lang={lang} intent={intent}")

    # Social
    if intent == "social":
        reply, model = await _state.gemini.reply_social(q, lang)
        log.info(f"[ASK] social — {model} {round((time.time()-t0)*1000)}ms")
        return AskResp(
            query=q, reply=reply, model_used=model, matches=[],
            is_medical=False, found_in_database=False, low_confidence=False,
            language=lang, disclaimer=MEDICAL_DISCLAIMER,
        )

    # Emergency check
    is_emg = _is_emergency(q, lang)
    if is_emg: log.warning("[ASK] 🚨 EMERGENCY detected")

    # Retrieve
    try:
        raw = await _state.kb.search(q, top_k=TOP_K)
    except Exception as e:
        log.error(f"[ASK] KB failed: {e}")
        raw = []

    clean    = KBService._clean(raw)
    selected = KBService._top(clean, MAX_CONTEXT)
    top_sc   = selected[0].confidence if selected else 0.0
    rel_ok   = KBService._relevance_ok(q, selected)
    garbage  = sum(1 for m in raw if m.is_garbage) / max(len(raw), 1)

    # Mode decision
    if is_emg:
        mode = "EMERGENCY"
    elif not selected or top_sc < MIN_CONFIDENCE or not rel_ok or garbage > 0.5:
        mode = "GEMINI_ONLY"
    elif top_sc >= 0.72:
        mode = "RAG_STRONG"
    else:
        mode = "RAG_LIGHT"

    log.info(f"[ASK] mode={mode} top_sc={top_sc:.3f} rel={rel_ok} garbage={garbage:.2f} history={len(history)}")

    match_out = [
        MatchOut(
            question=m.question,
            answer=m.answer,
            confidence=m.confidence,
            category=m.category,
        )
        for m in selected
    ]

    ctx = QCtx(query=q, language=lang, matches=selected, history=history)

    # Build prompt
    if mode == "EMERGENCY":
        prompt = _state.prompts.emergency(q, lang)
    elif mode in ("RAG_STRONG", "RAG_LIGHT"):
        prompt = _state.prompts.rag(ctx)
    else:
        prompt = _state.prompts.gemini_only(q, lang, history)

    reply, model = await _state.gemini.generate(prompt)

    log.info(f"[ASK] done mode={mode} model={model} {round((time.time()-t0)*1000)}ms")

    return AskResp(
        query=q,
        reply=reply,
        model_used=model,
        matches=match_out,
        is_medical=True,
        found_in_database=mode in ("RAG_STRONG", "RAG_LIGHT"),
        low_confidence=mode in ("RAG_LIGHT", "GEMINI_ONLY"),
        language=lang,
        disclaimer=MEDICAL_DISCLAIMER,
    )

# ── /analyze-image ────────────────────────────────────────────────
@app.post("/analyze-image")
async def analyze_image(
    file: UploadFile = File(...),
    lang: Optional[str] = Form(None),
    request: Request = None,
):
    """
    Analyze a medical image.
    Language detection priority:
    1. `lang` form field (ar/en)
    2. Accept-Language header
    3. Default: ar
    """
    # Detect language
    if lang and lang in ("ar", "en"):
        detected_lang = lang
    elif request:
        al = request.headers.get("Accept-Language", "")
        detected_lang = "en" if al.lower().startswith("en") else "ar"
    else:
        detected_lang = "ar"

    # Validate content type
    ct = file.content_type or ""
    if ct not in ALLOWED_IMAGE_TYPES:
        msg = (
            f"نوع الملف '{ct}' غير مدعوم. الأنواع المقبولة: JPEG, PNG, WEBP, HEIC."
            if detected_lang == "ar" else
            f"File type '{ct}' not supported. Accepted: JPEG, PNG, WEBP, HEIC."
        )
        return JSONResponse(
            status_code=400,
            content={"status":"error","analysis":msg,"model_used":"none","disclaimer":MEDICAL_DISCLAIMER},
        )

    # Read file
    try:
        image_bytes = await file.read()
    except Exception:
        msg = "فشل في قراءة الملف." if detected_lang=="ar" else "Failed to read file."
        return JSONResponse(
            status_code=400,
            content={"status":"error","analysis":msg,"model_used":"none","disclaimer":MEDICAL_DISCLAIMER},
        )

    if not image_bytes:
        msg = "الملف فارغ." if detected_lang=="ar" else "File is empty."
        return JSONResponse(
            status_code=400,
            content={"status":"error","analysis":msg,"model_used":"none","disclaimer":MEDICAL_DISCLAIMER},
        )

    if len(image_bytes) > MAX_IMAGE_BYTES:
        msg = (
            f"حجم الصورة ({len(image_bytes)//1024//1024}MB) يتجاوز الحد الأقصى {MAX_IMAGE_MB}MB."
            if detected_lang=="ar" else
            f"Image size ({len(image_bytes)//1024//1024}MB) exceeds limit of {MAX_IMAGE_MB}MB."
        )
        return JSONResponse(
            status_code=413,
            content={"status":"error","analysis":msg,"model_used":"none","disclaimer":MEDICAL_DISCLAIMER},
        )

    log.info(f"[IMG] {file.filename} {len(image_bytes)//1024}KB lang={detected_lang} type={ct}")

    try:
        status, analysis, model = await asyncio.wait_for(
            _state.gemini.analyze_image(image_bytes, ct, detected_lang),
            timeout=EXTERNAL_CALL_TIMEOUT,
        )
    except asyncio.TimeoutError:
        msg = (
            "انتهت مهلة تحليل الصورة. يرجى المحاولة مجدداً."
            if detected_lang=="ar" else
            "Image analysis timed out. Please try again."
        )
        return JSONResponse(
            status_code=504,
            content={"status":"error","analysis":msg,"model_used":"none","disclaimer":MEDICAL_DISCLAIMER},
        )

    http_code = 200 if status in ("success", "rejected") else 503
    return JSONResponse(
        status_code=http_code,
        content={
            "status":     status,
            "analysis":   analysis,
            "model_used": model,
            "disclaimer": MEDICAL_DISCLAIMER,
        },
    )

# ── Entry ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        log_level="info",
    )