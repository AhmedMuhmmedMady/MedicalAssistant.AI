import hashlib
import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional

import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pinecone import Pinecone
from pydantic import BaseModel, field_validator
from sentence_transformers import SentenceTransformer

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "medical-index")
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.70"))
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "500"))

if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY")
if not PINECONE_API_KEY:
    raise RuntimeError("Missing PINECONE_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

MEDICAL_DISCLAIMER = (
    "⚠️ تنبيه: هذه المعلومات للتوجيه فقط ولا تُغني عن استشارة طبيب متخصص."
)

SYSTEM_PROMPT = """أنت مساعد طبي ذكي ومتخصص، مدرب على بيانات طبية عربية دقيقة.
مهمتك تقديم إجابات طبية احترافية باللغة العربية أو الإنجليزية حسب لغة السؤال.

القواعد:
1. إذا كان السؤال غير طبي، أخبر المستخدم بلطف أنك مخصص للاستشارات الطبية فقط.
2. استخدم السياق الطبي المقدم كمرجع أساسي وأثرِه بمعرفتك الطبية.
3. لا تضع تشخيصاً نهائياً، لكن قدم توجيهاً طبياً مفيداً وواضحاً.
4. تحدث بأسلوب دافئ وإنساني كما يفعل الطبيب مع مريضه.
5. اذكر دائماً متى يجب التوجه للطوارئ أو الطبيب فوراً."""

MEDICAL_KEYWORDS_AR = {
    "ألم", "وجع", "مرض", "دواء", "طبيب", "مستشفى", "أعراض", "علاج",
    "صداع", "حمى", "سعال", "ضغط", "سكر", "قلب", "كلى", "معدة",
    "عظام", "جلد", "عين", "أذن", "أنف", "رئة", "كبد", "دم",
    "تعب", "إرهاق", "دوار", "غثيان", "إسهال", "إمساك", "حرقة",
    "الم", "عندي", "عندى", "اشعر", "احس", "اعاني", "يؤلم", "عندو",
    "عندع", "عندها", "بوجعني", "بتوجعني", "حاسس", "حاسه",
}

MEDICAL_KEYWORDS_EN = {
    "pain", "ache", "fever", "cough", "headache", "nausea", "dizzy",
    "vomit", "diarrhea", "symptom", "disease", "doctor", "hospital",
    "medicine", "drug", "blood", "heart", "lung", "kidney", "liver",
    "diabetes", "pressure", "infection", "allergy", "rash", "swelling",
    "fatigue", "tired", "breathe", "chest", "stomach", "throat",
}

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

_cache: dict[str, str] = {}
embed_model: Optional[SentenceTransformer] = None
pinecone_index = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embed_model, pinecone_index
    log.info("Loading embedding model...")
    embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    log.info("Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(INDEX_NAME)
    log.info("System ready.")
    yield
    log.info("Shutdown complete.")


app = FastAPI(
    title="Medical AI Assistant",
    description="مساعد طبي ذكي مدعوم بالذكاء الاصطناعي — RAG + Gemini",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error. Please try again later."},
    )


class AskRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty")
        if len(v) > MAX_QUERY_LENGTH:
            raise ValueError(f"Query too long (max {MAX_QUERY_LENGTH} characters)")
        return v


class MatchResult(BaseModel):
    symptom: str
    reply: str
    category: str
    confidence: float


class AskResponse(BaseModel):
    query: str
    gemini_reply: str
    model_used: str
    matches: List[MatchResult]
    low_confidence: bool
    is_medical: bool
    disclaimer: str
    language: str


def detect_language(text: str) -> str:
    arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
    return "ar" if arabic_chars / max(len(text), 1) > 0.3 else "en"


def is_medical_query(query: str, matches: List[MatchResult]) -> bool:
    if matches and matches[0].confidence >= MIN_CONFIDENCE:
        return True
    q = query.lower()
    return any(kw in q for kw in MEDICAL_KEYWORDS_AR | MEDICAL_KEYWORDS_EN)


def search_pinecone(query: str, top_k: int = 5) -> List[MatchResult]:
    vector = embed_model.encode(query).tolist()
    result = pinecone_index.query(vector=vector, top_k=top_k, include_metadata=True)
    matches = []
    for m in result.matches:
        if m.score < MIN_CONFIDENCE * 0.8:
            continue
        meta = m.metadata or {}
        matches.append(MatchResult(
            symptom=meta.get("symptom", ""),
            reply=meta.get("reply", ""),
            category=meta.get("category", ""),
            confidence=round(float(m.score), 4),
        ))
    return matches


def build_prompt(query: str, matches: List[MatchResult], language: str) -> str:
    context_parts = []
    for i, m in enumerate(matches):
        context_parts.append(
            f"[حالة {i + 1}]\n"
            f"الأعراض: {m.symptom}\n"
            f"التوجيه الطبي: {m.reply}\n"
            f"التصنيف: {m.category}\n"
            f"درجة التشابه: {m.confidence:.0%}"
        )
    context = "\n\n".join(context_parts) if context_parts else "لا توجد حالات مشابهة في قاعدة البيانات."

    lang_note = "أجب باللغة العربية." if language == "ar" else "Answer in English."

    return (
        f"{SYSTEM_PROMPT}\n{lang_note}\n\n"
        "─────────────────────────────\n"
        "📋 حالات طبية مشابهة:\n\n"
        f"{context}\n\n"
        "─────────────────────────────\n"
        f"🔹 سؤال المريض: {query}\n\n"
        "قدم إجابة طبية منظمة تشمل:\n"
        "• الأسباب المحتملة بناءً على الأعراض\n"
        "• التوصيات العملية الفورية\n"
        "• التخصص الطبي المناسب للمراجعة\n"
        "• علامات الخطر التي تستدعي الطوارئ فوراً\n\n"
        "الإجابة:"
    )


def ask_gemini(prompt: str) -> tuple[str, str]:
    cache_key = hashlib.md5(prompt.encode()).hexdigest()
    if cache_key in _cache:
        log.info("Cache hit")
        return _cache[cache_key], "cache"

    last_error = None
    for model_name in GEMINI_MODELS:
        try:
            log.info(f"Trying model: {model_name}")
            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=genai.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=2048,
                ),
            )
            response = model.generate_content(prompt)
            text = response.text.strip()
            _cache[cache_key] = text
            return text, model_name
        except Exception as e:
            log.warning(f"Model {model_name} failed: {e}")
            last_error = e

    log.error(f"All models failed: {last_error}")
    return "عذراً، حدث خطأ مؤقت في الخدمة. يرجى المحاولة مرة أخرى.", "none"


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    language = detect_language(req.text)
    matches = search_pinecone(req.text, top_k=5)
    medical = is_medical_query(req.text, matches)

    if not medical:
        reply = (
            "أنا مساعد طبي متخصص، ويسعدني مساعدتك في الاستفسارات الطبية والصحية فقط. 🏥\n"
            "إذا كان لديك أي سؤال يتعلق بأعراض أو أمراض أو أدوية أو توجيهات طبية، فأنا هنا لمساعدتك."
            if language == "ar" else
            "I'm a medical AI assistant. I can only help with health and medical questions. "
            "Please ask me about symptoms, conditions, medications, or medical guidance."
        )
        return AskResponse(
            query=req.text,
            gemini_reply=reply,
            model_used="none",
            matches=[],
            low_confidence=True,
            is_medical=False,
            disclaimer=MEDICAL_DISCLAIMER,
            language=language,
        )

    prompt = build_prompt(req.text, matches, language)
    reply, model_used = ask_gemini(prompt)

    return AskResponse(
        query=req.text,
        gemini_reply=reply,
        model_used=model_used,
        matches=matches,
        low_confidence=not matches or matches[0].confidence < MIN_CONFIDENCE,
        is_medical=True,
        disclaimer=MEDICAL_DISCLAIMER,
        language=language,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "3.0.0",
        "cache_size": len(_cache),
    }


@app.get("/")
def root():
    return {
        "name": "Medical AI Assistant",
        "version": "3.0.0",
        "status": "running",
        "docs": "/docs",
    }