"""
Medical AI Assistant — Strict RAG Edition
==========================================
Answers ONLY from Pinecone knowledge base.
If no relevant match is found, it says so clearly.
"""

import hashlib
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import List, Optional

import google.generativeai as genai
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pinecone import Pinecone
from pydantic import BaseModel, field_validator
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = os.getenv("PINECONE_INDEX", "medical-index")
MIN_CONFIDENCE   = float(os.getenv("MIN_CONFIDENCE", "0.70"))
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "500"))

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY environment variable.")
if not PINECONE_API_KEY:
    raise RuntimeError("Missing PINECONE_API_KEY environment variable.")

genai.configure(api_key=GEMINI_API_KEY)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("medical_ai")


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

MEDICAL_DISCLAIMER = (
    "⚠️ تنبيه: هذه المعلومات للتوجيه العام فقط ولا تُغني عن استشارة طبيب متخصص."
)

MEDICAL_KEYWORDS_AR = {
    "ألم", "وجع", "مرض", "دواء", "طبيب", "مستشفى", "أعراض", "علاج",
    "صداع", "حمى", "سعال", "ضغط", "سكر", "قلب", "كلى", "معدة",
    "عظام", "جلد", "عين", "أذن", "أنف", "رئة", "كبد", "دم",
    "تعب", "إرهاق", "دوار", "غثيان", "إسهال", "إمساك", "حرقة",
    "الم", "عندي", "عندى", "اشعر", "احس", "اعاني", "يؤلم",
    "بوجعني", "بتوجعني", "حاسس", "حاسه", "حبوب", "طفح", "حكة",
}

MEDICAL_KEYWORDS_EN = {
    "pain", "ache", "fever", "cough", "headache", "nausea", "dizzy",
    "vomit", "diarrhea", "symptom", "disease", "doctor", "hospital",
    "medicine", "drug", "blood", "heart", "lung", "kidney", "liver",
    "diabetes", "pressure", "infection", "allergy", "rash", "swelling",
    "fatigue", "tired", "breathe", "chest", "stomach", "throat",
}


# ─────────────────────────────────────────────
# Domain Models
# ─────────────────────────────────────────────

@dataclass
class KnowledgeMatch:
    symptom: str
    reply: str
    category: str
    confidence: float

    @property
    def is_reliable(self) -> bool:
        return self.confidence >= MIN_CONFIDENCE


@dataclass
class QueryContext:
    raw_query: str
    language: str
    is_medical: bool
    matches: List[KnowledgeMatch] = field(default_factory=list)

    @property
    def has_reliable_matches(self) -> bool:
        return any(m.is_reliable for m in self.matches)

    @property
    def best_confidence(self) -> float:
        return self.matches[0].confidence if self.matches else 0.0


# ─────────────────────────────────────────────
# Pydantic Schemas (API layer)
# ─────────────────────────────────────────────

class AskRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty.")
        if len(v) > MAX_QUERY_LENGTH:
            raise ValueError(f"Query exceeds maximum length of {MAX_QUERY_LENGTH} characters.")
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
    found_in_database: bool
    disclaimer: str
    language: str


# ─────────────────────────────────────────────
# Service Layer
# ─────────────────────────────────────────────

class LanguageDetector:
    @staticmethod
    def detect(text: str) -> str:
        arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
        ratio = arabic_chars / max(len(text), 1)
        return "ar" if ratio > 0.3 else "en"


class MedicalClassifier:
    @staticmethod
    def is_medical(query: str, matches: List[KnowledgeMatch]) -> bool:
        if matches and matches[0].confidence >= MIN_CONFIDENCE:
            return True
        q = query.lower()
        return any(kw in q for kw in MEDICAL_KEYWORDS_AR | MEDICAL_KEYWORDS_EN)


class KnowledgeBaseService:
    def __init__(self, index, embed_model: SentenceTransformer):
        self._index = index
        self._embed_model = embed_model

    def search(self, query: str, top_k: int = 5) -> List[KnowledgeMatch]:
        vector = self._embed_model.encode(query).tolist()
        result = self._index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
        )
        matches = []
        for m in result.matches:
            # Filter out very weak matches early
            if m.score < MIN_CONFIDENCE * 0.75:
                continue
            meta = m.metadata or {}
            matches.append(KnowledgeMatch(
                symptom=meta.get("symptom", ""),
                reply=meta.get("reply", ""),
                category=meta.get("category", ""),
                confidence=round(float(m.score), 4),
            ))
        return matches


class PromptBuilder:
    """Builds strict RAG prompts — Gemini answers ONLY from provided context."""

    _SYSTEM_AR = (
        "أنت مساعد طبي ذكي ومتخصص. مهمتك تقديم ردود طبية احترافية ودقيقة.\n"
        "القواعد الصارمة:\n"
        "1. أجب فقط بناءً على الحالات الطبية المقدمة في السياق أدناه.\n"
        "2. لا تستخدم أي معرفة خارجية أو افتراضات من تلقاء نفسك.\n"
        "3. إذا كانت المعلومات المتاحة غير كافية للإجابة، قل ذلك بوضوح.\n"
        "4. لا تضع تشخيصاً نهائياً — قدم توجيهاً مبنياً على البيانات المتاحة.\n"
        "5. أذكر دائماً متى يجب التوجه للطوارئ إن وُجد ما يستدعي ذلك في السياق."
    )

    _SYSTEM_EN = (
        "You are a specialized medical AI assistant. Your task is to provide accurate, professional responses.\n"
        "Strict rules:\n"
        "1. Answer ONLY based on the medical cases provided in the context below.\n"
        "2. Do NOT use external knowledge or personal assumptions.\n"
        "3. If the available information is insufficient, state that clearly.\n"
        "4. Do NOT give a final diagnosis — provide guidance based on available data.\n"
        "5. Always mention when emergency care is needed if the context suggests it."
    )

    _NO_DATA_RESPONSE_AR = (
        "عذراً، لا تتوفر في قاعدة بياناتنا الطبية معلومات كافية لهذه الحالة بالتحديد.\n\n"
        "**نصيحتنا:**\n"
        "• تواصل مع طبيب متخصص للحصول على تقييم دقيق لحالتك.\n"
        "• إذا كانت الأعراض حادة أو مفاجئة، توجه للطوارئ فوراً.\n\n"
        "_سيتم توسيع قاعدة بياناتنا باستمرار لتغطية حالات أكثر._"
    )

    _NO_DATA_RESPONSE_EN = (
        "We're sorry, our medical database doesn't contain sufficient information for this specific case.\n\n"
        "**Our recommendation:**\n"
        "• Please consult a qualified physician for a proper evaluation.\n"
        "• If symptoms are severe or sudden, seek emergency care immediately.\n\n"
        "_Our database is continuously expanding to cover more cases._"
    )

    def get_no_data_response(self, language: str) -> str:
        return self._NO_DATA_RESPONSE_AR if language == "ar" else self._NO_DATA_RESPONSE_EN

    def build(self, ctx: QueryContext) -> str:
        system = self._SYSTEM_AR if ctx.language == "ar" else self._SYSTEM_EN
        lang_note = "أجب باللغة العربية فقط." if ctx.language == "ar" else "Answer in English only."

        context_parts = []
        for i, m in enumerate(ctx.matches):
            context_parts.append(
                f"[حالة {i + 1} — ثقة: {m.confidence:.0%}]\n"
                f"الأعراض: {m.symptom}\n"
                f"التوجيه الطبي: {m.reply}\n"
                f"التصنيف: {m.category}"
            )
        context_block = "\n\n".join(context_parts)

        structure_note = (
            "قدم إجابة منظمة تشمل (بناءً على السياق المتاح فقط):\n"
            "• الأسباب المحتملة\n"
            "• التوصيات الفورية\n"
            "• التخصص الطبي المناسب\n"
            "• علامات الخطر إن وُجدت"
            if ctx.language == "ar" else
            "Provide a structured response covering (based on available context only):\n"
            "• Possible causes\n"
            "• Immediate recommendations\n"
            "• Appropriate medical specialty\n"
            "• Warning signs if applicable"
        )

        return (
            f"{system}\n{lang_note}\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "📋 السياق الطبي المتاح:\n\n"
            f"{context_block}\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🔹 سؤال المريض: {ctx.raw_query}\n\n"
            f"{structure_note}\n\n"
            "الإجابة:"
        )


class GeminiService:
    def __init__(self):
        self._cache: dict[str, tuple[str, str]] = {}

    def generate(self, prompt: str) -> tuple[str, str]:
        cache_key = hashlib.md5(prompt.encode()).hexdigest()
        if cache_key in self._cache:
            log.info("Cache hit for prompt.")
            return self._cache[cache_key]

        last_error = None
        for model_name in GEMINI_MODELS:
            try:
                log.info(f"Calling model: {model_name}")
                model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=genai.GenerationConfig(
                        temperature=0.2,          # Lower temp = more faithful to context
                        max_output_tokens=2048,
                    ),
                )
                response = model.generate_content(prompt)
                text = response.text.strip()
                self._cache[cache_key] = (text, model_name)
                return text, model_name
            except Exception as e:
                log.warning(f"Model {model_name} failed: {e}")
                last_error = e

        log.error(f"All Gemini models failed. Last error: {last_error}")
        return "عذراً، حدث خطأ مؤقت في الخدمة. يرجى المحاولة مرة أخرى.", "none"

    @property
    def cache_size(self) -> int:
        return len(self._cache)


# ─────────────────────────────────────────────
# Application State
# ─────────────────────────────────────────────

class AppState:
    knowledge_base: Optional[KnowledgeBaseService] = None
    gemini: Optional[GeminiService] = None
    prompt_builder: Optional[PromptBuilder] = None

state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("🔧 Loading embedding model...")
    embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    log.info("🔌 Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    state.knowledge_base = KnowledgeBaseService(index, embed_model)
    state.gemini = GeminiService()
    state.prompt_builder = PromptBuilder()

    log.info("✅ Medical AI Assistant is ready.")
    yield
    log.info("🛑 Shutdown complete.")


# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────

app = FastAPI(
    title="Medical AI Assistant",
    description="مساعد طبي ذكي — Strict RAG + Gemini",
    version="4.0.0",
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
    log.error(f"Unhandled error on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error. Please try again later."},
    )


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    language = LanguageDetector.detect(req.text)
    matches  = state.knowledge_base.search(req.text, top_k=5)
    is_medical = MedicalClassifier.is_medical(req.text, matches)

    # ── Not a medical question ──────────────────
    if not is_medical:
        reply = (
            "أنا مساعد طبي متخصص، ويسعدني مساعدتك في الاستفسارات الطبية والصحية فقط. 🏥\n"
            "إذا كان لديك سؤال عن أعراض، أمراض، أدوية، أو توجيهات طبية — فأنا هنا."
            if language == "ar" else
            "I'm a specialized medical AI assistant. I can only help with health and medical questions. "
            "Feel free to ask about symptoms, conditions, medications, or medical guidance."
        )
        return AskResponse(
            query=req.text,
            gemini_reply=reply,
            model_used="none",
            matches=[],
            low_confidence=True,
            is_medical=False,
            found_in_database=False,
            disclaimer=MEDICAL_DISCLAIMER,
            language=language,
        )

    ctx = QueryContext(
        raw_query=req.text,
        language=language,
        is_medical=True,
        matches=matches,
    )

    # ── No reliable matches → honest response ──
    if not ctx.has_reliable_matches:
        reply = state.prompt_builder.get_no_data_response(language)
        return AskResponse(
            query=req.text,
            gemini_reply=reply,
            model_used="none",
            matches=[MatchResult(**m.__dict__) for m in matches],
            low_confidence=True,
            is_medical=True,
            found_in_database=False,
            disclaimer=MEDICAL_DISCLAIMER,
            language=language,
        )

    # ── Reliable matches found → generate answer ─
    prompt = state.prompt_builder.build(ctx)
    reply, model_used = state.gemini.generate(prompt)

    return AskResponse(
        query=req.text,
        gemini_reply=reply,
        model_used=model_used,
        matches=[MatchResult(**m.__dict__) for m in matches],
        low_confidence=False,
        is_medical=True,
        found_in_database=True,
        disclaimer=MEDICAL_DISCLAIMER,
        language=language,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "4.0.0",
        "cache_size": state.gemini.cache_size if state.gemini else 0,
        "min_confidence_threshold": MIN_CONFIDENCE,
    }


@app.get("/")
def root():
    return {
        "name": "Medical AI Assistant",
        "version": "4.0.0",
        "mode": "strict-rag",
        "status": "running",
        "docs": "/docs",
    }