import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import google.generativeai as genai


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "medical-index"

if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY environment variable")
if not PINECONE_API_KEY:
    raise RuntimeError("Missing PINECONE_API_KEY environment variable")

genai.configure(api_key=GEMINI_API_KEY)

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

MEDICAL_SYSTEM_PROMPT = """
أنت مساعد طبي ذكي ومتخصص، مدرب على بيانات طبية عربية دقيقة.
مهمتك هي تقديم إجابات طبية احترافية وواضحة باللغة العربية.

قواعد يجب الالتزام بها دائماً:
1. إذا كان السؤال غير طبي تماماً (مثل الطبخ أو الرياضة أو السياسة)، أخبر المريض بلطف أنك مخصص للاستشارات الطبية فقط.
2. استخدم السياق الطبي المقدم كمرجع أساسي، لكن لا تقتصر عليه — استخدم معرفتك الطبية لإثراء الإجابة.
3. قدم إجابة منظمة تشمل: الأسباب المحتملة، التوصيات العملية، وعلامات الخطر إن وجدت.
4. تحدث بأسلوب دافئ وإنساني كما يفعل الطبيب مع مريضه.
5. لا تضع تشخيصاً نهائياً، لكن قدم توجيهاً طبياً مفيداً وواضحاً.
6. اذكر دائماً متى يجب التوجه للطوارئ أو الطبيب فوراً.
""".strip()


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

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
    description="مساعد طبي ذكي مدعوم بالذكاء الاصطناعي",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    text: str


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


def search_pinecone(query: str, top_k: int = 5) -> List[MatchResult]:
    vector = embed_model.encode(query).tolist()
    result = pinecone_index.query(vector=vector, top_k=top_k, include_metadata=True)

    matches = []
    for m in result.matches:
        meta = m.metadata or {}
        matches.append(MatchResult(
            symptom=meta.get("symptom", ""),
            reply=meta.get("reply", ""),
            category=meta.get("category", ""),
            confidence=float(m.score),
        ))
    return matches


def build_prompt(query: str, matches: List[MatchResult]) -> str:
    context_parts = []
    for i, m in enumerate(matches):
        context_parts.append(
            f"[حالة {i+1}]\n"
            f"الأعراض: {m.symptom}\n"
            f"التوجيه الطبي: {m.reply}\n"
            f"التصنيف: {m.category}\n"
            f"درجة التشابه: {m.confidence:.0%}"
        )
    context = "\n\n".join(context_parts)

    return (
        f"{MEDICAL_SYSTEM_PROMPT}\n\n"
        "─────────────────────────────\n"
        "📋 حالات طبية مشابهة من قاعدة البيانات:\n\n"
        f"{context}\n\n"
        "─────────────────────────────\n"
        f"🔹 سؤال المريض: {query}\n\n"
        "قدم إجابة طبية احترافية ومنظمة تشمل:\n"
        "• **الأسباب المحتملة** بناءً على الأعراض\n"
        "• **التوصيات العملية** التي يمكن للمريض اتباعها الآن\n"
        "• **التخصص الطبي المناسب** للمراجعة\n"
        "• **علامات الخطر** التي تستدعي التوجه للطوارئ فوراً\n\n"
        "الإجابة:"
    )


def ask_gemini(prompt: str) -> tuple[str, str]:
    last_error = None
    for model_name in GEMINI_MODELS:
        try:
            log.info(f"Trying Gemini model: {model_name}")
            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=genai.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=1024,
                ),
            )
            response = model.generate_content(prompt)
            return response.text.strip(), model_name
        except Exception as e:
            log.warning(f"Model {model_name} failed: {e}")
            last_error = e
    return f"عذراً، حدث خطأ مؤقت. يرجى المحاولة مرة أخرى.", "none"


def is_medical_query(query: str, matches: List[MatchResult]) -> bool:
    if matches and matches[0].confidence >= 0.75:
        return True
    medical_keywords = [
        "ألم", "وجع", "مرض", "دواء", "طبيب", "مستشفى", "أعراض", "علاج",
        "صداع", "حمى", "سعال", "ضغط", "سكر", "قلب", "كلى", "معدة",
        "عظام", "جلد", "عين", "أذن", "أنف", "رئة", "كبد", "دم",
        "تعب", "إرهاق", "دوار", "غثيان", "إسهال", "إمساك", "حرقة",
        "الم", "وجع", "عندي", "عندى", "اشعر", "أشعر", "احس", "أحس",
    ]
    query_lower = query.lower()
    return any(kw in query_lower for kw in medical_keywords)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="النص فارغ")

    matches = search_pinecone(req.text, top_k=5)
    medical = is_medical_query(req.text, matches)

    if not medical:
        return AskResponse(
            query=req.text,
            gemini_reply=(
                "أنا مساعد طبي متخصص، ويسعدني مساعدتك في الاستفسارات الطبية والصحية فقط. 🏥\n"
                "إذا كان لديك أي سؤال يتعلق بأعراض أو أمراض أو أدوية أو توجيهات طبية، فأنا هنا لمساعدتك."
            ),
            model_used="none",
            matches=[],
            low_confidence=True,
            is_medical=False,
        )

    prompt = build_prompt(req.text, matches)
    reply, model_used = ask_gemini(prompt)

    return AskResponse(
        query=req.text,
        gemini_reply=reply,
        model_used=model_used,
        matches=matches,
        low_confidence=matches[0].confidence < 0.75 if matches else True,
        is_medical=True,
    )


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


@app.get("/")
def root():
    return {
        "name": "Medical AI Assistant",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
    }