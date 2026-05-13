"""
╔══════════════════════════════════════════════════════════════════╗
║          SILA — Medical AI Assistant  v8.0                      ║
║          Graduation Project — Production Ready 🚀               ║
║                                                                  ║
║  Stack : FastAPI + Pinecone + Gemini (google-genai SDK)         ║
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
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import List, Optional

# ──────────────────────────────────────────────────────────────────
# Third-Party
# ──────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
from pinecone import Pinecone
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────────
load_dotenv()

GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = os.getenv("PINECONE_INDEX", "sila-medical")
MIN_CONFIDENCE   = float(os.getenv("MIN_CONFIDENCE", "0.55"))
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "500"))
MAX_IMAGE_MB     = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
MAX_IMAGE_BYTES  = MAX_IMAGE_MB * 1024 * 1024

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is not set. Add it to your .env file.")

# ──────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("sila.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("sila")

# ──────────────────────────────────────────────────────────────────
# Gemini Client  (single module-level instance)
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

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/heic",
    "image/heif",
}

MEDICAL_DISCLAIMER = (
    "⚠️ تنبيه: هذه المعلومات للتوجيه العام فقط "
    "ولا تُغني عن استشارة طبيب متخصص."
)

# Arabic + English medical keyword sets for fallback classification
MEDICAL_KEYWORDS: set[str] = {
    # Arabic
    "ألم", "وجع", "مرض", "دواء", "طبيب", "مستشفى", "أعراض", "علاج",
    "صداع", "حمى", "سعال", "ضغط", "سكر", "قلب", "كلى", "معدة",
    "عظام", "جلد", "عين", "أذن", "أنف", "رئة", "كبد", "دم",
    "تعب", "إرهاق", "دوار", "غثيان", "إسهال", "إمساك", "حرقة",
    "الم", "عندي", "عندى", "اشعر", "احس", "اعاني", "يؤلم",
    "بوجعني", "بتوجعني", "حاسس", "حاسه", "حبوب", "طفح", "حكة",
    # English
    "pain", "ache", "fever", "cough", "headache", "nausea", "dizzy",
    "vomit", "diarrhea", "symptom", "disease", "doctor", "hospital",
    "medicine", "drug", "blood", "heart", "lung", "kidney", "liver",
    "diabetes", "pressure", "infection", "allergy", "rash", "swelling",
    "fatigue", "tired", "breathe", "chest", "stomach", "throat",
}


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


# ──────────────────────────────────────────────────────────────────
# Pydantic Schemas
# ──────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    text: Optional[str] = None
    question: Optional[str] = None

    @property
    def query(self) -> str:
        return (self.text or self.question or "").strip()

    def model_post_init(self, __context) -> None:
        if not self.query:
            raise ValueError("Request must include a non-empty 'text' or 'question' field.")
        if len(self.query) > MAX_QUERY_LENGTH:
            raise ValueError(f"Query exceeds {MAX_QUERY_LENGTH} characters.")


class MatchResult(BaseModel):
    question: str
    answer: str
    confidence: float


class AskResponse(BaseModel):
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
    status: str
    analysis: Optional[str] = None
    model_used: Optional[str] = None
    disclaimer: str = MEDICAL_DISCLAIMER


# ──────────────────────────────────────────────────────────────────
# Services
# ──────────────────────────────────────────────────────────────────

class LanguageDetector:
    """Detects Arabic vs English based on character ratio."""

    @staticmethod
    def detect(text: str) -> str:
        arabic = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
        return "ar" if arabic / max(len(text), 1) > 0.25 else "en"


class IntentClassifier:
    """
    Two-layer classification:
      1. Fast Gemini LLM call (cheap model, 1-word answer)
      2. Keyword fallback if Gemini fails
    Returns: 'social' | 'medical'
    """

    _PROMPT = (
        "Classify this message into exactly one category:\n"
        "- social  → greetings, thanks, casual talk, non-medical\n"
        "- medical → symptoms, diseases, medications, body, pain, health\n\n"
        "Reply with ONE word only: social OR medical\n\n"
        "Message: {query}"
    )

    @classmethod
    def classify(cls, query: str) -> str:
        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=cls._PROMPT.format(query=query),
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=5,
                ),
            )
            result = response.text.strip().lower()
            if "social" in result:
                return "social"
            if "medical" in result:
                return "medical"
        except Exception as e:
            log.warning(f"IntentClassifier LLM failed, using keyword fallback: {e}")

        # Keyword fallback
        q_lower = query.lower()
        return "medical" if any(kw in q_lower for kw in MEDICAL_KEYWORDS) else "social"


class KnowledgeBaseService:
    """Searches Pinecone for relevant medical Q&A pairs."""

    def __init__(self, index, embed_model: SentenceTransformer):
        self._index = index
        self._model = embed_model

    def search(self, query: str, top_k: int = 5) -> List[KnowledgeMatch]:
        vector = self._model.encode(query).tolist()
        results = self._index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
        )
        matches = []
        for m in results.matches:
            # Skip very low scores early to keep noise out
            if m.score < MIN_CONFIDENCE * 0.70:
                continue
            meta = m.metadata or {}
            matches.append(KnowledgeMatch(
                question=meta.get("question", ""),
                answer=meta.get("answer", ""),
                confidence=round(float(m.score), 4),
            ))
        return matches


class PromptBuilder:
    """Builds the full RAG prompt sent to Gemini."""

    # ── System instructions ─────────────────────────────────────────
    _SYSTEM = {
        "ar": (
            "أنت 'سيلا'، مساعد طبي ذكي وموثوق.\n"
            "أسلوبك: دافئ واحترافي، كأنك طبيب يشرح لمريضه باهتمام.\n\n"
            "قواعد صارمة:\n"
            "١. استخدم فقط المعلومات الطبية المقدمة في السياق أدناه.\n"
            "٢. لا تستخدم أي معرفة خارجية أو افتراضات.\n"
            "٣. إذا كانت المعلومات غير كافية، قل صراحةً: "
            "'معلوماتي محدودة في هذه الحالة، يُرجى استشارة طبيب.'\n"
            "٤. لا تُقدم تشخيصاً نهائياً أبداً.\n"
            "٥. اذكر علامات الخطر إن وُجدت.\n"
            "٦. اختم دائماً بالتوصية بمراجعة طبيب متخصص.\n\n"
            "لغة الإجابة: عربية فصحى واضحة (أو حسب طلب المستخدم)."
        ),
        "en": (
            "You are 'Sila', a trusted medical AI assistant.\n"
            "Tone: warm and professional.\n\n"
            "Strict rules:\n"
            "1. Use ONLY the medical context provided below.\n"
            "2. Do NOT use external knowledge or assumptions.\n"
            "3. If data is insufficient say clearly: "
            "'My knowledge is limited on this case. Please consult a doctor.'\n"
            "4. NEVER provide a final diagnosis.\n"
            "5. Highlight warning signs if present.\n"
            "6. Always end by recommending a specialist consultation.\n\n"
            "Language: English (or follow user's request)."
        ),
    }

    # ── Structure instructions ──────────────────────────────────────
    _STRUCTURE = {
        "ar": (
            "قدّم إجابتك بهذا التنسيق:\n"
            "• الأسباب المحتملة\n"
            "• التوصيات والخطوات العملية\n"
            "• التخصص الطبي المناسب\n"
            "• علامات الخطر التي تستدعي الطوارئ (إن وجدت)"
        ),
        "en": (
            "Structure your response as:\n"
            "• Possible causes\n"
            "• Recommendations & next steps\n"
            "• Relevant medical specialty\n"
            "• Warning signs requiring emergency care (if any)"
        ),
    }

    _NO_DATA = {
        "ar": (
            "لا تتوفر لديّ معلومات كافية في قاعدة بياناتي لهذه الحالة تحديداً. "
            "أنصحك بمراجعة طبيب متخصص للحصول على تقييم دقيق."
        ),
        "en": (
            "I don't have sufficient information in my database for this specific case. "
            "I recommend consulting a specialist for a proper evaluation."
        ),
    }

    def build(self, ctx: QueryContext) -> str:
        lang = ctx.language
        system = self._SYSTEM.get(lang, self._SYSTEM["ar"])
        structure = self._STRUCTURE.get(lang, self._STRUCTURE["ar"])

        context_parts = []
        for i, m in enumerate(ctx.matches, start=1):
            context_parts.append(
                f"[Case {i} — Confidence: {m.confidence:.0%}]\n"
                f"Q: {m.question}\n"
                f"A: {m.answer}"
            )
        context_block = "\n\n".join(context_parts)

        return (
            f"{system}\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "📋 Medical Knowledge Base Context:\n\n"
            f"{context_block}\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🧑‍⚕️ Patient Question:\n{ctx.raw_query}\n\n"
            f"{structure}\n\n"
            "Answer:"
        )

    def no_data_response(self, language: str) -> str:
        return self._NO_DATA.get(language, self._NO_DATA["ar"])


class GeminiService:
    """
    All Gemini interactions: social chat, RAG generation, image analysis.
    - Model fallback chain for resilience
    - MD5-based response cache to avoid redundant API calls
    """

    def __init__(self):
        self._cache: dict[str, tuple[str, str]] = {}

    # ── Shared config ───────────────────────────────────────────────
    @staticmethod
    def _cfg(temperature: float = 0.2, max_tokens: int = 2048) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    # ── Social / greeting reply ─────────────────────────────────────
    def reply_social(self, query: str, language: str) -> tuple[str, str]:
        """
        Warm, friendly reply for non-medical messages.
        Egyptian Arabic by default, English if detected.
        """
        if language == "en":
            system = (
                "You are 'Sila', a friendly medical AI assistant. "
                "Reply warmly and naturally in English. "
                "If the topic is non-medical, politely explain your specialization. "
                "Keep replies short (1-2 sentences)."
            )
        else:
            system = (
                "أنت 'سيلا'، مساعد طبي ذكي وودود. "
                "ردودك دايماً بالعامية المصرية الدافية والطبيعية. "
                "لو الكلام مش طبي، بلطف قول إنك متخصص في الاستشارات الطبية. "
                "الرد قصير (جملة أو اتنين بالكتير)."
            )

        for model_name in GEMINI_TEXT_MODELS:
            try:
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=query,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=0.7,
                        max_output_tokens=150,
                    ),
                )
                return response.text.strip(), model_name
            except Exception as e:
                log.warning(f"Social reply: model {model_name} failed — {e}")

        fallback = (
            "Hello! 😊 I'm Sila, your medical assistant. How can I help you today?"
            if language == "en"
            else "أهلاً وسهلاً! 😊 أنا سيلا، مساعدتك الطبية. إزيك النهارده؟"
        )
        return fallback, "fallback"

    # ── RAG text generation ─────────────────────────────────────────
    def generate(self, prompt: str) -> tuple[str, str]:
        """Generate a medical RAG response. Cached by prompt hash."""
        cache_key = hashlib.md5(prompt.encode()).hexdigest()
        if cache_key in self._cache:
            log.info("Cache hit — returning cached response.")
            return self._cache[cache_key]

        for model_name in GEMINI_TEXT_MODELS:
            try:
                log.info(f"Generating with: {model_name}")
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=self._cfg(),
                )
                text = response.text.strip()
                self._cache[cache_key] = (text, model_name)
                return text, model_name
            except Exception as e:
                log.warning(f"Generate: model {model_name} failed — {e}")

        log.error("All text models failed.")
        return "عذراً، حدث خطأ مؤقت. يرجى المحاولة مرة أخرى.", "none"

    # ── Vision / image analysis ─────────────────────────────────────
    def analyze_image(self, image_bytes: bytes, mime_type: str) -> tuple[str, str, str]:
        """
        Analyze a medical image. Returns (status, analysis_text, model_used).
        Tries two content-building styles per model for maximum compatibility.
        """
        system_prompt = (
            "You are a specialized medical AI. Analyze ONLY medical images such as:\n"
            "lab results, prescriptions, X-rays, MRI, CT scans, ECG, pathology reports.\n\n"
            "RULES:\n"
            "1. If the image is NOT medical → respond ONLY with JSON:\n"
            '   {"status": "rejected", "analysis": "<explanation>"}\n\n'
            "2. If the image IS medical → respond ONLY with JSON:\n"
            '   {"status": "success", "analysis": "<full structured analysis>"}\n\n'
            "3. Medical analysis must include:\n"
            "   - Document type\n"
            "   - Key findings / values\n"
            "   - Values outside normal range (clearly highlighted)\n"
            "   - Recommended next steps / specialist\n"
            "   - Any urgent findings\n\n"
            "4. NEVER provide a final diagnosis.\n"
            "5. Respond in the language of the text in the image (AR or EN).\n"
            "6. Output ONLY valid JSON — no markdown, no extra text."
        )

        b64_data = base64.b64encode(image_bytes).decode("utf-8")

        for model_name in GEMINI_VISION_MODELS:
            # Try two content formats: new SDK Part vs legacy inline_data dict
            content_options = [
                # Option A — new SDK (preferred)
                [system_prompt, types.Part.from_bytes(data=image_bytes, mime_type=mime_type)],
                # Option B — base64 dict (legacy fallback)
                [
                    {"text": system_prompt},
                    {"inline_data": {"mime_type": mime_type, "data": b64_data}},
                ],
            ]

            for contents in content_options:
                try:
                    log.info(f"Vision analysis: trying {model_name}")
                    response = gemini_client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=self._cfg(max_tokens=4096),
                    )
                    raw = response.text.strip()

                    # Parse JSON response
                    clean = raw.replace("```json", "").replace("```", "").strip()
                    parsed = json.loads(clean)
                    status = parsed.get("status", "success")
                    analysis = parsed.get("analysis", raw)

                    # Unwrap if analysis is itself a nested object
                    if isinstance(analysis, dict):
                        analysis = analysis.get("analysis") or json.dumps(analysis, ensure_ascii=False, indent=2)
                    if not isinstance(analysis, str):
                        analysis = json.dumps(analysis, ensure_ascii=False, indent=2)

                    log.info(f"Vision analysis succeeded with {model_name}")
                    return status, analysis, model_name

                except json.JSONDecodeError:
                    # Model returned plain text — still usable
                    log.warning(f"Vision model {model_name} returned non-JSON, using raw text.")
                    return "success", raw, model_name
                except Exception as e:
                    log.warning(f"Vision model {model_name} / content style failed — {e}")
                    continue

        log.error("All vision models failed.")
        return (
            "error",
            "Medical image analysis is temporarily unavailable. Please try again later.",
            "none",
        )

    @property
    def cache_size(self) -> int:
        return len(self._cache)


# ──────────────────────────────────────────────────────────────────
# Application State  (populated during lifespan startup)
# ──────────────────────────────────────────────────────────────────

class AppState:
    knowledge_base: Optional[KnowledgeBaseService] = None
    gemini: Optional[GeminiService] = None
    prompt_builder: Optional[PromptBuilder] = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("🚀 Sila is starting up...")

    log.info("📦 Loading embedding model: paraphrase-multilingual-MiniLM-L12-v2")
    embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    log.info(f"🔌 Connecting to Pinecone index: {INDEX_NAME}")
    pc    = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    state.knowledge_base = KnowledgeBaseService(index, embed_model)
    state.gemini         = GeminiService()
    state.prompt_builder = PromptBuilder()

    log.info("✅ Sila v8.0 is ready to serve.")
    yield
    log.info("🛑 Sila is shutting down.")


# ──────────────────────────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Sila — Medical AI Assistant",
    description="مساعد طبي ذكي | Strict RAG + Gemini Vision",
    version="8.0.0",
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
    log.error(f"Unhandled error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error. Please try again later."},
    )


# ──────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name":      "Sila — Medical AI Assistant",
        "version":   "8.0.0",
        "status":    "running",
        "endpoints": ["/ask", "/analyze-image", "/health", "/docs"],
    }


@app.get("/health")
def health():
    return {
        "status":            "ok",
        "version":           "8.0.0",
        "cache_size":        state.gemini.cache_size if state.gemini else 0,
        "min_confidence":    MIN_CONFIDENCE,
        "image_analysis":    "enabled",
        "index":             INDEX_NAME,
    }


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """
    Main medical Q&A endpoint.
    Flow:
      1. Detect language
      2. Classify intent (social / medical) via Gemini + keyword fallback
      3. If social → warm reply, no RAG
      4. If medical → search Pinecone → build prompt → Gemini RAG answer
    """
    q        = req.query
    language = LanguageDetector.detect(q)
    intent   = IntentClassifier.classify(q)
    log.info(f"Query: '{q[:80]}' | lang={language} | intent={intent}")

    # ── Social path ──────────────────────────────────────────────────
    if intent == "social":
        reply, model_used = state.gemini.reply_social(q, language)
        return AskResponse(
            query=q, reply=reply, model_used=model_used,
            matches=[], is_medical=False, found_in_database=False,
            low_confidence=False, language=language, disclaimer=MEDICAL_DISCLAIMER,
        )

    # ── Medical path ─────────────────────────────────────────────────
    matches = state.knowledge_base.search(q, top_k=5)
    ctx     = QueryContext(raw_query=q, language=language, matches=matches)

    # No reliable data in the knowledge base
    if not ctx.has_reliable_matches:
        reply = state.prompt_builder.no_data_response(language)
        log.info(f"No reliable matches found (best score: {matches[0].confidence if matches else 0:.2f})")
        return AskResponse(
            query=q, reply=reply, model_used="none",
            matches=[MatchResult(**m.__dict__) for m in matches],
            is_medical=True, found_in_database=False,
            low_confidence=True, language=language, disclaimer=MEDICAL_DISCLAIMER,
        )

    # Build RAG prompt and generate answer
    prompt = state.prompt_builder.build(ctx)
    reply, model_used = state.gemini.generate(prompt)
    log.info(f"RAG answer generated via {model_used} | top confidence: {matches[0].confidence:.2f}")

    return AskResponse(
        query=q, reply=reply, model_used=model_used,
        matches=[MatchResult(**m.__dict__) for m in matches],
        is_medical=True, found_in_database=True,
        low_confidence=False, language=language, disclaimer=MEDICAL_DISCLAIMER,
    )


@app.post("/analyze-image", response_model=ImageAnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    """
    Medical image analysis endpoint.
    Accepts: JPEG, PNG, WEBP, HEIC, HEIF
    Returns: structured analysis or polite rejection if not a medical image.
    """
    # ── Validate MIME type ───────────────────────────────────────────
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        return JSONResponse(
            status_code=400,
            content={
                "status":    "error",
                "analysis":  (
                    f"نوع الملف '{file.content_type}' غير مدعوم. "
                    "المسموح به: JPEG, PNG, WEBP, HEIC."
                ),
                "model_used": "none",
                "disclaimer": MEDICAL_DISCLAIMER,
            },
        )

    # ── Read bytes ───────────────────────────────────────────────────
    try:
        image_bytes = await file.read()
    except Exception as e:
        log.error(f"Failed to read image '{file.filename}': {e}")
        return JSONResponse(
            status_code=400,
            content={
                "status":    "error",
                "analysis":  "فشل في قراءة الملف. تأكد من أن الصورة غير تالفة.",
                "model_used": "none",
                "disclaimer": MEDICAL_DISCLAIMER,
            },
        )

    # ── Validate size ─────────────────────────────────────────────────
    if len(image_bytes) > MAX_IMAGE_BYTES:
        return JSONResponse(
            status_code=413,
            content={
                "status":    "error",
                "analysis":  f"حجم الصورة كبير جداً. الحد الأقصى هو {MAX_IMAGE_MB}MB.",
                "model_used": "none",
                "disclaimer": MEDICAL_DISCLAIMER,
            },
        )

    log.info(
        f"Analyzing image: '{file.filename}' | "
        f"{len(image_bytes) / 1024:.1f} KB | {file.content_type}"
    )

    status, analysis, model_used = state.gemini.analyze_image(
        image_bytes, file.content_type
    )

    http_status = 503 if status == "error" else 200
    return JSONResponse(
        status_code=http_status,
        content={
            "status":     status,
            "analysis":   analysis,
            "model_used": model_used,
            "disclaimer": MEDICAL_DISCLAIMER,
        },
    )