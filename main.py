"""
╔══════════════════════════════════════════════════════════════════╗
║          SILA — Medical AI Assistant  v9.0                      ║
║          Railway-Safe · Memory-Optimized · Lazy Loading         ║
║                                                                  ║
║  Stack : FastAPI + Pinecone + Gemini (google-genai SDK)         ║
║  Mode  : Strict RAG for medical · Social chat for greetings     ║
║  Vision: Medical image analysis via Gemini Vision               ║
╚══════════════════════════════════════════════════════════════════╝

Memory strategy
───────────────
• App boots instantly — NO model loaded at startup.
• SentenceTransformer is loaded lazily on the FIRST /ask request
  via a thread-safe singleton (threading.Lock).
• Torch CPU threads are capped to avoid RAM spikes.
• Lighter model (all-MiniLM-L6-v2, ~90 MB) replaces the heavy
  multilingual one (~470 MB).
  ⚠️  If your Pinecone index was built with a DIFFERENT model, set
      EMBED_MODEL env var to match it exactly.
"""

# ──────────────────────────────────────────────────────────────────
# Standard Library
# ──────────────────────────────────────────────────────────────────
import base64
import hashlib
import json
import logging
import os
import threading
import time
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
EMBED_MODEL      = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

# ── CPU memory cap (critical for Railway free tier) ────────────────
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
try:
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
except Exception:
    pass

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
# Gemini Client  (lightweight HTTP client — no RAM cost)
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

MEDICAL_KEYWORDS: set[str] = {
    "ألم", "وجع", "مرض", "دواء", "طبيب", "مستشفى", "أعراض", "علاج",
    "صداع", "حمى", "سعال", "ضغط", "سكر", "قلب", "كلى", "معدة",
    "عظام", "جلد", "عين", "أذن", "أنف", "رئة", "كبد", "دم",
    "تعب", "إرهاق", "دوار", "غثيان", "إسهال", "إمساك", "حرقة",
    "الم", "عندي", "عندى", "اشعر", "احس", "اعاني", "يؤلم",
    "بوجعني", "بتوجعني", "حاسس", "حاسه", "حبوب", "طفح", "حكة",
    "pain", "ache", "fever", "cough", "headache", "nausea", "dizzy",
    "vomit", "diarrhea", "symptom", "disease", "doctor", "hospital",
    "medicine", "drug", "blood", "heart", "lung", "kidney", "liver",
    "diabetes", "pressure", "infection", "allergy", "rash", "swelling",
    "fatigue", "tired", "breathe", "chest", "stomach", "throat",
}


# ──────────────────────────────────────────────────────────────────
# Lazy Embedding Model Singleton  ← THE core memory fix
# ──────────────────────────────────────────────────────────────────

class _EmbedModelSingleton:
    """
    Thread-safe lazy singleton for SentenceTransformer.

    The model is loaded ONCE on the first call to .get() and reused
    forever. A threading.Lock + double-checked locking ensures that
    even concurrent first-requests only load the model once.

    Why lazy?
    - Railway health-check hits /health immediately after boot.
    - If the model loads at startup, RAM spikes before health-check
      passes → Railway kills the container → deploy fails.
    - Lazy loading lets /health respond in <100ms, then the model
      loads on the first real /ask request.
    """

    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:          # double-checked
                    log.info(f"⏳ Loading embedding model: {EMBED_MODEL}")
                    t0 = time.perf_counter()

                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer(EMBED_MODEL)

                    try:
                        import torch
                        model = model.to(torch.device("cpu"))
                    except Exception:
                        pass

                    elapsed = time.perf_counter() - t0
                    log.info(f"✅ Embedding model ready in {elapsed:.2f}s")
                    cls._instance = model

        return cls._instance


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
    @staticmethod
    def detect(text: str) -> str:
        arabic = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
        return "ar" if arabic / max(len(text), 1) > 0.25 else "en"


class IntentClassifier:
    """
    Layer 1: Gemini LLM (cloud — zero local RAM)
    Layer 2: Keyword fallback if Gemini fails
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
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=5),
            )
            result = response.text.strip().lower()
            if "social" in result:
                return "social"
            if "medical" in result:
                return "medical"
        except Exception as e:
            log.warning(f"IntentClassifier failed, using keyword fallback: {e}")

        q_lower = query.lower()
        return "medical" if any(kw in q_lower for kw in MEDICAL_KEYWORDS) else "social"


class KnowledgeBaseService:
    """
    Pinecone search — uses lazy singleton model.
    No embed_model stored at __init__; fetched per call via singleton.
    """

    def __init__(self, index):
        self._index = index

    def search(self, query: str, top_k: int = 5) -> List[KnowledgeMatch]:
        model  = _EmbedModelSingleton.get()
        vector = model.encode(query, show_progress_bar=False).tolist()

        results = self._index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
        )

        matches = []
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


class PromptBuilder:

    _SYSTEM = {
        "ar": (
            "أنت 'سيلا'، مساعد طبي ذكي وموثوق.\n"
            "أسلوبك: دافئ واحترافي، كأنك طبيب يشرح لمريضه باهتمام.\n\n"
            "قواعد صارمة:\n"
            "١. استخدم فقط المعلومات الطبية المقدمة في السياق أدناه.\n"
            "٢. لا تستخدم أي معرفة خارجية أو افتراضات.\n"
            "٣. إذا كانت المعلومات غير كافية، قل: 'معلوماتي محدودة في هذه الحالة، يُرجى استشارة طبيب.'\n"
            "٤. لا تُقدم تشخيصاً نهائياً أبداً.\n"
            "٥. اذكر علامات الخطر إن وُجدت.\n"
            "٦. اختم دائماً بالتوصية بمراجعة طبيب متخصص.\n\n"
            "لغة الإجابة: عربية فصحى واضحة."
        ),
        "en": (
            "You are 'Sila', a trusted medical AI assistant.\n"
            "Tone: warm and professional.\n\n"
            "Strict rules:\n"
            "1. Use ONLY the medical context provided below.\n"
            "2. Do NOT use external knowledge or assumptions.\n"
            "3. If data is insufficient say: 'My knowledge is limited. Please consult a doctor.'\n"
            "4. NEVER provide a final diagnosis.\n"
            "5. Highlight warning signs if present.\n"
            "6. Always end by recommending a specialist consultation."
        ),
    }

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
            "لا تتوفر لديّ معلومات كافية في قاعدة بياناتي لهذه الحالة. "
            "أنصحك بمراجعة طبيب متخصص للحصول على تقييم دقيق."
        ),
        "en": (
            "I don't have sufficient information in my database for this case. "
            "I recommend consulting a specialist for a proper evaluation."
        ),
    }

    def build(self, ctx: QueryContext) -> str:
        lang      = ctx.language
        system    = self._SYSTEM.get(lang, self._SYSTEM["ar"])
        structure = self._STRUCTURE.get(lang, self._STRUCTURE["ar"])

        context_parts = [
            f"[Case {i} — Confidence: {m.confidence:.0%}]\nQ: {m.question}\nA: {m.answer}"
            for i, m in enumerate(ctx.matches, start=1)
        ]

        return (
            f"{system}\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "📋 Medical Knowledge Base Context:\n\n"
            f"{chr(10).join(context_parts)}\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🧑‍⚕️ Patient Question:\n{ctx.raw_query}\n\n"
            f"{structure}\n\nAnswer:"
        )

    def no_data_response(self, language: str) -> str:
        return self._NO_DATA.get(language, self._NO_DATA["ar"])


class GeminiService:
    """All Gemini calls: social, RAG, vision. Zero local RAM."""

    def __init__(self):
        self._cache: dict[str, tuple[str, str]] = {}

    @staticmethod
    def _cfg(temperature: float = 0.2, max_tokens: int = 2048) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    def reply_social(self, query: str, language: str) -> tuple[str, str]:
        system = (
            "You are 'Sila', a friendly medical AI assistant. "
            "Reply warmly in English. If non-medical, politely mention your specialization. "
            "Keep replies short (1-2 sentences)."
        ) if language == "en" else (
            "أنت 'سيلا'، مساعد طبي ذكي وودود. "
            "ردودك بالعامية المصرية الدافية. "
            "لو الكلام مش طبي، بلطف قول إنك متخصص في الاستشارات الطبية. "
            "الرد قصير (جملة أو اتنين)."
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
                log.warning(f"Social reply: {model_name} failed — {e}")

        fallback = (
            "Hello! 😊 I'm Sila, your medical assistant. How can I help?"
            if language == "en"
            else "أهلاً! 😊 أنا سيلا، مساعدتك الطبية. إزيك؟"
        )
        return fallback, "fallback"

    def generate(self, prompt: str) -> tuple[str, str]:
        cache_key = hashlib.md5(prompt.encode()).hexdigest()
        if cache_key in self._cache:
            log.info("Cache hit.")
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
                log.warning(f"Generate: {model_name} failed — {e}")

        log.error("All text models failed.")
        return "عذراً، حدث خطأ مؤقت. يرجى المحاولة مرة أخرى.", "none"

    def analyze_image(self, image_bytes: bytes, mime_type: str) -> tuple[str, str, str]:
        system_prompt = (
            "You are a specialized medical AI. Analyze ONLY medical images such as:\n"
            "lab results, prescriptions, X-rays, MRI, CT scans, ECG, pathology reports.\n\n"
            "RULES:\n"
            "1. If NOT medical → respond ONLY with JSON:\n"
            '   {"status": "rejected", "analysis": "<explanation>"}\n\n'
            "2. If medical → respond ONLY with JSON:\n"
            '   {"status": "success", "analysis": "<full structured analysis>"}\n\n'
            "3. Include: document type, key findings, abnormal values, next steps, urgent findings.\n"
            "4. NEVER provide a final diagnosis.\n"
            "5. Respond in the image's language (AR or EN).\n"
            "6. Output ONLY valid JSON — no markdown."
        )

        b64_data = base64.b64encode(image_bytes).decode("utf-8")

        for model_name in GEMINI_VISION_MODELS:
            content_options = [
                [system_prompt, types.Part.from_bytes(data=image_bytes, mime_type=mime_type)],
                [
                    {"text": system_prompt},
                    {"inline_data": {"mime_type": mime_type, "data": b64_data}},
                ],
            ]

            for contents in content_options:
                try:
                    log.info(f"Vision: trying {model_name}")
                    response = gemini_client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=self._cfg(max_tokens=4096),
                    )
                    raw    = response.text.strip()
                    clean  = raw.replace("```json", "").replace("```", "").strip()
                    parsed = json.loads(clean)
                    status   = parsed.get("status", "success")
                    analysis = parsed.get("analysis", raw)

                    if isinstance(analysis, dict):
                        analysis = analysis.get("analysis") or json.dumps(
                            analysis, ensure_ascii=False, indent=2
                        )
                    if not isinstance(analysis, str):
                        analysis = json.dumps(analysis, ensure_ascii=False, indent=2)

                    log.info(f"Vision succeeded: {model_name}")
                    return status, analysis, model_name

                except json.JSONDecodeError:
                    log.warning(f"Vision {model_name}: non-JSON response — using raw text.")
                    return "success", raw, model_name
                except Exception as e:
                    log.warning(f"Vision {model_name} failed — {e}")
                    continue

        log.error("All vision models failed.")
        return "error", "Medical image analysis is temporarily unavailable.", "none"

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
    Lightweight startup — only creates a Pinecone connection (pure HTTP).
    SentenceTransformer is NOT loaded here.
    It loads lazily via _EmbedModelSingleton on the first /ask request.
    """
    log.info("🚀 Sila v9.0 starting — lightweight boot")
    log.info(f"🔌 Connecting to Pinecone index: {INDEX_NAME}")

    pc    = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    state.knowledge_base = KnowledgeBaseService(index)   # no model passed
    state.gemini         = GeminiService()
    state.prompt_builder = PromptBuilder()

    log.info("✅ Boot complete. Embedding model will load on first /ask request.")
    yield
    log.info("🛑 Sila shutting down.")


# ──────────────────────────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Sila — Medical AI Assistant",
    description="مساعد طبي ذكي | Strict RAG + Gemini Vision",
    version="9.0.0",
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
        "version":   "9.0.0",
        "status":    "running",
        "endpoints": ["/ask", "/analyze-image", "/health", "/docs"],
    }


@app.get("/health")
def health():
    """
    Instant health-check — never triggers model loading.
    Railway polls this right after boot; it must respond in <100ms.
    """
    return {
        "status":         "ok",
        "version":        "9.0.0",
        "model_loaded":   _EmbedModelSingleton._instance is not None,
        "embed_model":    EMBED_MODEL,
        "cache_size":     state.gemini.cache_size if state.gemini else 0,
        "min_confidence": MIN_CONFIDENCE,
        "image_analysis": "enabled",
        "index":          INDEX_NAME,
    }


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """
    Main Q&A endpoint.
    Social queries → Gemini (no local model).
    Medical queries → lazy-load embed model → Pinecone → Gemini RAG.
    """
    q        = req.query
    language = LanguageDetector.detect(q)
    intent   = IntentClassifier.classify(q)
    log.info(f"Query='{q[:80]}' lang={language} intent={intent}")

    # ── Social path (zero local RAM) ──────────────────────────────────
    if intent == "social":
        reply, model_used = state.gemini.reply_social(q, language)
        return AskResponse(
            query=q, reply=reply, model_used=model_used,
            matches=[], is_medical=False, found_in_database=False,
            low_confidence=False, language=language, disclaimer=MEDICAL_DISCLAIMER,
        )

    # ── Medical path (model loads here on first call) ─────────────────
    try:
        matches = state.knowledge_base.search(q, top_k=5)
    except Exception as e:
        log.error(f"Search failed: {e}")
        return AskResponse(
            query=q,
            reply=(
                "عذراً، حدث خطأ في البحث. يرجى المحاولة مرة أخرى."
                if language == "ar"
                else "Sorry, search failed. Please try again."
            ),
            model_used="none", matches=[], is_medical=True,
            found_in_database=False, low_confidence=True,
            language=language, disclaimer=MEDICAL_DISCLAIMER,
        )

    ctx = QueryContext(raw_query=q, language=language, matches=matches)

    if not ctx.has_reliable_matches:
        reply = state.prompt_builder.no_data_response(language)
        log.info(f"No reliable matches. Best={matches[0].confidence if matches else 0:.2f}")
        return AskResponse(
            query=q, reply=reply, model_used="none",
            matches=[MatchResult(**m.__dict__) for m in matches],
            is_medical=True, found_in_database=False,
            low_confidence=True, language=language, disclaimer=MEDICAL_DISCLAIMER,
        )

    prompt = state.prompt_builder.build(ctx)
    reply, model_used = state.gemini.generate(prompt)
    log.info(f"RAG via {model_used} | top confidence={matches[0].confidence:.2f}")

    return AskResponse(
        query=q, reply=reply, model_used=model_used,
        matches=[MatchResult(**m.__dict__) for m in matches],
        is_medical=True, found_in_database=True,
        low_confidence=False, language=language, disclaimer=MEDICAL_DISCLAIMER,
    )


@app.post("/analyze-image", response_model=ImageAnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    """
    Medical image analysis via Gemini Vision.
    No local model — pure cloud inference.
    """
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        return JSONResponse(
            status_code=400,
            content={
                "status":     "error",
                "analysis":   f"نوع الملف '{file.content_type}' غير مدعوم. المسموح به: JPEG, PNG, WEBP, HEIC.",
                "model_used": "none",
                "disclaimer": MEDICAL_DISCLAIMER,
            },
        )

    try:
        image_bytes = await file.read()
    except Exception as e:
        log.error(f"Failed to read '{file.filename}': {e}")
        return JSONResponse(
            status_code=400,
            content={
                "status":     "error",
                "analysis":   "فشل في قراءة الملف. تأكد من أن الصورة غير تالفة.",
                "model_used": "none",
                "disclaimer": MEDICAL_DISCLAIMER,
            },
        )

    if len(image_bytes) > MAX_IMAGE_BYTES:
        return JSONResponse(
            status_code=413,
            content={
                "status":     "error",
                "analysis":   f"حجم الصورة كبير جداً. الحد الأقصى {MAX_IMAGE_MB}MB.",
                "model_used": "none",
                "disclaimer": MEDICAL_DISCLAIMER,
            },
        )

    log.info(f"Image: '{file.filename}' {len(image_bytes)/1024:.1f}KB {file.content_type}")
    status, analysis, model_used = state.gemini.analyze_image(image_bytes, file.content_type)

    return JSONResponse(
        status_code=503 if status == "error" else 200,
        content={
            "status":     status,
            "analysis":   analysis,
            "model_used": model_used,
            "disclaimer": MEDICAL_DISCLAIMER,
        },
    )