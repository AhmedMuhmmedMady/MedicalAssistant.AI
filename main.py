"""
Medical AI Assistant — Strict RAG Edition v7.0
===============================================
- Migrated from google-generativeai → google-genai (new SDK)
- Answers ONLY from Pinecone knowledge base (text queries)
- Analyzes medical images via Gemini Vision (/analyze-image)
- If no relevant match is found, says so clearly
"""

import base64
import hashlib
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import List, Optional

from google import genai
from google.genai import types
from fastapi import FastAPI, File, Request, UploadFile
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
MAX_IMAGE_SIZE   = int(os.getenv("MAX_IMAGE_SIZE_MB", "10")) * 1024 * 1024

GEMINI_MODELS = [
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

if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY environment variable.")
if not PINECONE_API_KEY:
    raise RuntimeError("Missing PINECONE_API_KEY environment variable.")

# ── New SDK: single client instance ────────────────────────────────────────
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("medical_ai")


# ─────────────────────────────────────────────
# Constants & Prompts
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

VISION_SYSTEM_PROMPT = """You are a specialized medical AI assistant trained to analyze medical documents and images.

STRICT VALIDATION — apply BEFORE any analysis:
1. You ONLY analyze medical-related images such as:
   - Laboratory test results (blood work, urine analysis, cultures, lipid panels, CBC, etc.)
   - Medical prescriptions and medication lists
   - Radiology images (X-rays, MRI, CT scans, ultrasounds)
   - Pathology reports and microscopy slides
   - Medical charts, ECG/EKG readings, vital sign charts
   - Clinical notes and discharge summaries

2. If the image is NOT medical, respond ONLY with this exact JSON:
   {"status": "rejected", "analysis": "This image does not appear to be a medical document. I can only analyze lab results, prescriptions, X-rays, and other medical records."}

3. If the image IS medical, respond ONLY with this exact JSON:
   {"status": "success", "analysis": "<your full structured analysis here>"}

4. For medical images, your analysis must include:
   - Document type identified
   - Key findings or values observed
   - Values outside normal range (if any), clearly highlighted
   - Recommended next steps or specialist to consult
   - Any urgent findings that require immediate attention

5. NEVER provide a final diagnosis — provide observations and recommend consulting a specialist.
6. Respond in the SAME language as the text found in the image (Arabic or English).
7. Output ONLY valid JSON — no markdown, no extra text outside the JSON object."""


INTENT_CLASSIFICATION_PROMPT = """Classify the following message into one of two categories:
- "social"  → greetings, thanks, casual conversation, non-medical small talk
- "medical" → any question about symptoms, diseases, medications, body parts, pain, health

Respond with EXACTLY one word: social OR medical. Nothing else."""


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
# Pydantic Schemas
# ─────────────────────────────────────────────

class AskRequest(BaseModel):
    # Accept both "text" (internal) and "question" (C# client) field names
    text: Optional[str] = None
    question: Optional[str] = None

    @property
    def query(self) -> str:
        """Unified accessor regardless of which field was sent."""
        return (self.text or self.question or "").strip()

    def model_post_init(self, __context) -> None:
        if not self.query:
            raise ValueError("Request must include a non-empty 'text' or 'question' field.")
        if len(self.query) > MAX_QUERY_LENGTH:
            raise ValueError(f"Query exceeds maximum length of {MAX_QUERY_LENGTH} characters.")


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


class IntentClassifier:
    """
    Uses Gemini to classify whether a query is social or medical.
    Falls back to keyword-based classification if Gemini is unavailable.
    """
    @staticmethod
    def classify(query: str) -> str:
        """Returns 'social' or 'medical'."""
        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.0-flash-lite",   # cheapest & fastest for classification
                contents=f"{INTENT_CLASSIFICATION_PROMPT}\n\nMessage: {query}",
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=5,
                ),
            )
            result = response.text.strip().lower()
            return "social" if "social" in result else "medical"
        except Exception as e:
            log.warning(f"Intent classification failed, defaulting to keyword check: {e}")
            return "unknown"


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
    _SYSTEM_AR = (
        "أنت 'سيلا'، مساعد طبي ذكي ومتخصص يعمل كمرجع طبي موثوق.\n"
        "أسلوبك: دافئ واحترافي، كأنك طبيب يشرح لمريضه بوضوح واهتمام.\n"
        "القواعد الصارمة:\n"
        "1. أجب فقط بناءً على الحالات الطبية المقدمة في السياق أدناه — لا تتجاوزها.\n"
        "2. لا تستخدم أي معرفة خارجية أو افتراضات من تلقاء نفسك.\n"
        "3. إذا كانت المعلومات المتاحة غير كافية، قل بوضوح: 'معلوماتي محدودة في هذه الحالة'.\n"
        "4. لا تضع تشخيصاً نهائياً — قدم احتمالات وتوجيهاً مبنياً على البيانات المتاحة.\n"
        "5. اذكر علامات الخطر التي تستدعي التوجه للطوارئ فوراً إن وُجدت في السياق.\n"
        "6. اختم دائماً بتوصية بمراجعة الطبيب المختص."
    )

    _SYSTEM_EN = (
        "You are 'Sila', a specialized medical AI assistant serving as a reliable medical reference.\n"
        "Your tone: warm and professional, like a doctor explaining clearly to their patient.\n"
        "Strict rules:\n"
        "1. Answer ONLY based on the medical cases provided in the context below — do not go beyond it.\n"
        "2. Do NOT use external knowledge or personal assumptions.\n"
        "3. If available information is insufficient, clearly state: 'My knowledge is limited on this case.'\n"
        "4. Do NOT give a final diagnosis — provide possible explanations based on available data.\n"
        "5. Mention emergency warning signs if the context suggests any.\n"
        "6. Always end with a recommendation to consult the appropriate specialist."
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
        system    = self._SYSTEM_AR if ctx.language == "ar" else self._SYSTEM_EN
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
    """
    Wrapper around the new google-genai SDK.
    Uses gemini_client (module-level) for all calls.
    """

    def __init__(self):
        self._cache: dict[str, tuple[str, str]] = {}

    def _config(self, max_tokens: int = 2048) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=max_tokens,
        )

    # ── Social reply (Egyptian Arabic) ────────────────────────────────
    def reply_social(self, query: str) -> tuple[str, str]:
        """
        Generates a warm, friendly reply in Egyptian Arabic for non-medical queries.
        Returns (reply_text, model_used).
        """
        system = (
            "أنت مساعد طبي ذكي واسمك 'سيلا'. "
            "ردودك دايماً بالعامية المصرية الدافية والودية، زي طبيب صاحبك. "
            "لو حد بيسلم عليك أو بيشكرك أو بيتكلم معاك بشكل عام، رد عليه بطبيعية ودفا. "
            "لو حد سألك عن حاجة مش طبية، بلطف وده قوله إنك متخصص في الاستشارات الطبية. "
            "الرد يكون قصير (جملة أو اتنين بالكتير)."
        )
        last_error = None
        for model_name in GEMINI_MODELS:
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
                log.warning(f"Social reply model {model_name} failed: {e}")
                last_error = e
        log.error(f"All models failed for social reply: {last_error}")
        return "أهلاً وسهلاً! 😊 أنا هنا لمساعدتك في أي استفسار طبي.", "none"

    # ── Text generation ────────────────────────────────────────────────
    def generate(self, prompt: str) -> tuple[str, str]:
        cache_key = hashlib.md5(prompt.encode()).hexdigest()
        if cache_key in self._cache:
            log.info("Cache hit for prompt.")
            return self._cache[cache_key]

        last_error = None
        for model_name in GEMINI_MODELS:
            try:
                log.info(f"Calling model: {model_name}")
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=self._config(),
                )
                text = response.text.strip()
                self._cache[cache_key] = (text, model_name)
                return text, model_name
            except Exception as e:
                log.warning(f"Model {model_name} failed: {e}")
                last_error = e

        log.error(f"All Gemini models failed. Last error: {last_error}")
        return "عذراً، حدث خطأ مؤقت في الخدمة. يرجى المحاولة مرة أخرى.", "none"

    # ── Vision / image analysis ────────────────────────────────────────
    def analyze_image(self, image_bytes: bytes, mime_type: str) -> tuple[str, str, str]:
        """
        Analyze a medical image using Gemini Vision.
        Returns (status, analysis, model_used).
        """
        last_error = None

        for model_name in GEMINI_VISION_MODELS:
            try:
                log.info(f"Trying vision model: {model_name}")
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=[
                        VISION_SYSTEM_PROMPT,
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    ],
                    config=self._config(max_tokens=8192),  # larger for detailed image analysis
                )
                raw_text = response.text.strip()

                # Parse JSON response from Gemini — handle nested JSON
                try:
                    clean  = raw_text.replace("```json", "").replace("```", "").strip()
                    parsed = json.loads(clean)
                    status   = parsed.get("status", "success")
                    analysis = parsed.get("analysis", raw_text)

                    # Gemini sometimes nests another JSON object inside analysis
                    # Unwrap it recursively until we get a plain string
                    max_depth = 3
                    depth = 0
                    while isinstance(analysis, (dict, list)) and depth < max_depth:
                        if isinstance(analysis, dict):
                            inner = analysis.get("analysis")
                            if inner is not None:
                                analysis = inner
                                depth += 1
                            else:
                                # Convert dict to readable string
                                analysis = json.dumps(analysis, ensure_ascii=False, indent=2)
                                break
                        else:
                            analysis = json.dumps(analysis, ensure_ascii=False, indent=2)
                            break

                    # Final safety: if still not a string, serialize it
                    if not isinstance(analysis, str):
                        analysis = json.dumps(analysis, ensure_ascii=False, indent=2)

                    return status, analysis, model_name

                except (json.JSONDecodeError, KeyError):
                    log.warning(f"Vision model {model_name} did not return valid JSON — using raw text.")
                    return "success", raw_text, model_name

            except Exception as e:
                log.warning(f"Vision model {model_name} failed: {e}")
                last_error = e

        log.error(f"All vision models failed. Last error: {last_error}")
        return "error", "Medical image analysis service is temporarily unavailable. Please try again later.", "none"

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
    pc    = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    state.knowledge_base = KnowledgeBaseService(index, embed_model)
    state.gemini         = GeminiService()
    state.prompt_builder = PromptBuilder()

    log.info("✅ Medical AI Assistant v7.0 is ready.")
    yield
    log.info("🛑 Shutdown complete.")


# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────

app = FastAPI(
    title="Medical AI Assistant",
    description="مساعد طبي ذكي — Strict RAG + Gemini Vision",
    version="7.0.0",
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
    q        = req.query
    language = LanguageDetector.detect(q)

    # ── Step 1: Intent Classification ──────────────────────────────────
    intent = IntentClassifier.classify(q)

    if intent == "social":
        # Warm Egyptian Arabic reply directly from Gemini — no RAG needed
        reply, model_used = state.gemini.reply_social(q)
        return AskResponse(
            query=q, gemini_reply=reply, model_used=model_used,
            matches=[], low_confidence=False, is_medical=False,
            found_in_database=False, disclaimer=MEDICAL_DISCLAIMER, language=language,
        )

    # ── Step 2: Medical path — keyword fallback if intent == "unknown" ──
    matches    = state.knowledge_base.search(q, top_k=5)
    is_medical = (
        intent == "medical"
        or MedicalClassifier.is_medical(q, matches)
    )

    if not is_medical:
        reply, model_used = state.gemini.reply_social(q)
        return AskResponse(
            query=q, gemini_reply=reply, model_used=model_used,
            matches=[], low_confidence=True, is_medical=False,
            found_in_database=False, disclaimer=MEDICAL_DISCLAIMER, language=language,
        )

    ctx = QueryContext(
        raw_query=q, language=language,
        is_medical=True, matches=matches,
    )

    # ── Step 3: No reliable matches → honest response ───────────────────
    if not ctx.has_reliable_matches:
        reply = state.prompt_builder.get_no_data_response(language)
        return AskResponse(
            query=q, gemini_reply=reply, model_used="none",
            matches=[MatchResult(**m.__dict__) for m in matches],
            low_confidence=True, is_medical=True, found_in_database=False,
            disclaimer=MEDICAL_DISCLAIMER, language=language,
        )

    # ── Step 4: Reliable matches → Strict RAG answer ────────────────────
    prompt = state.prompt_builder.build(ctx)
    reply, model_used = state.gemini.generate(prompt)

    return AskResponse(
        query=q, gemini_reply=reply, model_used=model_used,
        matches=[MatchResult(**m.__dict__) for m in matches],
        low_confidence=False, is_medical=True, found_in_database=True,
        disclaimer=MEDICAL_DISCLAIMER, language=language,
    )


@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze a medical image (lab report, prescription, X-ray, etc.)
    using Gemini Vision. Returns structured analysis or a polite rejection
    if the image is not medical-related.
    """
    # ── Validate content type ───────────────────────────────────────────
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        return JSONResponse(
            status_code=400,
            content={
                "status":     "error",
                "analysis":   f"Unsupported file type '{file.content_type}'. "
                              "Allowed: JPEG, PNG, WEBP, HEIC.",
                "model_used": "none",
                "disclaimer": MEDICAL_DISCLAIMER,
            },
        )

    # ── Read bytes ──────────────────────────────────────────────────────
    try:
        image_bytes = await file.read()
    except Exception as e:
        log.error(f"Failed to read uploaded image '{file.filename}': {e}")
        return JSONResponse(
            status_code=400,
            content={
                "status":     "error",
                "analysis":   "Failed to read the uploaded file.",
                "model_used": "none",
                "disclaimer": MEDICAL_DISCLAIMER,
            },
        )

    # ── Validate size ───────────────────────────────────────────────────
    if len(image_bytes) > MAX_IMAGE_SIZE:
        return JSONResponse(
            status_code=400,
            content={
                "status":     "error",
                "analysis":   f"File too large. Maximum allowed size is "
                              f"{MAX_IMAGE_SIZE // (1024 * 1024)}MB.",
                "model_used": "none",
                "disclaimer": MEDICAL_DISCLAIMER,
            },
        )

    log.info(f"Analyzing image: {file.filename} ({len(image_bytes) / 1024:.1f} KB, {file.content_type})")

    # ── Analyze via Gemini Vision ───────────────────────────────────────
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


# ─────────────────────────────────────────────
# Utility Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":                   "ok",
        "version":                  "7.0.0",
        "cache_size":               state.gemini.cache_size if state.gemini else 0,
        "min_confidence_threshold": MIN_CONFIDENCE,
        "image_analysis":           "enabled",
    }


@app.get("/")
def root():
    return {
        "name":      "Medical AI Assistant",
        "version":   "7.0.0",
        "mode":      "strict-rag + vision",
        "status":    "running",
        "endpoints": ["/ask", "/analyze-image", "/health"],
        "docs":      "/docs",
    }