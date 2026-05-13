"""
Medical AI Assistant — Strict RAG Edition v7.0
===============================================
Changes from v6:
- MedicalClassifier now handles social greetings & chitchat naturally
  (السلام عليكم، ازيك، شكراً، etc.) with warm, human-like responses
  before steering back to the medical context.
- Greeting responses are context-aware (AR/EN/EG dialect).
- All other logic remains strictly RAG-based.
"""

import base64
import hashlib
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

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
    "جرح", "كدمة", "خدر", "تنميل", "تورم", "حساسية", "زكام",
    "برد", "انفلونزا", "بكتيريا", "فيروس", "فحص", "تحليل", "اشعة",
}

MEDICAL_KEYWORDS_EN = {
    "pain", "ache", "fever", "cough", "headache", "nausea", "dizzy",
    "vomit", "diarrhea", "symptom", "disease", "doctor", "hospital",
    "medicine", "drug", "blood", "heart", "lung", "kidney", "liver",
    "diabetes", "pressure", "infection", "allergy", "rash", "swelling",
    "fatigue", "tired", "breathe", "chest", "stomach", "throat",
    "prescription", "diagnosis", "treatment", "surgery", "lab", "scan",
    "xray", "mri", "test", "result", "clinic", "pharmacy", "dose",
}

VISION_SYSTEM_PROMPT = """You are a specialized medical AI assistant trained to analyze medical documents and images.

STRICT VALIDATION — apply BEFORE any analysis:

1. You ONLY analyze medical-related images such as:
   - Laboratory test results (CBC, Hematology, Urine Analysis, Lipid Profile, etc.)
   - Medical prescriptions and medication lists
   - Radiology images (X-rays, MRI, CT scans, ultrasounds)
   - Pathology reports and microscopy slides
   - ECG/EKG readings
   - Clinical notes and discharge summaries

2. If the image is NOT medical, respond ONLY with this exact JSON:
{
  "status": "rejected",
  "analysis_ar": "هذه الصورة لا تبدو مستندًا طبيًا. يمكنني فقط تحليل التحاليل الطبية، الأشعات، والوصفات العلاجية.",
  "technical_details": "Non-medical image detected."
}

3. If the image IS medical, respond ONLY with valid JSON using this EXACT structure:
{
  "status": "success",
  "analysis_ar": "<Arabic explanation for the patient>",
  "technical_details": "<Technical medical details in English>"
}

4. Rules for medical analysis:
   - Explain the findings in SIMPLE Arabic for normal patients
   - Keep medical terminology and abbreviations in English
   - Clearly highlight abnormal values
   - Mention possible concerns without giving a final diagnosis
   - Mention recommended specialist or next medical step
   - Mention urgent findings if they exist

5. NEVER provide a confirmed diagnosis.

6. The Arabic explanation MUST:
   - Be medically accurate
   - Be easy to understand
   - Sound natural for Arabic-speaking users

7. The English technical section should:
   - Be concise and professional
   - Include abnormal findings and observations

8. Output ONLY valid JSON. Do NOT use markdown.
"""


# ─────────────────────────────────────────────
# Domain Enums & Models
# ─────────────────────────────────────────────

class QueryIntent(Enum):
    """High-level classification of what the user is trying to do."""
    GREETING      = auto()   # سلام / hello / ازيك
    GRATITUDE     = auto()   # شكراً / thanks
    FAREWELL      = auto()   # مع السلامة / bye
    AFFIRMATION   = auto()   # تمام / ok / نعم
    MEDICAL       = auto()   # actual medical question
    OFF_TOPIC     = auto()   # something unrelated and not social


@dataclass
class KnowledgeMatch:
    symptom:    str
    reply:      str
    category:   str
    confidence: float

    @property
    def is_reliable(self) -> bool:
        return self.confidence >= MIN_CONFIDENCE


@dataclass
class QueryContext:
    raw_query:  str
    language:   str
    intent:     QueryIntent
    is_medical: bool
    matches:    List[KnowledgeMatch] = field(default_factory=list)

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
    text: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty.")
        if len(v) > MAX_QUERY_LENGTH:
            raise ValueError(
                f"Query exceeds maximum length of {MAX_QUERY_LENGTH} characters."
            )
        return v


class MatchResult(BaseModel):
    symptom:    str
    reply:      str
    category:   str
    confidence: float


class AskResponse(BaseModel):
    query:            str
    gemini_reply:     str
    model_used:       str
    matches:          List[MatchResult]
    low_confidence:   bool
    is_medical:       bool
    found_in_database: bool
    disclaimer:       str
    language:         str
    intent:           str


# ─────────────────────────────────────────────
# Language Detector
# ─────────────────────────────────────────────

class LanguageDetector:
    """
    Detects Arabic vs English.
    Also attempts to detect Egyptian dialect (for friendlier responses).
    """

    _EG_MARKERS = {
        "ازيك", "ازيكم", "عامل", "عاملة", "ايه", "إيه",
        "بتوجعني", "بيوجعني", "بتوجعنى", "بيوجعنى",
        "مش", "كده", "كدا", "عشان", "بتاع", "زى", "زي",
        "تمام", "ماشي", "ماشى", "يعني", "يعنى", "أيوه", "ايوه",
        "فين", "هنا", "هناك", "مين", "ليه", "إمتى", "امتى",
    }

    @staticmethod
    def detect(text: str) -> str:
        arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
        ratio = arabic_chars / max(len(text), 1)
        return "ar" if ratio > 0.3 else "en"

    @classmethod
    def is_egyptian_dialect(cls, text: str) -> bool:
        lower = text.lower()
        return any(marker in lower for marker in cls._EG_MARKERS)


# ─────────────────────────────────────────────
# Intent Classifier (rule-based, zero-latency)
# ─────────────────────────────────────────────

class IntentClassifier:
    """
    Fast rule-based classifier that intercepts social/conversational
    messages BEFORE they reach the vector DB or Gemini.

    Priority order:
      1. Greetings
      2. Gratitude
      3. Farewells
      4. Affirmations / filler phrases
      5. Medical (keyword or KB hit)
      6. Off-topic
    """

    # ── Greeting patterns ────────────────────────────────────────────
    _GREETINGS_AR = {
        r"السلام\s*عليكم", r"وعليكم\s*السلام", r"مرحب[اً]?", r"أهلاً?",
        r"اهلاً?", r"هلا", r"صباح\s*(الخير|النور|الفل)",
        r"مساء\s*(الخير|النور)", r"ازيك", r"إزيك", r"ازيكم", r"إزيكم",
        r"عامل\s*ايه", r"عامل\s*إيه", r"كيف\s*حالك", r"كيفك",
        r"كيف\s*الحال", r"شو\s*(أخبارك|أخبارج|عمل)",
        r"هاي", r"هاى", r"هلو", r"حياك\s*الله",
    }

    _GREETINGS_EN = {
        r"\bhello\b", r"\bhi\b", r"\bhey\b", r"\bgreetings\b",
        r"\bgood\s*(morning|afternoon|evening|day)\b",
        r"\bhowdy\b", r"\bwassup\b", r"\bwhat'?s\s*up\b",
    }

    # ── Gratitude patterns ───────────────────────────────────────────
    _THANKS_AR = {
        r"شكر[اً]?", r"متشكر", r"ممنون", r"مشكور", r"يسلموا?",
        r"الله\s*يسلمك", r"جزاك\s*الله", r"بارك\s*الله",
    }

    _THANKS_EN = {
        r"\bthanks?\b", r"\bthank\s*you\b", r"\bthx\b",
        r"\bappreciat\w+\b", r"\bgrateful\b",
    }

    # ── Farewell patterns ────────────────────────────────────────────
    _FAREWELL_AR = {
        r"مع\s*السلامة", r"وداعاً?", r"إلى\s*اللقاء",
        r"باي", r"بائ", r"سلام\s*$", r"يلا\s*(باي|سلام)",
    }

    _FAREWELL_EN = {
        r"\bbye\b", r"\bgoodbye\b", r"\bsee\s*ya\b",
        r"\bsee\s*you\b", r"\btake\s*care\b", r"\blater\b",
    }

    # ── Affirmation / filler ─────────────────────────────────────────
    _AFFIRMATION_AR = {
        r"تمام", r"ماشي", r"ماشى", r"أوكي", r"اوكيه?",
        r"نعم", r"أيوه?", r"ايوه?", r"طيب", r"حسناً?",
        r"صح", r"زين", r"إن\s*شاء\s*الله",
    }

    _AFFIRMATION_EN = {
        r"\bok(ay)?\b", r"\bsure\b", r"\byes\b", r"\byep\b",
        r"\byup\b", r"\balright\b", r"\bgot\s*it\b",
    }

    # Pre-compile all patterns for speed
    _PATTERNS: dict[QueryIntent, list] = {}

    @classmethod
    def _compile(cls) -> None:
        if cls._PATTERNS:
            return
        mapping = {
            QueryIntent.GREETING:    cls._GREETINGS_AR | cls._GREETINGS_EN,
            QueryIntent.GRATITUDE:   cls._THANKS_AR    | cls._THANKS_EN,
            QueryIntent.FAREWELL:    cls._FAREWELL_AR  | cls._FAREWELL_EN,
            QueryIntent.AFFIRMATION: cls._AFFIRMATION_AR | cls._AFFIRMATION_EN,
        }
        for intent, patterns in mapping.items():
            cls._PATTERNS[intent] = [
                re.compile(p, re.IGNORECASE | re.UNICODE)
                for p in patterns
            ]

    @classmethod
    def classify(
        cls,
        query: str,
        kb_matches: List[KnowledgeMatch],
    ) -> QueryIntent:
        cls._compile()

        # Short-circuit: if KB returned a reliable medical hit → MEDICAL
        if kb_matches and kb_matches[0].confidence >= MIN_CONFIDENCE:
            return QueryIntent.MEDICAL

        for intent, compiled in cls._PATTERNS.items():
            for pattern in compiled:
                if pattern.search(query):
                    return intent

        # Keyword scan for medical content
        q_lower = query.lower()
        if any(kw in q_lower for kw in MEDICAL_KEYWORDS_AR | MEDICAL_KEYWORDS_EN):
            return QueryIntent.MEDICAL

        return QueryIntent.OFF_TOPIC


# ─────────────────────────────────────────────
# Social Response Generator
# ─────────────────────────────────────────────

class SocialResponseGenerator:
    """
    Produces warm, human-like responses for social/conversational inputs.
    Always ends with a gentle invitation to ask a medical question.
    """

    # Each category has a pool of responses; one is chosen deterministically
    # based on the query hash (so the same phrase doesn't always get the
    # same answer, but it IS reproducible for caching).

    _RESPONSES: dict[QueryIntent, dict[str, list[str]]] = {
        QueryIntent.GREETING: {
            "ar": [
                "وعليكم السلام ورحمة الله وبركاته 😊 أهلاً وسهلاً! أنا مساعدك الطبي. لو عندك أي استفسار صحي أو تحس بأي أعراض — أنا هنا ليك.",
                "أهلاً بيك! 👋 يسعدني أساعدك. لو عندك سؤال طبي أو بتحس بأي حاجة — قولي وأنا في الخدمة.",
                "مرحباً! 🌿 أنا مساعدك الصحي. هل تريد الاستفسار عن أعراض معينة أو حالة طبية؟",
                "وعليكم السلام! أهلاً بك. إذا كان لديك استفسار طبي أو صحي — تفضل بسؤالك وسأحاول مساعدتك.",
            ],
            "en": [
                "Hello! 👋 Welcome! I'm your medical assistant. Feel free to ask about any health concern or symptoms you have.",
                "Hi there! 😊 I'm here to help with any medical questions you might have. What's on your mind?",
                "Hey! Good to have you here. I'm a medical AI assistant — ask me anything health-related.",
            ],
        },
        QueryIntent.GRATITUDE: {
            "ar": [
                "العفو! 😊 ده واجبي. لو عندك أي سؤال تاني أو حاجة تانية — أنا هنا.",
                "بكل سرور! شرفني أساعدك. لو في أي حاجة تانية تخص صحتك — اسأل بكل راحة.",
                "لا شكر على واجب! 🙏 صحتك تهمنا. في أي وقت محتاج مساعدة طبية — أنا هنا.",
                "العفو تماماً. إذا كان لديك أي استفسار آخر — لا تتردد.",
            ],
            "en": [
                "You're welcome! 😊 Happy to help. Feel free to ask anything else.",
                "Of course! That's what I'm here for. Any other health questions?",
                "No problem at all! Let me know if there's anything else I can help with.",
            ],
        },
        QueryIntent.FAREWELL: {
            "ar": [
                "مع السلامة! 👋 اعتني بنفسك. لو احتجت مساعدة طبية في أي وقت — أنا هنا.",
                "إلى اللقاء! 🌿 ربنا يحفظك ويوفقك. لو عندك أي سؤال مستقبلاً — لا تتردد.",
                "وداعاً! تمنياتي لك بالصحة والعافية. 💙",
                "مع السلامة! في أمان الله. أي وقت محتاج فيه استشارة طبية — رجعلنا.",
            ],
            "en": [
                "Take care! 👋 Feel free to come back anytime you have health questions.",
                "Goodbye! Wishing you good health. Don't hesitate to reach out anytime.",
                "See you! Stay healthy. 💙",
            ],
        },
        QueryIntent.AFFIRMATION: {
            "ar": [
                "تمام! 😊 لو عندك أي سؤال طبي — أنا في الخدمة.",
                "حسناً! أنا هنا متى احتجت. هل عندك استفسار صحي؟",
                "أوكي! لو في حاجة تخص صحتك — قولي وأنا أساعدك.",
            ],
            "en": [
                "Got it! 😊 Let me know if you have any medical questions.",
                "Sure! I'm here whenever you need health advice.",
                "Alright! Feel free to ask anything health-related.",
            ],
        },
    }

    @classmethod
    def generate(
        cls,
        intent:    QueryIntent,
        query:     str,
        language:  str,
        is_egyptian: bool = False,
    ) -> str:
        pool = cls._RESPONSES.get(intent, {})
        lang_key = "ar" if language == "ar" else "en"
        options = pool.get(lang_key, [])

        if not options:
            # Fallback
            if language == "ar":
                return "أهلاً! كيف يمكنني مساعدتك طبياً اليوم؟"
            return "Hello! How can I assist you medically today?"

        # Deterministic but varied selection based on query content
        idx = hash(query.strip().lower()) % len(options)
        return options[idx]


# ─────────────────────────────────────────────
# Medical Classifier (updated façade)
# ─────────────────────────────────────────────

class MedicalClassifier:
    """
    Combines IntentClassifier + keyword scan.
    Returns (is_medical: bool, intent: QueryIntent).
    """

    @staticmethod
    def classify(
        query: str,
        matches: List[KnowledgeMatch],
    ) -> Tuple[bool, QueryIntent]:
        intent = IntentClassifier.classify(query, matches)
        is_medical = intent == QueryIntent.MEDICAL
        return is_medical, intent


# ─────────────────────────────────────────────
# Knowledge Base Service
# ─────────────────────────────────────────────

class KnowledgeBaseService:
    def __init__(self, index, embed_model: SentenceTransformer):
        self._index       = index
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


# ─────────────────────────────────────────────
# Prompt Builder
# ─────────────────────────────────────────────

class PromptBuilder:
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
        return (
            self._NO_DATA_RESPONSE_AR
            if language == "ar"
            else self._NO_DATA_RESPONSE_EN
        )

    def build(self, ctx: QueryContext) -> str:
        system    = self._SYSTEM_AR if ctx.language == "ar" else self._SYSTEM_EN
        lang_note = (
            "أجب باللغة العربية فقط."
            if ctx.language == "ar"
            else "Answer in English only."
        )

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


# ─────────────────────────────────────────────
# Gemini Service
# ─────────────────────────────────────────────

class GeminiService:
    """Wrapper around the google-genai SDK with caching and fallback."""

    def __init__(self):
        self._cache: dict[str, tuple[str, str]] = {}

    def _config(self, max_tokens: int = 2048) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=max_tokens,
        )

    # ── Text generation ────────────────────────────────────────────────
    def generate(self, prompt: str) -> Tuple[str, str]:
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
        return (
            "عذراً، حدث خطأ مؤقت في الخدمة. يرجى المحاولة مرة أخرى.",
            "none",
        )

    # ── Vision / image analysis ────────────────────────────────────────
    def analyze_image(
        self,
        image_bytes: bytes,
        mime_type:   str,
    ) -> Tuple[str, str, str, str]:
        """
        Returns (status, analysis_ar, technical_details, model_used).
        """
        last_error = None

        for model_name in GEMINI_VISION_MODELS:
            try:
                log.info(f"Trying vision model: {model_name}")
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=[
                        VISION_SYSTEM_PROMPT,
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type=mime_type,
                        ),
                    ],
                    config=self._config(max_tokens=8192),
                )
                raw_text = response.text.strip()

                try:
                    clean  = raw_text.replace("```json", "").replace("```", "").strip()
                    parsed = json.loads(clean)

                    status           = parsed.get("status", "success")
                    analysis_ar      = parsed.get(
                        "analysis_ar",
                        "لم يتمكن النظام من إنشاء تحليل عربي.",
                    )
                    technical_details = parsed.get(
                        "technical_details",
                        "No technical details available.",
                    )

                    # Unwrap nested analysis if Gemini returned a dict
                    max_depth, depth = 3, 0
                    while isinstance(analysis_ar, (dict, list)) and depth < max_depth:
                        if isinstance(analysis_ar, dict):
                            inner = analysis_ar.get("analysis_ar")
                            if inner is not None:
                                analysis_ar = inner
                                depth += 1
                            else:
                                analysis_ar = json.dumps(
                                    analysis_ar, ensure_ascii=False, indent=2
                                )
                                break
                        else:
                            analysis_ar = json.dumps(
                                analysis_ar, ensure_ascii=False, indent=2
                            )
                            break

                    if not isinstance(analysis_ar, str):
                        analysis_ar = json.dumps(
                            analysis_ar, ensure_ascii=False, indent=2
                        )

                    return status, analysis_ar, technical_details, model_name

                except (json.JSONDecodeError, KeyError) as json_err:
                    log.warning(
                        f"Invalid JSON from vision model {model_name}: {json_err}"
                    )
                    return "error", "حدث خطأ أثناء تحليل الصورة الطبية.", raw_text, model_name

            except Exception as e:
                log.warning(f"Vision model {model_name} failed: {e}")
                last_error = e

        log.error(f"All vision models failed. Last error: {last_error}")
        return (
            "error",
            "خدمة تحليل الصور الطبية غير متاحة حالياً.",
            "Vision service unavailable.",
            "none",
        )

    @property
    def cache_size(self) -> int:
        return len(self._cache)


# ─────────────────────────────────────────────
# Application State
# ─────────────────────────────────────────────

class AppState:
    knowledge_base:  Optional[KnowledgeBaseService] = None
    gemini:          Optional[GeminiService]        = None
    prompt_builder:  Optional[PromptBuilder]        = None


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
    description="مساعد طبي ذكي — Strict RAG + Gemini Vision + Social Awareness",
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
# /ask Endpoint
# ─────────────────────────────────────────────

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    language    = LanguageDetector.detect(req.text)
    is_egyptian = LanguageDetector.is_egyptian_dialect(req.text)

    # 1. Search knowledge base (needed for intent classification)
    matches = state.knowledge_base.search(req.text, top_k=5)

    # 2. Classify intent
    is_medical, intent = MedicalClassifier.classify(req.text, matches)

    log.info(
        f"Query='{req.text[:60]}' | lang={language} | "
        f"egyptian={is_egyptian} | intent={intent.name}"
    )

    # ── Social / conversational intent ─────────────────────────────────
    if intent in {
        QueryIntent.GREETING,
        QueryIntent.GRATITUDE,
        QueryIntent.FAREWELL,
        QueryIntent.AFFIRMATION,
    }:
        reply = SocialResponseGenerator.generate(
            intent=intent,
            query=req.text,
            language=language,
            is_egyptian=is_egyptian,
        )
        return AskResponse(
            query=req.text,
            gemini_reply=reply,
            model_used="rule-based",
            matches=[],
            low_confidence=False,
            is_medical=False,
            found_in_database=False,
            disclaimer=MEDICAL_DISCLAIMER,
            language=language,
            intent=intent.name,
        )

    # ── Off-topic (non-medical, non-social) ────────────────────────────
    if intent == QueryIntent.OFF_TOPIC:
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
            intent=intent.name,
        )

    # ── Medical intent ─────────────────────────────────────────────────
    ctx = QueryContext(
        raw_query=req.text,
        language=language,
        intent=intent,
        is_medical=True,
        matches=matches,
    )

    # No reliable KB matches → honest fallback
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
            intent=intent.name,
        )

    # Reliable matches → generate Gemini answer
    prompt             = state.prompt_builder.build(ctx)
    reply, model_used  = state.gemini.generate(prompt)

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
        intent=intent.name,
    )


# ─────────────────────────────────────────────
# /analyze-image Endpoint
# ─────────────────────────────────────────────

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze a medical image (lab report, prescription, X-ray, etc.)
    using Gemini Vision. Returns structured analysis or a polite rejection
    if the image is not medical-related.
    """
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        return JSONResponse(
            status_code=400,
            content={
                "status":           "error",
                "analysis_ar":      "نوع الملف غير مدعوم.",
                "technical_details": f"Unsupported file type: {file.content_type}",
                "model_used":       "none",
                "disclaimer":       MEDICAL_DISCLAIMER,
            },
        )

    try:
        image_bytes = await file.read()
    except Exception as e:
        log.error(f"Failed reading uploaded image: {e}")
        return JSONResponse(
            status_code=400,
            content={
                "status":           "error",
                "analysis_ar":      "فشل في قراءة الصورة المرفوعة.",
                "technical_details": str(e),
                "model_used":       "none",
                "disclaimer":       MEDICAL_DISCLAIMER,
            },
        )

    if len(image_bytes) > MAX_IMAGE_SIZE:
        return JSONResponse(
            status_code=400,
            content={
                "status":           "error",
                "analysis_ar":      "حجم الصورة أكبر من الحد المسموح.",
                "technical_details": f"Max allowed size is {MAX_IMAGE_SIZE // (1024 * 1024)}MB",
                "model_used":       "none",
                "disclaimer":       MEDICAL_DISCLAIMER,
            },
        )

    log.info(
        f"Analyzing image: {file.filename} "
        f"({len(image_bytes) / 1024:.1f} KB, {file.content_type})"
    )

    status, analysis_ar, technical_details, model_used = state.gemini.analyze_image(
        image_bytes,
        file.content_type,
    )

    http_status = 503 if status == "error" else 200

    return JSONResponse(
        status_code=http_status,
        content={
            "status":           status,
            "analysis_ar":      analysis_ar,
            "technical_details": technical_details,
            "model_used":       model_used,
            "disclaimer":       MEDICAL_DISCLAIMER,
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
        "social_awareness":         "enabled",
    }


@app.get("/")
def root():
    return {
        "name":      "Medical AI Assistant",
        "version":   "7.0.0",
        "mode":      "strict-rag + vision + social-awareness",
        "status":    "running",
        "endpoints": ["/ask", "/analyze-image", "/health"],
        "docs":      "/docs",
    }