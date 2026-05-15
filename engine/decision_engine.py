import asyncio
import hashlib
from typing import Tuple, List

from core.constants import EMERGENCY_KEYWORDS_AR, EMERGENCY_KEYWORDS_EN, CAUSE_MAP, MEDICAL_KEYWORDS
from core.config import MIN_CONFIDENCE
from core.logging import log
from utils.text import normalize_text, NORMALIZED_SYMPTOM_MAP
from utils.cache import AsyncCache
from services.gemini_service import get_gemini_sync, gemini_types

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

    _cache = AsyncCache(maxsize=500)

    @classmethod
    async def classify(cls, query: str) -> str:
        cache_key = f"intent_{hashlib.sha256(query.encode()).hexdigest()}"
        
        async def _compute():
            q = query.lower()
            if any(kw in q for kw in MEDICAL_KEYWORDS):
                log.info("[Intent] Medical keyword detected instantly")
                return "medical"
            try:
                types = gemini_types()
                client = get_gemini_sync()
                
                def _call():
                    return client.models.generate_content(
                        model="gemini-2.0-flash-lite",
                        contents=cls._PROMPT.format(query=query),
                        config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=5),
                    )
                
                if hasattr(client, "aio"):
                    coro = client.aio.models.generate_content(
                        model="gemini-2.0-flash-lite",
                        contents=cls._PROMPT.format(query=query),
                        config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=5)
                    )
                else:
                    coro = asyncio.to_thread(_call)
                    
                resp = await asyncio.wait_for(coro, timeout=10.0)
                result = resp.text.strip().lower()
                return "medical" if "medical" in result else "social"
            except Exception as exc:
                log.warning(f"[Intent] Gemini failed: {exc} — defaulting to medical for safety")
                return "medical"
                
        val, _ = await cls._cache.get_or_compute(cache_key, _compute)
        return val

def generate_deterministic_fallback(query: str, language: str) -> str:
    q_norm = normalize_text(query)
    possible_causes = []
    specialties = []
    
    for kw, cats in NORMALIZED_SYMPTOM_MAP.items():
        if kw in q_norm:
            for cat in cats:
                cause = CAUSE_MAP.get(cat, "Medical condition")
                if cause not in possible_causes: possible_causes.append(cause)
                if cat not in specialties: specialties.append(cat)
    
    possible_causes = possible_causes[:3]
    specialty = specialties[0] if specialties else "general"
    
    red_flags = []
    emergency_kws = EMERGENCY_KEYWORDS_AR if language == "ar" else EMERGENCY_KEYWORDS_EN
    for kw in emergency_kws:
        if kw.lower() in query.lower(): red_flags.append(kw)
    
    if language == "ar":
        causes_text = "\n".join([f"→ {c}" for c in possible_causes]) if possible_causes else "→ حالة طبية عامة"
        red_flags_text = "\n".join([f"→ {rf}" for rf in red_flags[:3]]) if red_flags else "لم يتم اكتشاف علامات خطر فورية"
        return (
            f"🔍 الأسباب المحتملة:\n{causes_text}\n\n"
            f"🚨 علامات الخطر:\n{red_flags_text}\n\n"
            f"❓ الأسئلة:\n→ كم مدة الأعراض؟\n→ ما هي شدة الأعراض؟\n→ هل هناك أعراض أخرى مصاحبة؟\n\n"
            f"💊 النصائح:\n→ الحفاظ على الترطيب والراحة\n→ مراقبة الأعراض عن كثب\n→ تجنب الأدوية دون استشارة طبية\n\n"
            f"🏥 التوصية:\n→ مراجعة طبيب {specialty} في أقرب وقت\n→ مستوى الإلحاح: متوسط"
        )
    else:
        causes_text = "\n".join([f"→ {c}" for c in possible_causes]) if possible_causes else "→ General medical condition"
        red_flags_text = "\n".join([f"→ {rf}" for rf in red_flags[:3]]) if red_flags else "No immediate red flags detected"
        return (
            f"🔍 Possible Causes:\n{causes_text}\n\n"
            f"🚨 Red Flags:\n{red_flags_text}\n\n"
            f"❓ Questions:\n→ How long have symptoms lasted?\n→ What is the severity?\n→ Are there other associated symptoms?\n\n"
            f"💊 Advice:\n→ Maintain hydration and rest\n→ Monitor symptoms closely\n→ Avoid self-medication\n\n"
            f"🏥 Recommendation:\n→ Consult a {specialty} specialist soon\n→ Urgency level: moderate"
        )

def determine_rag_mode(is_emg: bool, selected_matches: List, top_score: float, rel_ok: bool, garbage_r: float, cat_cons: float) -> Tuple[str, str]:
    if is_emg:
        return "EMERGENCY_OVERRIDE", "emergency_keyword_match"
    elif not selected_matches or top_score < MIN_CONFIDENCE:
        return "GEMINI_ONLY", f"no_valid_matches(score={top_score:.3f})"
    elif not rel_ok:
        return "GEMINI_ONLY", "failed_relevance_guard"
    elif garbage_r > 0.5:
        return "GEMINI_ONLY", f"high_garbage_ratio={garbage_r:.2f}"
    elif top_score >= 0.80 and cat_cons >= 0.7:
        return "RAG_STRONG", f"high_confidence={top_score:.3f}_and_consistent"
    elif top_score >= MIN_CONFIDENCE:
        return "RAG_LIGHT", f"medium_confidence={top_score:.3f}"
    else:
        return "GEMINI_ONLY", "fallback_decision"
