import json
import re
from typing import Dict, List, Tuple, Optional
from core.logging import log
from utils.cache import AsyncCache
from rag.query_processor import ArabicQueryProcessor

# Namespaces List
NAMESPACES = [
    "internal_medicine",
    "pediatrics",
    "obstetrics_gynecology",
    "general_surgery",
    "ophthalmology",
    "dentistry",
    "dermatology",
    "psychiatry",
    "cardiology",
    "orthopedics",
    "ent",
    "urology"
]

# Keywords mapped to each medical specialty namespace
SPECIALTY_KEYWORDS: Dict[str, List[str]] = {
    "internal_medicine": [
        "سحر", "ضغط", "سكر", "غدة", "هرمون", "كبد", "كلى", "قولون", "معدة", "جهاز هضمي", 
        "امساك", "اسهال", "مرارة", "حموضة", "جاف", "رئة", "ربو", "بلغم", "تسمم", "سخونية", 
        "حرارة", "حمى", "برد", "انفلونزا", "رشح", "زكام"
    ],
    "pediatrics": [
        "طفل", "أطفال", "رضيع", "رضاعة", "تطعيم", "تبول لاإرادي", "تسنين", "مغص رضع", 
        "طفلة", "صغيري", "بيبي"
    ],
    "obstetrics_gynecology": [
        "حمل", "ولادة", "رحم", "دورة", "حيض", "مبيض", "تكيس", "إجهاض", "جنين", "إباضة", 
        "افرازات مهبلية", "مهبل", "ثدي", "ولاده", "حامل"
    ],
    "general_surgery": [
        "عملية", "جراحة", "استئصال", "فتق", "زائدة", "خياطة", "منظار", "بواسير", "ناسور", 
        "خراج", "ورم", "سرطان", "غرز", "شرح جراحي"
    ],
    "ophthalmology": [
        "عين", "نظر", "رؤية", "رمد", "قرنية", "شبكية", "مياه بيضاء", "نظارة", "احمرار عين", 
        "جفاف عين", "حول", "رمش", "عدسات"
    ],
    "dentistry": [
        "أسنان", "ضرس", "لثة", "تسوس", "تقويم", "حشو", "خلع", "تبييض أسنان", "رائحة فم", 
        "اسنان", "ضرسي", "لثتي"
    ],
    "dermatology": [
        "جلد", "طفح", "حكة", "هرش", "حبوب", "بشرة", "صدفية", "إكزيما", "تساقط شعر", "ثعلبة", 
        "بهاق", "حساسية جلدية", "تسلخات", "قشرة رأس"
    ],
    "psychiatry": [
        "قلق", "اكتئاب", "توتر", "نفسية", "وسواس", "هلوسة", "انعزال", "خوف", "رهاب", "ارق", 
        "نوم", "انفصام", "نوبات هلع", "حزن", "نفسي"
    ],
    "cardiology": [
        "قلب", "نبض", "شريان", "ذبحة", "نهجان", "خفقان", "صمام", "شرايين", "ضربات قلب", 
        "الم صدر", "تسارع نبض"
    ],
    "orthopedics": [
        "عظام", "مفاصل", "ركبة", "فقرات", "كسر", "ظهر", "غضروف", "روماتيزم", "التواء", 
        "عمود فقري", "خشونة ركبة", "تمزق اربطة", "كتف", "رقبة"
    ],
    "ent": [
        "أذن", "أنف", "حنجرة", "حلق", "لوز", "سمع", "دوخة", "طنين", "جيوب أنفية", "بلع", 
        "احتقان", "صوت", "بحة", "ودني"
    ],
    "urology": [
        "بول", "مثانة", "بروستاتا", "حصوة", "حرقان بول", "مسالك", "كلى", "تبول", "خصية", 
        "عقم", "ضعف جنسي"
    ]
}

class MedicalNamespaceRouter:
    """
    Intelligent router that determines which Pinecone namespace to target 
    based on a combination of fast rule-based matching and a fallback Gemini LLM routing layer.
    """
    
    def __init__(self, gemini_service = None):
        self.gemini_service = gemini_service
        self._cache = AsyncCache(maxsize=500)

    def route_by_keywords(self, normalized_query: str) -> Tuple[Optional[str], float]:
        """
        Performs fast, local rule-based keyword matching.
        Returns the predicted namespace and a confidence score.
        """
        scores = {ns: 0 for ns in NAMESPACES}
        words = normalized_query.split()
        
        for ns, kws in SPECIALTY_KEYWORDS.items():
            for kw in kws:
                kw_norm = ArabicQueryProcessor.clean_and_normalize(kw)
                # Exact phrase matching
                if kw_norm in normalized_query:
                    scores[ns] += 3
                # Individual word matches
                for word in words:
                    if word == kw_norm:
                        scores[ns] += 1
                        
        total_score = sum(scores.values())
        if total_score == 0:
            return None, 0.0
            
        best_ns = max(scores, key=scores.get)
        confidence = scores[best_ns] / total_score
        
        # Boost confidence if the absolute match score is high
        if scores[best_ns] >= 6:
            confidence = min(0.95, confidence + 0.15)
            
        return best_ns, round(confidence, 2)

    async def route_by_llm(self, query: str) -> Tuple[str, float]:
        """
        Calls Gemini to perform a medical classification of the query.
        Returns predicted namespace and confidence score.
        """
        if not self.gemini_service:
            log.warning("[Router] GeminiService not provided to router, defaulting to general search.")
            return "internal_medicine", 0.1
            
        prompt = (
            "أنت نظام طبي ذكي لتصنيف الأسئلة الطبية وتوجيهها للقسم المناسب (Medical Triage Router).\n"
            "قم بتصنيف السؤال التالي للمريض إلى تخصص طبي واحد بالضبط من القائمة التالية:\n"
            f"{', '.join(NAMESPACES)}\n\n"
            "يجب أن تكون الإجابة عبارة عن كود JSON صالح تماماً بدون أي علامات تشفير أو كود (دون ```json) بالهيكل التالي:\n"
            "{\n"
            '  "category": "اسم التخصص من القائمة حرفياً",\n'
            '  "confidence": نسبة الثقة من 0.0 إلى 1.0,\n'
            '  "reason": "سبب اختيارك باللغة العربية باختصار"\n'
            "}\n\n"
            f"سؤال المريض: {query}\n\n"
            "JSON Output:"
        )
        
        try:
            raw_response, _ = await self.gemini_service.generate(prompt)
            # Sanitise JSON fences if model outputted them
            clean_json = re.sub(r"^\s*```+(?:json)?\s*|\s*```+\s*$", "", raw_response, flags=re.MULTILINE).strip()
            data = json.loads(clean_json)
            category = data.get("category", "").strip().lower()
            confidence = float(data.get("confidence", 0.5))
            
            if category in NAMESPACES:
                return category, confidence
            else:
                # Fallback matching
                for ns in NAMESPACES:
                    if ns in category:
                        return ns, confidence
                return "internal_medicine", 0.1
        except Exception as exc:
            log.error(f"[Router] LLM routing failed: {exc}")
            return "internal_medicine", 0.1

    async def route_query(self, query: str) -> Dict[str, any]:
        """
        Routes the user query using a hybrid caching approach.
        Attempts keyword routing first, and uses Gemini routing as a fallback.
        """
        normalized_query = ArabicQueryProcessor.clean_and_normalize(query)
        cache_key = f"route_{hash(normalized_query)}"
        
        async def _compute_routing():
            # 1. Keyword-based routing
            ns, conf = self.route_by_keywords(normalized_query)
            if ns and conf >= 0.70:
                log.info(f"[Router] ⚡ Fast Keyword Route SUCCESS: {ns} (conf={conf:.2f})")
                return {
                    "primary_namespace": ns,
                    "confidence": conf,
                    "all_candidate_namespaces": [ns],
                    "method": "keyword"
                }
                
            # 2. LLM-based routing
            log.info(f"[Router] 🔄 Fast Keyword Route weak or missing. Invoking Gemini Router...")
            llm_ns, llm_conf = await self.route_by_llm(query)
            
            # Formulate parallel routing candidates if confidence is moderate
            candidates = [llm_ns]
            
            # If confidence is low, fall back to adding a secondary namespace or allow general fallback
            if llm_conf < 0.65:
                # Add internal_medicine and cardiology as standard fallbacks for general/adult issues
                if "internal_medicine" not in candidates:
                    candidates.append("internal_medicine")
                log.info(f"[Router] ⚠️ Low confidence LLM Route: {llm_ns} (conf={llm_conf:.2f}). Triggering multi-namespace search.")
            else:
                log.info(f"[Router] 🎯 Gemini Route SUCCESS: {llm_ns} (conf={llm_conf:.2f})")
                
            return {
                "primary_namespace": llm_ns,
                "confidence": llm_conf,
                "all_candidate_namespaces": candidates,
                "method": "gemini"
            }

        result, _ = await self._cache.get_or_compute(cache_key, _compute_routing)
        return result
