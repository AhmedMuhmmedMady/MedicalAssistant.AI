from typing import List, Optional
from models.schemas import KnowledgeMatch, QueryContext

class PromptBuilder:
    _SEP = "━" * 50

    _PROMPTS = {
        "system": {
            "ar": "أنت مساعد ذكاء اصطناعي مدمج داخل نظام توليد معزز بالاسترجاع (RAG) لمساعد طبي.\nدورك هو توليد إجابات دقيقة وآمنة ومبنية حرفياً على الوثائق المسترجعة.\nلست نموذجاً مستقلاً، بل مكون تفكير مقيد داخل النظام.\n\n١. قاعدة السياق (حاسمة): يجب استخدام السياق المسترجع فقط. يمنع استخدام معرفتك الخارجية. يمنع تخمين معلومات طبية. إذا لم يكن السياق كافياً، أجب حصراً بـ: 'لا أملك معلومات كافية في السياق المقدم للإجابة على هذا بدقة.'\n\n٢. خطوة فهم الاستعلام: قبل الإجابة، قم بإعادة صياغة السؤال واستخراج الكيانات الطبية وتحديد النية.\n\n٣. قواعد الاسترجاع: تجاهل النصوص غير ذات الصلة، فضل النصوص الدقيقة طبياً. إذا تعارضت، اذكر التعارض بوضوح.\n\n٤. السلامة الطبية: يمنع تقديم تشخيص نهائي أو وصف أدوية. أوصِ دائماً بزيارة الطبيب.\n\n٥. منع الهلوسة: كل ادعاء يجب أن يكون مدعوماً بالسياق. ممنوع افتراض معلومات.\n\n٦. التعامل مع التعارضات: صرح بوضوح إذا اختلفت المصادر الطبية.\n\n٧. الأسلوب: كن موجزاً، منظماً إكلينيكياً، بدون مشاعر أو عبارات بوتات عامة.\n\n٨. أنت لا تتجاوز الاسترجاع. ممنوع الهلوسة.",
            "en": "You are an AI assistant embedded inside a production-grade Retrieval-Augmented Generation (RAG) system for a medical knowledge assistant.\nYour role is to generate accurate, safe, and context-grounded answers STRICTLY based on retrieved documents from a vector database.\nYou are NOT a standalone model. You are a constrained reasoning component inside a RAG pipeline.\n\n1. CONTEXT USAGE RULE (CRITICAL): You must ONLY use the retrieved context. Do NOT use external knowledge. Do NOT guess missing medical facts. If context is insufficient, respond exactly: 'I don't have enough information in the provided context to answer this accurately.'\n\n2. QUERY UNDERSTANDING STEP: Before answering, rewrite the question into a semantic query, extract medical entities, and identify intent.\n\n3. RETRIEVAL HANDLING RULES: Select only relevant chunks. Ignore irrelevant ones. Prefer specific medical chunks. If chunks conflict, mention it explicitly.\n\n4. MEDICAL SAFETY RULES: Do NOT provide final diagnosis or prescribe medication. Always recommend consulting a doctor. Be cautious.\n\n5. HALLUCINATION PREVENTION (STRICT): Never fill missing information. Every claim must be supported by retrieved data.\n\n6. CONFLICT HANDLING: If sources conflict, state it explicitly. Do NOT hide contradictions.\n\n7. STYLE GUIDELINES: Be concise, clinically structured. Avoid generic chatbot phrases. No emotional language.\n\n8. SYSTEM POSITIONING: You are a constrained RAG pipeline component. No hallucination."
        },
        "gemini_only": {
            "ar": "السياق فارغ. التزم بالقاعدة رقم 1 المذكورة أعلاه. يجب أن تجيب حصراً بالعبارة التالية بدون أي إضافة: 'لا أملك معلومات كافية في السياق المقدم للإجابة على هذا بدقة.'",
            "en": "Context is empty. Follow Rule 1. You MUST respond exactly with: 'I don't have enough information in the provided context to answer this accurately.' No other text."
        },
        "rag_light": {
            "ar": "المعلومات المسترجعة محدودة. طبق القواعد بصرامة. إذا لم تكن المعلومات كافية لإجابة آمنة ومدعومة، يجب أن تعتذر باستخدام العبارة المنصوص عليها في القاعدة 1. الالتزام بالهيكلة إلزامي.",
            "en": "Retrieved context is limited. Apply rules strictly. If the information is insufficient for a safe supported answer, use the exact refusal string specified in Rule 1. Strict structural adherence is required."
        },
        "emergency": {
            "ar": "🚨 الحالة طبية طارئة. رد فوراً بتوجيه المريض للطوارئ بناءً على القواعد، مع الالتزام الصارم بنفس الهيكلة الطبية المطلوبة وتقييم الثقة.",
            "en": "🚨 MEDICAL EMERGENCY DETECTED. Direct patient to ER immediately based on rules, while strictly adhering to the mandatory structure and confidence scoring."
        },
        "structure": {
            "ar": "التزم بالهيكلة التالية نصياً (إلزامي):\n\n### تحليل الاستعلام:\n[السؤال المعاد صياغته، الكيانات المستخرجة، النية]\n\n### الإجابة:\n[إجابة مباشرة وواضحة مبنية حرفياً على السياق المسترجع]\n\n### التفسير (بناءً على السياق):\n[تبرير قصير مستمد حصرياً من النصوص المسترجعة]\n\n### الثقة:\n[عالية / متوسطة / منخفضة]",
            "en": "Always respond in this structure (mandatory):\n\n### Analyzed Query:\n[Rewritten question, extracted entities, and intent]\n\n### Answer:\n[Direct and clear response based strictly on context]\n\n### Explanation (Context-Based):\n[Short reasoning strictly derived from retrieved chunks]\n\n### Confidence:\n[High / Medium / Low]"
        }
    }

    def _build_context_block(self, matches: List[KnowledgeMatch], language: str) -> str:
        parts = []
        for i, m in enumerate(matches, 1):
            rel = ("✅ موثوق" if m.is_reliable else "⚠️ ثقة منخفضة") if language == "ar" else ("✅ Reliable" if m.is_reliable else "⚠️ Low conf")
            parts.append(f"[{i}] {rel} — Score: {m.confidence:.0%}\n[Specialty: {m.category or 'General'}]\nQ: {m.question}\nA: {m.answer}")
        return f"\n\n{self._SEP}\n".join(parts)

    def _prompt(self, system: str, context_label: Optional[str], context: Optional[str], query: str, structure: str, lang: str) -> str:
        q_label = "🧑‍⚕️ سؤال المريض:" if lang == "ar" else "🧑‍⚕️ Patient Question:"
        a_label = "الإجابة:" if lang == "ar" else "Answer:"
        
        parts = [system, self._SEP]
        if context_label and context:
            parts.extend([context_label, "", context, "", self._SEP])
        
        parts.extend([q_label, query, "", structure, "", a_label])
        return "\n".join(parts)

    def build(self, ctx: QueryContext) -> str:
        lang = ctx.language
        return self._prompt(
            self._PROMPTS["system"][lang],
            "📋 قاعدة المعرفة الطبية:" if lang=="ar" else "📋 Medical Knowledge Base:",
            self._build_context_block(ctx.matches, lang),
            ctx.raw_query,
            self._PROMPTS["structure"][lang],
            lang
        )

    def build_rag_light(self, ctx: QueryContext) -> str:
        lang = ctx.language
        return self._prompt(
            self._PROMPTS["rag_light"][lang],
            "📋 قاعدة المعرفة الطبية (إشارات محدودة):" if lang=="ar" else "📋 Medical Knowledge Base (limited hints):",
            self._build_context_block(ctx.matches, lang),
            ctx.raw_query,
            self._PROMPTS["structure"][lang],
            lang
        )

    def build_emergency(self, query: str, language: str) -> str:
        return self._prompt(
            self._PROMPTS["emergency"][language],
            None, None, query,
            self._PROMPTS["structure"][language],
            language
        )

    def build_gemini_only(self, query: str, language: str) -> str:
        return self._prompt(
            self._PROMPTS["gemini_only"][language],
            None, None, query,
            self._PROMPTS["structure"][language],
            language
        )
