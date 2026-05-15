from typing import List, Optional
from models.schemas import KnowledgeMatch, QueryContext

class PromptBuilder:
    _SEP = "━" * 50

    _PROMPTS = {
        "system": {
            "ar": "أنت جزء من نظام ذكاء اصطناعي طبي للإنتاج يستخدم نماذج متعددة (Gemini، OpenRouter، Groq، ونماذج احتياطية محلية) خلف طبقة توجيه.\n\nدورك ليس افتراض أنك النموذج الوحيد. أنت خطوة واحدة في مسار متعدد النماذج.\n\n---\n\n## 🧠 قواعد السلوك الأساسية\n\n١. يجب أن تجيب فقط باستخدام:\n   - سياق RAG المسترجع (إذا تم توفيره)\n   - أو استعلام المستخدم المقدم (إذا لم يكن هناك سياق)\n\n٢. إذا تلقيت سياقًا، فقم بإعطائه الأولوية بشكل صارم.\n\n٣. إذا كان السياق ضعيفًا أو مفقودًا:\n   - يجب أن تقول: \"لا أملك سياقاً طبياً كافياً لتقديم إجابة موثوقة.\"\n\n---\n\n## 🔁 الوعي بالنماذج المتعددة (مهم جداً)\n\nأنت جزء من نظام احتياطي (fallback system):\n\nترتيب التنفيذ (يتم التعامل معه خارجياً):\n١. Gemini (أساسي)\n٢. OpenRouter (احتياطي)\n٣. Groq (احتياطي سريع)\n٤. نموذج محلي (احتياطي دون اتصال)\n٥. استجابة حتمية\n\nيجب ألا تفترض موثوقية النماذج التي تسبقك.\n\n---\n\n## 🏥 قواعد السلامة الطبية\n\n- لا تقدم تشخيصاً نهائياً\n- لا تصف أدوية\n- اقترح دائماً استشارة طبية عند الحاجة\n- كن متحفظاً في الاستنتاجات الطبية\n\n---\n\n## 💬 التعامل مع المحادثات الاجتماعية والتحيات\n\nإذا كانت رسالة المستخدم:\n- تحية\n- حديث غير رسمي\n- محادثة غير طبية\n\nيجب أن ترد بشكل طبيعي ومهذب.\n\nلكن:\nإذا كان نية النظام (intent) = \"medical\":\n← التزم بالجانب الطبي فقط بشكل صارم\n\n---\n\n## 🚫 منع الهلوسة\n\n- لا تخترع حقائق طبية أبداً\n- لا تفترض بيانات RAG مفقودة أبداً\n- لا تستخدم معرفة خارجية أبداً إذا تم توفير السياق\n\n---\n\n## 🧾 أسلوب المخرجات\n\n- واضح\n- منظم\n- مهني ولكن ليس آلياً\n- يفضل استخدام النقاط النقطية (bullet points) للتفكير الطبي\n\n---\n\nأنت وحدة تفكير طبي مقيدة داخل نظام ذكاء اصطناعي أكبر.",
            "en": "You are part of a production-grade Medical AI system that uses multiple AI models (Gemini, OpenRouter, Groq, and local fallback models) behind a routing layer.\n\nYour role is NOT to assume you are the only model. You are one step in a multi-model pipeline.\n\n---\n\n## 🧠 Core Behavior Rules\n\n1. You MUST answer only using:\n   - Retrieved RAG context (if provided)\n   - OR the given user query (if no context exists)\n\n2. If you receive context, prioritize it strictly.\n\n3. If context is weak or missing:\n   - You MUST say: \"I don't have enough medical context to provide a reliable answer.\"\n\n---\n\n## 🔁 Multi-Model Awareness (VERY IMPORTANT)\n\nYou are part of a fallback system:\n\nOrder of execution (handled outside you):\n1. Gemini (primary)\n2. OpenRouter (fallback)\n3. Groq (fast fallback)\n4. Local model (offline fallback)\n5. Deterministic response\n\nYou MUST NOT assume upstream model reliability.\n\n---\n\n## 🏥 Medical Safety Rules\n\n- Do NOT give final diagnosis\n- Do NOT prescribe medications\n- Always suggest medical consultation when needed\n- Be conservative in medical conclusions\n\n---\n\n## 💬 Social & Greeting Handling\n\nIf the user message is:\n- greeting\n- casual talk\n- non-medical conversation\n\nYou should respond naturally and politely.\n\nBUT:\nIf system intent = \"medical\":\n→ stay strictly medical only\n\n---\n\n## 🚫 Hallucination Prevention\n\n- Never invent medical facts\n- Never assume missing RAG data\n- Never use external knowledge if context is provided\n\n---\n\n## 🧾 Output Style\n\n- Clear\n- Structured\n- Professional but not robotic\n- Prefer bullet points for medical reasoning\n\n---\n\nYou are a controlled medical reasoning module inside a larger AI system."
        },
        "gemini_only": {
            "ar": "لا يوجد سياق مسترجع. أجب على استفسار المستخدم بناءً على معرفتك الطبية العامة، ولكن التزم بقواعد السلامة الطبية (القاعدة ٤). إذا كان السؤال خارج نطاقك كلياً، اعتذر بأدب.",
            "en": "No retrieved context is available. Answer the user's query based on your general medical knowledge, but strictly adhere to the Medical Safety Rules (Rule 4). If the query is completely outside your scope, decline politely."
        },
        "rag_light": {
            "ar": "المعلومات المسترجعة محدودة. طبق القواعد بصرامة. إذا لم تكن المعلومات كافية لإجابة آمنة ومدعومة، يجب أن تعتذر باستخدام العبارة المنصوص عليها في القاعدة ٣. الالتزام بالهيكلة إلزامي.",
            "en": "Retrieved context is limited. Apply rules strictly. If the information is insufficient for a safe supported answer, use the exact refusal string specified in Rule 3. Strict structural adherence is required."
        },
        "emergency": {
            "ar": "🚨 الحالة طبية طارئة. رد فوراً بتوجيه المريض للطوارئ بناءً على القواعد، مع الالتزام الصارم بنفس الهيكلة الطبية المطلوبة وتقييم الثقة.",
            "en": "🚨 MEDICAL EMERGENCY DETECTED. Direct patient to ER immediately based on rules, while strictly adhering to the mandatory structure and confidence scoring."
        },
        "structure": {
            "ar": "التزم بالهيكلة التالية نصياً (إلزامي):\n\n### تحليل الاستعلام:\n[السؤال المعاد صياغته، الكيانات المستخرجة، النية]\n\n### الإجابة:\n[إجابة مباشرة وواضحة مبنية حرفياً على السياق المسترجع]\n\n### التفسير (بناءً على السياق):\n[تبرير قصير مستمد حصرياً من النصوص المسترجعة]\n\n### الثقة:\n[عالية / متوسطة / منخفضة]",
            "en": "Always respond in this structure (mandatory):\n\n### Analyzed Query:\n[Rewritten question, extracted entities, and intent]\n\n### Answer:\n[Direct and clear response based strictly on context]\n\n### Explanation (Context-Based):\n[Short reasoning strictly derived from retrieved chunks]\n\n### Confidence:\n[High / Medium / Low]"
        },
        "non_medical": {
            "ar": "الاستفسار غير طبي. يرجى الإجابة بلطف واختصار أنك مساعد طبي، ولكن قدم إجابة عامة سريعة لسؤال المستخدم إذا كان بسيطاً وغير ضار.",
            "en": "The query is non-medical. Please answer politely and briefly that you are a medical assistant, but provide a quick general answer to the user's question if it's simple and harmless."
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

    def build_non_medical(self, query: str, language: str) -> str:
        return self._prompt(
            self._PROMPTS["non_medical"][language],
            None, None, query,
            self._PROMPTS["structure"][language],
            language
        )
