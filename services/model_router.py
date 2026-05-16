import httpx
from typing import Dict, Any, Tuple
from core.config import OPENROUTER_API_KEY, GROQ_API_KEY, PRIMARY_MODEL, ENABLE_FALLBACK
from core.logging import log
from services.gemini_service import GeminiService
from engine.decision_engine import generate_deterministic_fallback

class ModelRouter:
    def __init__(self, gemini_service: GeminiService):
        self.gemini_service = gemini_service
        self.client = httpx.AsyncClient(timeout=20.0)

    async def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        payload expects:
        - prompt: the fully formatted string
        - query: the original user query (for local fallback)
        - language: the language detected (for local fallback)
        - temperature: optional float
        - max_tokens: optional int
        - image_bytes: optional bytes
        - mime_type: optional str
        """
        prompt = payload.get("prompt", "")
        query = payload.get("query", "")
        language = payload.get("language", "en")
        temperature = payload.get("temperature", 0.2)
        max_tokens = payload.get("max_tokens", 2048)
        image_bytes = payload.get("image_bytes")
        mime_type = payload.get("mime_type")

        # 1. Gemini
        if PRIMARY_MODEL == "gemini":
            try:
                log.info("MODEL_ATTEMPT: Gemini")
                if image_bytes and mime_type:
                    status, reply, model = await self.gemini_service.analyze_image(image_bytes, mime_type)
                    if status == "error":
                        raise RuntimeError("Gemini image analysis failed")
                else:
                    reply, model = await self.gemini_service.generate(prompt)
                log.info("MODEL_SUCCESS: Gemini")
                return {
                    "status": "success",
                    "model_used": model,
                    "response": reply
                }
            except Exception as e:
                log.error(f"MODEL_FAILED: Gemini | {e}")
                if not ENABLE_FALLBACK:
                    return self._fallback_deterministic(query, language)

        # 2. Groq
        try:
            log.info("MODEL_ATTEMPT: Groq")
            model_name = "llama-3.2-90b-vision-preview" if (image_bytes and mime_type) else "llama-3.3-70b-versatile"
            response = await self._call_openai_compatible(
                base_url="https://api.groq.com/openai/v1/chat/completions",
                api_key=GROQ_API_KEY,
                model_name=model_name,
                prompt=prompt, temperature=temperature, max_tokens=max_tokens,
                image_bytes=image_bytes, mime_type=mime_type
            )
            log.info("MODEL_SUCCESS: Groq")
            return {
                "status": "fallback",
                "model_used": "groq-fallback",
                "response": response
            }
        except Exception as e:
            log.error(f"MODEL_FAILED: Groq | {e}")
            
        # 3. OpenRouter
        try:
            log.info("MODEL_ATTEMPT: OpenRouter")
            model_name = "google/gemini-2.5-flash" if (image_bytes and mime_type) else "meta-llama/llama-3.1-8b-instruct"
            extra_headers = {"HTTP-Referer": "https://your-domain.com", "X-Title": "Mady Medical AI"}
            response = await self._call_openai_compatible(
                base_url="https://openrouter.ai/api/v1/chat/completions",
                api_key=OPENROUTER_API_KEY,
                model_name=model_name,
                prompt=prompt, temperature=temperature, max_tokens=max_tokens,
                image_bytes=image_bytes, mime_type=mime_type,
                extra_headers=extra_headers
            )
            log.info("MODEL_SUCCESS: OpenRouter")
            return {
                "status": "fallback",
                "model_used": "openrouter-fallback",
                "response": response
            }
        except Exception as e:
            log.error(f"MODEL_FAILED: OpenRouter | {e}")
            
        # 4. Local Model (Rule-based RAG offline fallback)
        try:
            log.info("MODEL_ATTEMPT: LocalDeterministic")
            res = self._fallback_deterministic(query, language)
            log.info("MODEL_SUCCESS: LocalDeterministic")
            return res
        except Exception as e:
            log.error(f"MODEL_FAILED: LocalDeterministic | {e}")
            return {
                "status": "fallback",
                "model_used": "deterministic-safeguard",
                "response": "عذراً، أواجه مشكلة تقنية. يرجى استشارة طبيب متخصص." if language == "ar" else "Sorry, I am facing a technical issue. Please consult a medical professional."
            }

    async def _call_openai_compatible(
        self, base_url: str, api_key: str, model_name: str, 
        prompt: str, temperature: float, max_tokens: int, 
        image_bytes: bytes = None, mime_type: str = None, 
        extra_headers: dict = None
    ) -> str:
        if not api_key:
            raise ValueError(f"API key is not set for {base_url}")
            
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        if extra_headers:
            headers.update(extra_headers)
            
        if image_bytes and mime_type:
            if not prompt: prompt = "قم بتحليل هذه الصورة الطبية واستخراج الأسباب المحتملة، علامات الخطر، والتوصيات بدقة باللغة العربية."
            import base64
            b64_img = base64.b64encode(image_bytes).decode('utf-8')
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_img}"}}
            ]
        else:
            content = prompt

        data = {
            "model": model_name,
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = await self.client.post(base_url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()

    def _fallback_deterministic(self, query: str, language: str) -> Dict[str, Any]:
        reply = generate_deterministic_fallback(query, language)
        return {
            "status": "fallback",
            "model_used": "local-deterministic",
            "response": reply
        }

    async def close(self):
        await self.client.aclose()
