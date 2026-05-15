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
                log.info("[Router] Attempting Gemini (Primary)")
                if image_bytes and mime_type:
                    status, reply, model = await self.gemini_service.analyze_image(image_bytes, mime_type)
                    if status == "error":
                        raise RuntimeError("Gemini image analysis failed")
                else:
                    reply, model = await self.gemini_service.generate(prompt)
                return {
                    "status": "success",
                    "model_used": model,
                    "response": reply
                }
            except Exception as e:
                log.error(f"[Router] Gemini failed: {e}")
                if not ENABLE_FALLBACK:
                    return self._fallback_deterministic(query, language)

        # 2. OpenRouter
        try:
            log.info("[Router] Attempting OpenRouter (Fallback 1)")
            response = await self._call_openrouter(prompt, temperature, max_tokens, image_bytes)
            return {
                "status": "fallback",
                "model_used": "openrouter-llama3.1-8b",
                "response": response
            }
        except Exception as e:
            log.error(f"[Router] OpenRouter failed: {e}")

        # 3. Groq
        try:
            log.info("[Router] Attempting Groq (Fallback 2)")
            response = await self._call_groq(prompt, temperature, max_tokens, image_bytes)
            return {
                "status": "fallback",
                "model_used": "groq-llama3.1-70b",
                "response": response
            }
        except Exception as e:
            log.error(f"[Router] Groq failed: {e}")
            
        # 4. Local Model (Rule-based RAG offline fallback)
        try:
            log.info("[Router] Attempting Local/Deterministic Model (Fallback 3)")
            return self._fallback_deterministic(query, language)
        except Exception as e:
            log.error(f"[Router] Local model failed: {e}")
            return {
                "status": "fallback",
                "model_used": "deterministic-safeguard",
                "response": "عذراً، أواجه مشكلة تقنية. يرجى استشارة طبيب متخصص." if language == "ar" else "Sorry, I am facing a technical issue. Please consult a medical professional."
            }

    async def _call_openrouter(self, prompt: str, temperature: float, max_tokens: int, image_bytes: bytes = None) -> str:
        if image_bytes:
            raise ValueError("Image payloads not supported by text-only OpenRouter model")
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is not set")
            
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "meta-llama/llama-3.1-8b-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = await self.client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()

    async def _call_groq(self, prompt: str, temperature: float, max_tokens: int, image_bytes: bytes = None) -> str:
        if image_bytes:
            raise ValueError("Image payloads not supported by text-only Groq model")
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set")
            
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "llama-3.1-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = await self.client.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
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
