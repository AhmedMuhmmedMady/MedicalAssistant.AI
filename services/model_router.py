import httpx
from typing import Dict, Any, Tuple
from core.config import (
    OPENROUTER_API_KEY, GROQ_API_KEY, PRIMARY_MODEL, ENABLE_FALLBACK,
    GITHUB_TOKEN_PHI4, GITHUB_TOKEN_GPT4_MINI, GITHUB_TOKEN_GPT4, GITHUB_TOKEN_GROK3, GITHUB_TOKEN_GPT5
)
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

        # 2. GitHub Models Fallback (Premium HA Rotating Layer)
        if ENABLE_FALLBACK:
            try:
                log.info("MODEL_ATTEMPT: GitHub Models Fallback Pool")
                response, friendly_name = await self._call_github_fallback(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    image_bytes=image_bytes,
                    mime_type=mime_type
                )
                log.info(f"MODEL_SUCCESS: GitHub Models Fallback Pool ({friendly_name})")
                return {
                    "status": "fallback",
                    "model_used": friendly_name,
                    "response": response
                }
            except Exception as e:
                log.error(f"MODEL_FAILED: GitHub Models Fallback Pool | {e}")

        # 3. Groq
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
            
        # 4. OpenRouter Fallback Pool
        try:
            log.info("MODEL_ATTEMPT: OpenRouter Fallback Pool")
            response, friendly_name = await self._call_openrouter_fallback(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                image_bytes=image_bytes,
                mime_type=mime_type
            )
            log.info(f"MODEL_SUCCESS: OpenRouter Fallback Pool ({friendly_name})")
            return {
                "status": "fallback",
                "model_used": friendly_name,
                "response": response
            }
        except Exception as e:
            log.error(f"MODEL_FAILED: OpenRouter Fallback Pool | {e}")
            
        # 5. Local Model (Rule-based RAG offline fallback)
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

    async def _call_github_fallback(
        self, prompt: str, temperature: float, max_tokens: int,
        image_bytes: bytes = None, mime_type: str = None
    ) -> Tuple[str, str]:
        """
        Premium fallback layer that rotates through high-availability GitHub Models tokens
        to execute GPT-4o, GPT-4o-mini, or Phi-4 clinical reasoning.
        """
        github_endpoints = []
        
        # Build candidate list with tokens, models, and metadata
        if GITHUB_TOKEN_GPT4:
            github_endpoints.append((GITHUB_TOKEN_GPT4, "gpt-4o", "github-gpt-4.1"))
        if GITHUB_TOKEN_GROK3:
            github_endpoints.append((GITHUB_TOKEN_GROK3, "gpt-4o", "github-grok-3"))
        if GITHUB_TOKEN_GPT5:
            github_endpoints.append((GITHUB_TOKEN_GPT5, "gpt-4o", "github-gpt-5"))
        if GITHUB_TOKEN_GPT4_MINI:
            github_endpoints.append((GITHUB_TOKEN_GPT4_MINI, "gpt-4o-mini", "github-gpt-4.1-mini"))
        if GITHUB_TOKEN_PHI4:
            github_endpoints.append((GITHUB_TOKEN_PHI4, "Phi-4", "github-phi-4"))
            
        base_url = "https://models.github.ai/inference/chat/completions"
        
        for token, model, name in github_endpoints:
            # Phi-4 does not support vision
            if image_bytes and model == "Phi-4":
                log.warning(f"[Model Router] Skipping GitHub model {name} ({model}) for image input since Phi-4 does not support vision.")
                continue
                
            try:
                log.info(f"MODEL_ATTEMPT: GitHub Model {name} ({model})")
                reply = await self._call_openai_compatible(
                    base_url=base_url,
                    api_key=token,
                    model_name=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    image_bytes=image_bytes,
                    mime_type=mime_type
                )
                log.info(f"MODEL_SUCCESS: GitHub Model {name} ({model})")
                return reply, name
            except Exception as e:
                log.error(f"MODEL_FAILED: GitHub Model {name} ({model}) | {e}")
                
        raise RuntimeError("All GitHub Models fallback options exhausted.")

    async def _call_openrouter_fallback(
        self, prompt: str, temperature: float, max_tokens: int,
        image_bytes: bytes = None, mime_type: str = None
    ) -> Tuple[str, str]:
        """
        Fallback layer that rotates through curated, free OpenRouter models
        ordered from strongest/largest to weakest/most efficient.
        """
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is not set.")

        # If vision is requested, use Gemini 2.5 Flash on OpenRouter
        if image_bytes and mime_type:
            vision_model = "google/gemini-2.5-flash"
            log.info(f"MODEL_ATTEMPT: OpenRouter Vision ({vision_model})")
            reply = await self._call_openai_compatible(
                base_url="https://openrouter.ai/api/v1/chat/completions",
                api_key=OPENROUTER_API_KEY,
                model_name=vision_model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                image_bytes=image_bytes,
                mime_type=mime_type,
                extra_headers={"HTTP-Referer": "https://your-domain.com", "X-Title": "Mady Medical AI"}
            )
            return reply, f"openrouter-{vision_model}"

        # Otherwise, iterate through text models from strongest to weakest
        openrouter_models = [
            ("nousresearch/hermes-3-llama-3.1-405b:free", "Hermes-3-Llama-3.1-405B"),
            ("openai/gpt-oss-120b:free", "GPT-OSS-120B"),
            ("meta-llama/llama-3.3-70b-instruct:free", "Llama-3.3-70B-Instruct"),
            ("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", "Nemotron-3-Nano-Omni-30B"),
            ("openai/gpt-oss-20b:free", "GPT-OSS-20B"),
            ("poolside/laguna-m.1:free", "Laguna-M.1"),
            ("openrouter/owl-alpha", "Owl-Alpha"),
            ("meta-llama/llama-3.2-3b-instruct:free", "Llama-3.2-3B-Instruct"),
            ("poolside/laguna-xs.2:free", "Laguna-XS.2"),
            ("baidu/cobuddy:free", "CoBuddy")
        ]

        extra_headers = {"HTTP-Referer": "https://your-domain.com", "X-Title": "Mady Medical AI"}

        for model_id, friendly_name in openrouter_models:
            try:
                log.info(f"MODEL_ATTEMPT: OpenRouter Fallback Pool ({friendly_name})")
                reply = await self._call_openai_compatible(
                    base_url="https://openrouter.ai/api/v1/chat/completions",
                    api_key=OPENROUTER_API_KEY,
                    model_name=model_id,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    image_bytes=image_bytes,
                    mime_type=mime_type,
                    extra_headers=extra_headers
                )
                log.info(f"MODEL_SUCCESS: OpenRouter Fallback Pool ({friendly_name})")
                return reply, f"openrouter-{friendly_name}"
            except Exception as e:
                log.error(f"MODEL_FAILED: OpenRouter Fallback Pool ({friendly_name}) | {e}")

        raise RuntimeError("All OpenRouter Fallback Pool models exhausted.")

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
