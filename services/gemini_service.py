import asyncio
import hashlib
import json
import re
import threading
from typing import Tuple
from core.config import GEMINI_API_KEY, EXTERNAL_CALL_TIMEOUT
from core.constants import GEMINI_TEXT_MODELS, GEMINI_VISION_MODELS
from core.logging import log
from utils.cache import AsyncCache

_gemini_client = None
_gemini_lock   = threading.Lock()

def _init_gemini():
    global _gemini_client
    if _gemini_client is None:
        with _gemini_lock:
            if _gemini_client is None:
                log.info("🔄 Initialising Gemini client…")
                from google import genai as _genai
                _gemini_client = _genai.Client(api_key=GEMINI_API_KEY)
                log.info("✅ SDK type detected: google.genai")
                log.info(f"✅ vision model selected: {GEMINI_VISION_MODELS[0]}")
                log.info("✅ Gemini client ready.")
    return _gemini_client

def get_gemini_sync(): return _init_gemini()
def gemini_types():
    from google.genai import types
    return types

class GeminiService:
    def __init__(self) -> None:
        self._async_cache = AsyncCache(200)

    @staticmethod
    def _make_config(temperature: float = 0.2, max_tokens: int = 2048):
        types = gemini_types()
        return types.GenerateContentConfig(temperature=temperature, max_output_tokens=max_tokens)

    async def reply_social(self, query: str, language: str) -> Tuple[str, str]:
        cache_key = f"social_{language}_{hashlib.sha256(query.encode()).hexdigest()}"
        
        async def _compute():
            system = (
                "أنت 'سيلا'، مساعد طبي ذكي وودود. رد بالعربية بشكل طبيعي ودافئ. الرد قصير (جملة أو اتنين)."
                if language == "ar" else
                "You are 'Sila', a friendly medical AI. Reply naturally in English. Keep it brief (1-2 sentences)."
            )
            types = gemini_types()
            client = get_gemini_sync()
            
            for model in GEMINI_TEXT_MODELS:
                try:


                    def _call():
                        return client.models.generate_content(
                            model=model, contents=query,
                            config=types.GenerateContentConfig(system_instruction=system, temperature=0.75, max_output_tokens=200),
                        )
                    
                    if hasattr(client, "aio"):
                        coro = client.aio.models.generate_content(
                            model=model, contents=query,
                            config=types.GenerateContentConfig(system_instruction=system, temperature=0.75, max_output_tokens=200)
                        )
                    else:
                        coro = asyncio.to_thread(_call)
                        
                    resp = await asyncio.wait_for(coro, timeout=15.0)
                    text = resp.text.strip()
                    if text: return text, model
                except asyncio.TimeoutError:
                    log.error(f"[Social] {model} timed out")
                except Exception as exc:
                    log.warning(f"[Social] {model} failed: {exc}")
            raise RuntimeError("All models failed for social")
            
        try:
            val, _ = await self._async_cache.get_or_compute(cache_key, _compute)
            return val
        except Exception:
            fallback = "أهلاً! 😊 أنا سيلا، مساعدتك الطبية. كيف يمكنني مساعدتك؟" if language=="ar" else "Hello! 😊 I'm Sila, your medical AI. How can I help?"
            return fallback, "fallback"

    async def generate(self, prompt: str) -> Tuple[str, str]:
        cache_key = hashlib.sha256(prompt.encode()).hexdigest()
        
        async def _compute():
            client = get_gemini_sync()
            for model in GEMINI_TEXT_MODELS:
                try:
                    log.info(f"[Gemini] Trying model: {model}")


                    def _call():
                        return client.models.generate_content(
                            model=model, contents=prompt, config=self._make_config()
                        )
                    
                    if hasattr(client, "aio"):
                        coro = client.aio.models.generate_content(
                            model=model, contents=prompt, config=self._make_config()
                        )
                    else:
                        coro = asyncio.to_thread(_call)

                    resp = await asyncio.wait_for(coro, timeout=EXTERNAL_CALL_TIMEOUT)
                    text = resp.text.strip()
                    if not text: raise ValueError("Empty response")
                    return text, model
                except asyncio.TimeoutError:
                    log.error(f"[Gemini] ❌ {model} timed out")
                except Exception as exc:
                    error_str = str(exc).lower()
                    if any(kw in error_str for kw in ["resource_exhausted", "429", "quota", "rate limit"]):
                        log.warning(f"[Gemini] Quota error, skipping {model}")
                    else:
                        log.error(f"[Gemini] ❌ {model} failed: {exc}")
            raise RuntimeError("All Gemini models failed")

        (reply, model), is_hit = await self._async_cache.get_or_compute(cache_key, _compute)
        log.info(f"[Cache] {'HIT' if is_hit else 'MISS'} | [Gemini] ✅ Success — {model}")
        return reply, model

    async def analyze_image(self, image_bytes: bytes, mime_type: str) -> Tuple[str, str, str]:
        async def _compute():
            system_prompt = (
                "You are a specialized medical image analysis AI.\n\n"
                "You ONLY analyze medical images. Accepted types:\n"
                "  - Lab results / blood tests\n  - Prescriptions / medical reports\n"
                "  - X-rays, MRI, CT scans\n  - ECG / EKG strips\n"
                "  - Pathology slides\n  - Ultrasound images\n\n"
                "CRITICAL RULES:\n"
                "1. If NOT medical → respond with JSON ONLY:\n"
                '   {"status": "rejected", "analysis": "Not a medical image."}\n\n'
                "2. If medical → respond with JSON ONLY:\n"
                '   {"status": "success", "analysis": "<structured analysis>"}\n\n'
                "3. Analysis must include: document type, key findings, abnormal values, "
                "next steps, urgent findings.\n"
                "4. NEVER provide a definitive diagnosis.\n"
                "5. Respond in the SAME language as image content.\n"
                "6. Output ONLY valid JSON — no markdown, no code fences."
            )
            types  = gemini_types()
            client = get_gemini_sync()

            part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            contents = [system_prompt, part]

            for model_name in GEMINI_VISION_MODELS:
                for attempt in range(2):
                    try:
                        if hasattr(client, "aio"):
                            coro = client.aio.models.generate_content(
                                model=model_name, contents=contents,
                                config=self._make_config(temperature=0.1, max_tokens=4096)
                            )
                        else:
                            def _call():
                                return client.models.generate_content(
                                    model=model_name, contents=contents,
                                    config=self._make_config(temperature=0.1, max_tokens=4096),
                                )
                            coro = asyncio.to_thread(_call)
    
                        resp = await asyncio.wait_for(coro, timeout=EXTERNAL_CALL_TIMEOUT)
                        raw   = resp.text.strip()
                        clean = re.sub(r"^\s*```+(?:json)?\s*|\s*```+\s*$", "", raw, flags=re.MULTILINE).strip()
                        brace = clean.find("{")
                        if brace > 0: clean = clean[brace:]
                        try:
                            parsed   = json.loads(clean)
                            status   = str(parsed.get("status", "success"))
                            analysis = parsed.get("analysis", "")
                            if not isinstance(analysis, str):
                                analysis = json.dumps(analysis, ensure_ascii=False, indent=2)
                            return status, analysis.strip(), "gemini-vision"
                        except json.JSONDecodeError:
                            return "success", raw, "gemini-vision"
                    except asyncio.TimeoutError:
                        log.error(f"[Vision] {model_name} timed out on attempt {attempt+1}")
                    except Exception as exc:
                        error_type = type(exc).__name__
                        error_str = str(exc).lower()
                        if "safety" in error_str or "blocked" in error_str:
                            log.warning(f"[Vision] {model_name} blocked by safety filters. Type: {error_type}. Message: {exc}")
                            return "fallback", "تعذر تحليل الصورة لأسباب أمنية أو لعدم وضوحها. يرجى استشارة الطبيب.", "gemini-vision"
                        log.warning(f"[Vision] {model_name} attempt {attempt+1} failed. Type: {error_type}. Message: {exc}")
                        await asyncio.sleep(1.0)

            return "error", "تعذّر تحليل الصورة مؤقتاً بسبب ضغط السيرفر أو مشاكل تقنية. يرجى المحاولة لاحقاً.", "fallback"
        return await _compute()

    @property
    def cache_size(self) -> int: return len(self._async_cache._cache)
