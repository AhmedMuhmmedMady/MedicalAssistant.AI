import asyncio
import time
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from core.config import TOP_K, MIN_CONFIDENCE, MAX_CONTEXT_MATCHES, EXTERNAL_CALL_TIMEOUT
from core.constants import MEDICAL_DISCLAIMER
from core.logging import log
from models.schemas import AskRequest, AskResponse, MatchResult, QueryContext
from engine.decision_engine import IntentClassifier, generate_deterministic_fallback, determine_rag_mode
from services.knowledge_base import KnowledgeBaseService, _extract_expected_categories
from services.gemini_service import GeminiService
from rag.prompt_builder import PromptBuilder
from utils.text import LanguageDetector
from utils.concurrency import get_semaphore, check_rate_limit
from core.constants import EMERGENCY_KEYWORDS_AR, EMERGENCY_KEYWORDS_EN

router = APIRouter()

def is_emergency(query: str, language: str) -> bool:
    q = query.lower()
    kws = EMERGENCY_KEYWORDS_AR if language == "ar" else EMERGENCY_KEYWORDS_EN
    return any(kw in q for kw in kws)

@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, request: Request) -> AskResponse:
    try:
        semaphore = get_semaphore()
    except RuntimeError:
        return JSONResponse(status_code=503, content={"error":"Server not ready"})

    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(client_ip):
        lang = LanguageDetector.detect(req.query)
        msg  = ("لقد تجاوزت الحد المسموح من الطلبات." if lang=="ar" else "Rate limit exceeded. Please try again in a minute.")
        return JSONResponse(status_code=429, content={"error": msg})

    await semaphore.acquire()
    try:
        return await asyncio.wait_for(_ask_inner(req, request), timeout=EXTERNAL_CALL_TIMEOUT)
    except asyncio.TimeoutError:
        lang = LanguageDetector.detect(req.query)
        msg  = ("عذراً، استغرق الطلب وقتاً أطول من المتوقع." if lang=="ar" else "Request timed out. Please try again.")
        return AskResponse(
            query=req.query, reply=msg, model_used="none", matches=[],
            is_medical=True, found_in_database=False, low_confidence=True,
            language=lang, disclaimer=MEDICAL_DISCLAIMER,
        )
    except BaseException as exc:
        log.error(f"[ASK] Unhandled failure in outer ask layer: {exc}")
        lang = LanguageDetector.detect(req.query)
        reply, model = generate_deterministic_fallback(req.query, lang), "deterministic-fallback"
        return AskResponse(
            query=req.query, reply=f"{reply}\n\n{MEDICAL_DISCLAIMER}", model_used=model, matches=[],
            is_medical=True, found_in_database=False, low_confidence=True,
            language=lang, disclaimer=MEDICAL_DISCLAIMER,
        )
    finally:
        try: semaphore.release()
        except Exception: pass

async def _ask_inner(req: AskRequest, request: Request) -> AskResponse:
    kb_service: KnowledgeBaseService = request.app.state.knowledge_base
    gemini_service: GeminiService = request.app.state.gemini
    prompt_builder: PromptBuilder = request.app.state.prompt_builder

    start = time.time()
    q     = req.query
    lang  = LanguageDetector.detect(q)
    log.info(f"[ASK] Request started | query='{q[:80]}' | lang={lang}")

    is_emg = is_emergency(q, lang)
    if is_emg:
        log.warning("[ASK] 🚨 EMERGENCY DETECTED INSTANTLY")
        intent = "medical"
    else:
        intent = await IntentClassifier.classify(q)

    log.info(f"[ASK] Intent classified as: {intent}")

    if intent == "greeting":
        reply, model = await gemini_service.reply_social(q, lang)
        log.info(f"[ASK] ✅ Greeting — {model} | {round((time.time()-start)*1000)}ms")
        return AskResponse(query=q, reply=reply, model_used=model, matches=[],
                           is_medical=False, found_in_database=False, low_confidence=False,
                           language=lang, disclaimer="")

    if intent == "non_medical":
        reply = "أنا مساعد طبي متخصص 🏥\nأقدر أساعدك فقط في الأسئلة والاستفسارات الطبية." if lang == "ar" else "I am a specialized medical assistant. I can only help with medical-related questions."
        log.info(f"[ASK] ❌ Non-Medical Rejection | {round((time.time()-start)*1000)}ms")
        return AskResponse(query=q, reply=reply, model_used="none", matches=[],
                           is_medical=False, found_in_database=False, low_confidence=False,
                           language=lang, disclaimer="")

    try:
        raw_matches = await kb_service.search(q, top_k=TOP_K)
    except Exception as exc:
        log.error(f"[ASK] KB search failed: {exc}")
        raw_matches = []

    sanitized = kb_service._deduplicate_and_sanitize(raw_matches)
    garbage_r = kb_service._calculate_garbage_ratio(sanitized)
    selected  = kb_service._select_top_matches(sanitized, MAX_CONTEXT_MATCHES)

    top_score = selected[0].confidence if selected else 0.0
    cat_cons  = kb_service._category_consistency(selected)
    
    expected_cats = _extract_expected_categories(q)
    rel_ok = kb_service.relevance_ok(q, selected, expected_cats)

    rag_mode, reason = determine_rag_mode(is_emg, selected, top_score, rel_ok, garbage_r, cat_cons)

    log.info(f"[RAG Path] Mode: {rag_mode} | Reason: {reason} | Score: {top_score:.4f} | Garbage Ratio: {garbage_r:.2f} | Rel: {rel_ok}")

    match_results = [MatchResult(question=m.question, answer=m.answer, confidence=m.confidence, category=m.category) for m in selected]

    def _build_resp(reply: str, model: str, found: bool, low_conf: bool) -> AskResponse:
        if MEDICAL_DISCLAIMER not in reply:
            reply = f"{reply}\n\n{MEDICAL_DISCLAIMER}"
        return AskResponse(
            query=q, reply=reply, model_used=model, matches=match_results,
            is_medical=True, found_in_database=found, low_confidence=low_conf,
            language=lang, disclaimer=MEDICAL_DISCLAIMER
        )

    try:
        if rag_mode == "EMERGENCY_OVERRIDE":
            try:
                reply, model = await gemini_service.generate(prompt_builder.build_emergency(q, lang))
                log.info(f"[Model] ✅ Emergency response via {model}")
            except Exception as exc:
                log.warning(f"[Model] Emergency generation failed: {exc} - fallback triggered")
                reply, model = generate_deterministic_fallback(q, lang), "deterministic-fallback"
            return _build_resp(reply, model, False, False)

        elif rag_mode == "RAG_STRONG":
            if selected:
                top_match = selected[0]
                source_txt = "\n\n(المصدر: قاعدة المعرفة الطبية)" if lang == "ar" else "\n\n(Source: Medical Knowledge Base)"
                reply = f"{top_match.answer}{source_txt}"
                model = "rag-only"
                log.info(f"[Model] ✅ RAG-ONLY response - confidence={top_score:.4f}")
                return _build_resp(reply, model, True, False)
            else:
                ctx = QueryContext(raw_query=q, language=lang, matches=selected)
                reply, model = await gemini_service.generate(prompt_builder.build(ctx))
                return _build_resp(reply, model, True, False)

        elif rag_mode == "RAG_LIGHT":
            ctx = QueryContext(raw_query=q, language=lang, matches=selected)
            reply, model = await gemini_service.generate(prompt_builder.build_rag_light(ctx))
            return _build_resp(reply, model, True, True)

        else: # GEMINI_ONLY
            reply, model = await gemini_service.generate(prompt_builder.build_gemini_only(q, lang))
            return _build_resp(reply, model, False, True)

    except Exception as exc:
        log.error(f"[Model] Unhandled generation failure: {exc} - using deterministic fallback")
        reply, model = generate_deterministic_fallback(q, lang), "deterministic-fallback"
        return _build_resp(reply, model, False, True)
