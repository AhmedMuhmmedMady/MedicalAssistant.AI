import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.config import MAX_CONCURRENT_REQUESTS, EMBED_MODEL, EMBED_DIM, MIN_CONFIDENCE, WEIGHT_EXACT, WEIGHT_CATEGORY, EXACT_MATCH_THRESHOLD, INDEX_NAME
from core.logging import log
from utils.concurrency import init_semaphore
from services.knowledge_base import KnowledgeBaseService
from services.gemini_service import GeminiService
from rag.prompt_builder import PromptBuilder

from api.ask import router as ask_router
from api.image import router as image_router
from api.health import router as health_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("🚀 MADY v17.4 (Modular) starting…")
    init_semaphore(MAX_CONCURRENT_REQUESTS)
    
    app.state.knowledge_base = KnowledgeBaseService()
    gemini_service           = GeminiService()
    app.state.gemini         = gemini_service
    
    from services.model_router import ModelRouter
    app.state.model_router   = ModelRouter(gemini_service)
    
    app.state.prompt_builder = PromptBuilder()
    
    log.info(
        f"✅ Boot complete — embed={EMBED_MODEL} dim={EMBED_DIM} "
        f"min_confidence={MIN_CONFIDENCE} hybrid_weights=(cosine+exact*{WEIGHT_EXACT}+cat*{WEIGHT_CATEGORY}) "
        f"exact_threshold={EXACT_MATCH_THRESHOLD} index={INDEX_NAME}"
    )
    yield
    log.info("🛑 MADY shutting down")
    await app.state.model_router.close()

app = FastAPI(
    title="Mady — Medical AI Assistant Modular",
    description="Backend API for Mady Health AI with Hybrid RAG & Model Routing.",
    version="17.4.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.error(f"❌ {type(exc).__name__} on {request.url.path}: {exc}")
    return JSONResponse(status_code=500, content={"error": "An unexpected error occurred."})

@app.get("/")
def root():
    return {"name":"Mady — Medical AI","version":"17.4.0","status":"running ✅",
            "endpoints":["/ask","/analyze-image","/health","/docs"]}

app.include_router(health_router)
app.include_router(ask_router)
app.include_router(image_router)
