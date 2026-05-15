from fastapi import APIRouter
from core.config import EMBED_MODEL, EMBED_DIM, INDEX_NAME, MIN_CONFIDENCE, TOP_K, MAX_CONTEXT_MATCHES, EXACT_MATCH_THRESHOLD

router = APIRouter()

@router.get("/health")
def health():
    return {"status":"ok","version":"17.4.0","config":{
        "embed_model":EMBED_MODEL,"embed_dim":EMBED_DIM,"index":INDEX_NAME,
        "min_confidence":MIN_CONFIDENCE,"top_k":TOP_K,"max_context_matches":MAX_CONTEXT_MATCHES,
        "exact_match_threshold":EXACT_MATCH_THRESHOLD,
        "garbage_protection":"enabled","emergency_detection":"enabled",
        "rag_first":"enabled","deterministic_fallback":"enabled",
    }}
