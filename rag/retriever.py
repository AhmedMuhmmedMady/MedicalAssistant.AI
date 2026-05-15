import asyncio
import threading
import time
from typing import Any, List
from core.config import PINECONE_API_KEY, INDEX_NAME, EMBED_MODEL, EMBED_DIM
from core.logging import log

_pinecone_index = None
_pinecone_lock  = threading.Lock()

def _init_pinecone():
    global _pinecone_index
    if _pinecone_index is None:
        with _pinecone_lock:
            if _pinecone_index is None:
                log.info("🔄 Initialising Pinecone v3…")
                from pinecone import Pinecone as _PC
                _pinecone_index = _PC(api_key=PINECONE_API_KEY).Index(INDEX_NAME)
                log.info(f"✅ Pinecone ready — index: {INDEX_NAME}")
    return _pinecone_index

async def get_index():
    if _pinecone_index is None:
        await asyncio.to_thread(_init_pinecone)
    return _pinecone_index

_st_model: Any = None
_st_lock        = threading.Lock()

def _load_and_encode_sync(text: str) -> List[float]:
    global _st_model
    if _st_model is None:
        with _st_lock:
            if _st_model is None:
                try:
                    log.info(f"🔄 Lazy-loading SentenceTransformer: {EMBED_MODEL}")
                    t0 = time.perf_counter()
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer(EMBED_MODEL, device="cpu")
                    try:
                        import torch as _torch
                        _torch.set_num_threads(1)
                        _torch.set_num_interop_threads(1)
                        _torch.set_grad_enabled(False)
                    except ImportError:
                        pass
                    elapsed = time.perf_counter() - t0
                    log.info(f"✅ SentenceTransformer ready in {elapsed:.2f}s")
                    _st_model = model
                except Exception as exc:
                    log.error(f"❌ Failed to load SentenceTransformer: {exc}")
                    raise RuntimeError(f"Embedder failed to load: {exc}")
    try:
        vec: List[float] = _st_model.encode(
            text, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False,
        ).tolist()
        if len(vec) != EMBED_DIM:
            raise RuntimeError(f"❌ Embedding dim mismatch: got {len(vec)}, expected {EMBED_DIM}.")
        return vec
    except Exception as exc:
        log.error(f"❌ Encoding failed: {exc}")
        raise

async def encode_async(text: str) -> List[float]:
    try:
        return await asyncio.to_thread(_load_and_encode_sync, text)
    except Exception as exc:
        raise RuntimeError(f"❌ Embedding unavailable: {exc}") from exc
