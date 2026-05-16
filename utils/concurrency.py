import asyncio
import threading
import time
from typing import Dict, List, Optional
from core.config import RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW

_request_semaphore: Optional[asyncio.Semaphore] = None
_rate_limit_store: Dict[str, List[float]] = {}
_rate_limit_lock  = threading.Lock()

def get_semaphore() -> asyncio.Semaphore:
    global _request_semaphore
    if _request_semaphore is None:
        raise RuntimeError("Semaphore not initialized. App lifespan must run first.")
    return _request_semaphore

def init_semaphore(max_concurrent: int):
    global _request_semaphore
    _request_semaphore = asyncio.Semaphore(max_concurrent)

async def check_rate_limit(ip: str) -> bool:
    from utils.cache import redis_client
    now = time.time()
    
    if redis_client:
        try:
            key = f"rate_limit:{ip}"
            async with redis_client.pipeline(transaction=True) as pipe:
                pipe.zremrangebyscore(key, 0, now - RATE_LIMIT_WINDOW)
                pipe.zadd(key, {str(now): now})
                pipe.zcard(key)
                pipe.expire(key, RATE_LIMIT_WINDOW)
                _, _, req_count, _ = await pipe.execute()
            return req_count <= RATE_LIMIT_REQUESTS
        except Exception:
            pass # fallback to in-memory if Redis fails

    with _rate_limit_lock:
        if ip not in _rate_limit_store:
            _rate_limit_store[ip] = []
        _rate_limit_store[ip] = [ts for ts in _rate_limit_store[ip] if now - ts < RATE_LIMIT_WINDOW]
        if len(_rate_limit_store[ip]) < RATE_LIMIT_REQUESTS:
            _rate_limit_store[ip].append(now)
            return True
    return False
