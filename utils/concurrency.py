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

def check_rate_limit(ip: str) -> bool:
    now = time.time()
    with _rate_limit_lock:
        if ip not in _rate_limit_store:
            _rate_limit_store[ip] = []
        _rate_limit_store[ip] = [ts for ts in _rate_limit_store[ip] if now - ts < RATE_LIMIT_WINDOW]
        if len(_rate_limit_store[ip]) < RATE_LIMIT_REQUESTS:
            _rate_limit_store[ip].append(now)
            return True
    return False
