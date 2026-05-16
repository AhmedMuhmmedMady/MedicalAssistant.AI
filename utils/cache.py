import asyncio
import collections
import pickle
from typing import Dict, Any, Callable, Coroutine, Tuple
from core.config import REDIS_URL
from core.logging import log

try:
    if REDIS_URL:
        import redis.asyncio as aioredis
        redis_client = aioredis.from_url(REDIS_URL)
        log.info("[Cache] ✅ Redis client configured for distributed caching.")
    else:
        redis_client = None
except ImportError:
    log.warning("[Cache] ⚠️ redis package not installed. Using in-memory fallback.")
    redis_client = None
except Exception as e:
    log.warning(f"[Cache] ⚠️ Redis initialization failed: {e}. Using in-memory fallback.")
    redis_client = None

class AsyncCache:
    """Thread-safe, cancel-safe AsyncCache to prevent duplicate calls and rate-limiting issues.
       Supports graceful fallback from Redis to Memory."""
    def __init__(self, maxsize: int = 200, ttl_seconds: int = 3600):
        self._cache: collections.OrderedDict = collections.OrderedDict()
        self._max = maxsize
        self._ttl = ttl_seconds
        self._lock = asyncio.Lock()
        self._pending: Dict[str, asyncio.Future] = {}

    async def get_or_compute(self, key: str, compute_coro_func: Callable[[], Coroutine[Any, Any, Any]]) -> Tuple[Any, bool]:
        """Returns (value, is_cache_hit)"""
        if redis_client:
            try:
                cached = await redis_client.get(key)
                if cached:
                    return pickle.loads(cached), True
            except Exception as e:
                log.warning(f"[Cache] Redis read failed for {key}: {e}")

        async with self._lock:
            if not redis_client and key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key], True
                
            if key in self._pending:
                fut = self._pending[key]
                compute_needed = False
            else:
                fut = asyncio.Future()
                self._pending[key] = fut
                compute_needed = True

        if not compute_needed:
            try:
                return await fut, True
            except BaseException:
                raise

        try:
            val = await compute_coro_func()
            
            if redis_client:
                try:
                    await redis_client.setex(key, self._ttl, pickle.dumps(val))
                except Exception as e:
                    log.warning(f"[Cache] Redis write failed for {key}: {e}")
                    
            async with self._lock:
                if not redis_client:
                    self._cache[key] = val
                    if len(self._cache) > self._max:
                        self._cache.popitem(last=False)
                self._pending.pop(key, None)
                
            if not fut.done(): fut.set_result(val)
            return val, False
            
        except BaseException as exc:
            async with self._lock:
                self._pending.pop(key, None)
            if not fut.done():
                fut.set_exception(exc)
                fut.exception()  # Prevent "Task exception was never retrieved" warning
            raise
