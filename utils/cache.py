import asyncio
import collections
from typing import Dict, Any, Callable, Coroutine, Tuple

class AsyncCache:
    """Thread-safe, cancel-safe AsyncCache to prevent duplicate calls and rate-limiting issues."""
    def __init__(self, maxsize: int = 200):
        self._cache: collections.OrderedDict = collections.OrderedDict()
        self._max = maxsize
        self._lock = asyncio.Lock()
        self._pending: Dict[str, asyncio.Future] = {}

    async def get_or_compute(self, key: str, compute_coro_func: Callable[[], Coroutine[Any, Any, Any]]) -> Tuple[Any, bool]:
        """Returns (value, is_cache_hit)"""
        async with self._lock:
            if key in self._cache:
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
            async with self._lock:
                self._cache[key] = val
                if len(self._cache) > self._max:
                    self._cache.popitem(last=False)
                self._pending.pop(key, None)
            if not fut.done(): fut.set_result(val)
            return val, False
        except BaseException as exc:
            async with self._lock:
                self._pending.pop(key, None)
            if not fut.done(): fut.set_exception(exc)
            raise
