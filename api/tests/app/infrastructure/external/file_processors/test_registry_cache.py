"""B12 P3: FileProcessorRegistry session-scoped LRU cache."""
import time
from unittest.mock import AsyncMock

from app.domain.external.file_processor import FileProcessResult
from app.infrastructure.external.file_processors.registry import FileProcessorRegistry


def _reg() -> FileProcessorRegistry:
    return FileProcessorRegistry(sandbox=AsyncMock(), file_uploader=AsyncMock())


def test_cache_put_get_hit() -> None:
    reg = _reg()
    r = FileProcessResult(text="x")
    reg.cache_put(("k",), r, time.time())
    assert reg.cache_get(("k",)) is r


def test_cache_miss_returns_none() -> None:
    assert _reg().cache_get(("nope",)) is None


def test_cache_expired_returns_none() -> None:
    reg = _reg()
    reg.cache_put(("k",), FileProcessResult(text="x"), time.time() - reg._PRESIGNED_TTL)
    assert reg.cache_get(("k",)) is None


def test_cache_lru_evicts_oldest() -> None:
    reg = _reg()
    now = time.time()
    for i in range(reg._CACHE_CAP + 5):
        reg.cache_put((i,), FileProcessResult(text=str(i)), now)
    assert reg.cache_get((0,)) is None
    assert reg.cache_get((reg._CACHE_CAP + 4,)) is not None
    assert len(reg._result_cache) <= reg._CACHE_CAP
