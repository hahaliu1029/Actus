"""Embedding provider protocol for skill semantic search."""
from __future__ import annotations
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """向量化文本的抽象接口。"""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """批量文本 → 向量列表。"""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """向量维度。"""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """模型标识符，用于缓存 key 构建。"""
        ...


class EmbeddingUnavailableError(Exception):
    """Embedding provider 不可用（circuit breaker open 或 embedding 未启用）。"""


class DisabledEmbeddingProvider(EmbeddingProvider):
    """Null object：embedding_enabled=False 时使用。

    所有 embed() 调用抛 EmbeddingUnavailableError，
    消费方统一 try/except 即可，无需额外判空分支。
    """

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise EmbeddingUnavailableError("memory embedding is disabled")

    @property
    def dimensions(self) -> int:
        return 0

    @property
    def model_name(self) -> str:
        return "disabled"
