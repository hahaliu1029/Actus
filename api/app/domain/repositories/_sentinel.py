"""Sentinel singleton for patch-style repository updates."""

from __future__ import annotations


class UnsetType:
    """Singleton sentinel distinct from None."""

    _instance: "UnsetType | None" = None

    def __new__(cls) -> "UnsetType":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<UNSET>"

    def __bool__(self) -> bool:
        return False


_UNSET = UnsetType()

__all__ = ["_UNSET", "UnsetType"]
