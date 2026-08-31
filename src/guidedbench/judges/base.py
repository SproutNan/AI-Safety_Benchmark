from __future__ import annotations

from typing import Protocol, runtime_checkable


class MissingDependencyError(ImportError):
    """Raised when an optional judge backend has not been installed."""


@runtime_checkable
class Judge(Protocol):
    @property
    def name(self) -> str: ...

    def generate(self, prompt: str, *, system: str | None, max_tokens: int) -> str: ...
