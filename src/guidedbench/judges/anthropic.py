from __future__ import annotations

from typing import Any

from .base import MissingDependencyError


class AnthropicJudge:
    """Judge through Anthropic's native Messages API."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = 120.0,
        client: Any | None = None,
    ) -> None:
        self.model = model
        if client is None:
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise MissingDependencyError(
                    "Install the Anthropic backend with: pip install 'guidedbench[anthropic]'"
                ) from exc
            kwargs: dict[str, Any] = {"timeout": timeout}
            if api_key is not None:
                kwargs["api_key"] = api_key
            client = Anthropic(**kwargs)
        self.client = client

    @property
    def name(self) -> str:
        return f"anthropic:{self.model}:messages"

    def generate(self, prompt: str, *, system: str | None, max_tokens: int) -> str:
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            request["system"] = system
        message = self.client.messages.create(**request)
        blocks = getattr(message, "content", ())
        text_parts = []
        for block in blocks:
            block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
            if block_type == "text":
                text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
                if text:
                    text_parts.append(str(text))
        if not text_parts:
            raise RuntimeError("Anthropic Messages API returned no text content")
        return "\n".join(text_parts)
