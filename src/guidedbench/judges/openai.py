from __future__ import annotations

from typing import Any, Literal

from .base import MissingDependencyError


class OpenAICompatibleJudge:
    """Judge through an OpenAI-compatible endpoint.

    ``chat_completions`` is the broadly compatible default. Use ``responses``
    for the native OpenAI Responses API or a server that implements it.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        api_style: Literal["chat_completions", "responses"] = "chat_completions",
        temperature: float | None = None,
        timeout: float = 120.0,
        client: Any | None = None,
    ) -> None:
        if api_style not in {"chat_completions", "responses"}:
            raise ValueError("api_style must be 'chat_completions' or 'responses'")
        self.model = model
        self.api_style = api_style
        self.temperature = temperature
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise MissingDependencyError(
                    "Install the OpenAI backend with: pip install 'guidedbench[openai]'"
                ) from exc
            kwargs: dict[str, Any] = {"timeout": timeout}
            if api_key is not None:
                kwargs["api_key"] = api_key
            if base_url is not None:
                kwargs["base_url"] = base_url.rstrip("/")
            client = OpenAI(**kwargs)
        self.client = client

    @property
    def name(self) -> str:
        return f"openai-compatible:{self.model}:{self.api_style}"

    def generate(self, prompt: str, *, system: str | None, max_tokens: int) -> str:
        if self.api_style == "responses":
            request: dict[str, Any] = {
                "model": self.model,
                "input": prompt,
                "max_output_tokens": max_tokens,
            }
            if system:
                request["instructions"] = system
            if self.temperature is not None:
                request["temperature"] = self.temperature
            response = self.client.responses.create(**request)
            text = getattr(response, "output_text", None)
            if not text:
                raise RuntimeError("Responses API returned no output_text")
            return str(text)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        request = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        response = self.client.chat.completions.create(**request)
        text = response.choices[0].message.content
        if not text:
            raise RuntimeError("Chat Completions API returned no message content")
        return str(text)


class OpenAIJudge(OpenAICompatibleJudge):
    """Native OpenAI judge using the Responses API."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        temperature: float | None = None,
        timeout: float = 120.0,
        client: Any | None = None,
    ) -> None:
        super().__init__(
            model,
            api_key=api_key,
            api_style="responses",
            temperature=temperature,
            timeout=timeout,
            client=client,
        )

    @property
    def name(self) -> str:
        return f"openai:{self.model}:responses"
