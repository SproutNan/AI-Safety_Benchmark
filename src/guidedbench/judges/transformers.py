from __future__ import annotations

from typing import Any, Callable

from .base import MissingDependencyError


class TransformersJudge:
    """Run GuidedEval locally with a Hugging Face chat model."""

    def __init__(
        self,
        model: str | None = None,
        *,
        pipeline: Callable[..., Any] | None = None,
        device_map: str | None = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = False,
    ) -> None:
        if pipeline is None and model is None:
            raise ValueError("model is required when pipeline is not provided")
        model_label: Any = model or getattr(pipeline, "model", "custom-pipeline")
        if not isinstance(model_label, str):
            config = getattr(model_label, "config", None)
            model_label = getattr(config, "_name_or_path", model_label.__class__.__name__)
        self.model = str(model_label)
        if pipeline is None:
            try:
                from transformers import pipeline as make_pipeline
            except ImportError as exc:
                raise MissingDependencyError(
                    "Install the local backend with: pip install 'guidedbench[local]'"
                ) from exc
            kwargs: dict[str, Any] = {
                "task": "text-generation",
                "model": model,
                "dtype": dtype,
                "trust_remote_code": trust_remote_code,
            }
            if device_map is not None:
                kwargs["device_map"] = device_map
            pipeline = make_pipeline(**kwargs)
        self.pipeline = pipeline

    @property
    def name(self) -> str:
        return f"transformers:{self.model}"

    def generate(self, prompt: str, *, system: str | None, max_tokens: int) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        output = self.pipeline(
            messages,
            max_new_tokens=max_tokens,
            do_sample=False,
            return_full_text=True,
        )
        if not output:
            raise RuntimeError("Transformers pipeline returned no generations")
        generated = output[0].get("generated_text")
        if isinstance(generated, list) and generated:
            last = generated[-1]
            if isinstance(last, dict) and last.get("content"):
                return str(last["content"])
        if isinstance(generated, str) and generated:
            return generated
        raise RuntimeError("Could not extract text from Transformers pipeline output")
