from __future__ import annotations

from .dataset import get_case
from .judges.base import Judge
from .models import EvaluationResult, GuidedBenchCase
from .parsing import parse_guided_response
from .prompt import JUDGE_SYSTEM_PROMPT, build_guided_prompt


class GuidedBenchEvaluator:
    """Evaluate responses using GuidedBench's verified case-specific guidelines."""

    def __init__(self, judge: Judge, *, max_tokens: int = 800) -> None:
        self.judge = judge
        self.max_tokens = max_tokens

    def evaluate(
        self,
        case: GuidedBenchCase | int | str,
        response: str,
    ) -> EvaluationResult:
        resolved_case = case if isinstance(case, GuidedBenchCase) else get_case(case)
        prompt = build_guided_prompt(resolved_case, response)
        raw = self.judge.generate(
            prompt,
            system=JUDGE_SYSTEM_PROMPT,
            max_tokens=self.max_tokens,
        )
        return parse_guided_response(raw, resolved_case, judge=self.judge.name)
