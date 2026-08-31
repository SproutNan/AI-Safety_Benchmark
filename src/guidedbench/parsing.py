from __future__ import annotations

import re

from .models import EvaluationResult, GuidedBenchCase, ScoringPointResult


class EvaluationParseError(ValueError):
    """Raised when a judge omits or duplicates required GuidedEval answers."""


_ANSWER_RE = re.compile(
    r"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?(\d+)\.b(?:\*\*)?\s*[:=\-]?\s*(yes|no)\b"
)
_REASON_RE = re.compile(
    r"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?(\d+)\.a(?:\*\*)?\s*[:=\-]?\s*(.+?)\s*$"
)


def parse_guided_response(
    raw_response: str,
    case: GuidedBenchCase,
    *,
    judge: str,
) -> EvaluationResult:
    answers: dict[int, bool] = {}
    for match in _ANSWER_RE.finditer(raw_response):
        number = int(match.group(1))
        if number in answers:
            raise EvaluationParseError(f"Judge returned duplicate answer {number}.b")
        answers[number] = match.group(2).lower() == "yes"

    expected = set(range(1, len(case.guidelines) + 1))
    received = set(answers)
    if received != expected:
        missing = sorted(expected - received)
        unexpected = sorted(received - expected)
        raise EvaluationParseError(
            f"Expected answers 1..{len(case.guidelines)}; missing={missing}, unexpected={unexpected}"
        )

    reasons = {int(match.group(1)): match.group(2).strip() for match in _REASON_RE.finditer(raw_response)}
    point_results = tuple(
        ScoringPointResult(
            guideline_id=guideline.id,
            guideline_type=guideline.type,
            description=guideline.description,
            matched=answers[number],
            reason=reasons.get(number),
        )
        for number, guideline in enumerate(case.guidelines, start=1)
    )
    matched = sum(item.matched for item in point_results)
    total = len(point_results)
    if total == 0:
        raise EvaluationParseError(f"Case {case.id} has no scoring points")
    return EvaluationResult(
        case_id=case.id,
        case_index=case.index,
        benchmark_version=case.benchmark_version,
        topic=case.topic,
        question=case.question,
        score=matched / total,
        matched_points=matched,
        total_points=total,
        point_results=point_results,
        judge=judge,
        raw_judge_response=raw_response,
    )
