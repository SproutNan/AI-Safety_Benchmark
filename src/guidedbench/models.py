from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, cast


GuidelineType = Literal["entity", "action"]


@dataclass(frozen=True, slots=True)
class Guideline:
    """One case-specific scoring point in GuidedBench."""

    id: str
    type: GuidelineType
    description: str
    examples: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Guideline":
        guideline_id = str(value["id"])
        guideline_type = str(value["type"])

        if guideline_type not in {"entity", "action"}:
            raise ValueError(f"Unknown guideline type: {guideline_type!r}")
        if not guideline_id.startswith(f"{guideline_type}-"):
            raise ValueError(
                f"Guideline id {guideline_id!r} does not match type {guideline_type!r}"
            )

        return cls(
            id=guideline_id,
            type=cast(GuidelineType, guideline_type),
            description=str(value["description"]),
            examples=str(value["examples"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GuidedBenchCase:
    """A benchmark question and its verified, case-specific guidelines."""

    id: str
    index: int
    benchmark_version: str
    subset: Literal["core", "additional"]
    topic: str
    question: str
    guidelines: tuple[Guideline, ...]
    target: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GuidedBenchCase":
        return cls(
            id=str(value["id"]),
            index=int(value["index"]),
            benchmark_version=str(value["benchmark_version"]),
            subset=value["subset"],
            topic=str(value["topic"]),
            question=str(value["question"]),
            guidelines=tuple(Guideline.from_dict(item) for item in value["guidelines"]),
            target=str(value["target"]),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["guidelines"] = [item.to_dict() for item in self.guidelines]
        return value


@dataclass(frozen=True, slots=True)
class ScoringPointResult:
    guideline_id: str
    guideline_type: GuidelineType
    description: str
    matched: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    case_id: str
    case_index: int
    benchmark_version: str
    topic: str
    question: str
    score: float
    matched_points: int
    total_points: int
    point_results: tuple[ScoringPointResult, ...]
    judge: str
    raw_judge_response: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["point_results"] = [item.to_dict() for item in self.point_results]
        return value
