from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError

from ._version import BENCHMARK_VERSION
from .models import GuidedBenchCase


Subset = Literal["all", "core", "additional"]

DATASET_REPO_ID = "HRXUST/GuidedBench"
DATASET_REVISION = "10cc683c8df8c07139d3f900e381fb52738123c3"
_DATA_FILES = (
    ("core", "core.jsonl", 180),
    ("additional", "additional.jsonl", 20),
)


@lru_cache(maxsize=8)
def _all_cases(
    repo_id: str,
    revision: str,
    cache_dir: str | None,
    local_files_only: bool,
) -> tuple[GuidedBenchCase, ...]:
    cases: list[GuidedBenchCase] = []
    try:
        for expected_subset, filename, expected_count in _DATA_FILES:
            data_file = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            subset_cases: list[GuidedBenchCase] = []
            with Path(data_file).open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        case = GuidedBenchCase.from_dict(json.loads(line))
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                        raise RuntimeError(
                            f"Invalid GuidedBench record in {filename}:{line_number}"
                        ) from exc
                    if case.subset != expected_subset:
                        raise RuntimeError(
                            f"Expected subset {expected_subset!r} in {filename}, "
                            f"found {case.subset!r}"
                        )
                    subset_cases.append(case)
            if len(subset_cases) != expected_count:
                raise RuntimeError(
                    f"Expected {expected_count} cases in {filename}, "
                    f"found {len(subset_cases)}"
                )
            cases.extend(subset_cases)
    except GatedRepoError as exc:
        raise RuntimeError(
            "GuidedBench dataset access is gated. Request access at "
            f"https://huggingface.co/datasets/{repo_id}, then authenticate with "
            "`hf auth login` or set HF_TOKEN."
        ) from exc

    if len(cases) != 200:
        raise RuntimeError(f"Expected 200 GuidedBench cases, found {len(cases)}")
    if len({case.id for case in cases}) != len(cases):
        raise RuntimeError("GuidedBench contains duplicate case ids")
    if any(case.benchmark_version != BENCHMARK_VERSION for case in cases):
        raise RuntimeError(
            f"Dataset revision {revision} does not match benchmark version "
            f"{BENCHMARK_VERSION}"
        )
    return tuple(cases)


def _load_source(
    *,
    repo_id: str,
    revision: str,
    cache_dir: str | Path | None,
    local_files_only: bool,
) -> tuple[GuidedBenchCase, ...]:
    normalized_cache = (
        str(Path(cache_dir).expanduser()) if cache_dir is not None else None
    )
    return _all_cases(repo_id, revision, normalized_cache, local_files_only)


def load_cases(
    *,
    subset: Subset = "all",
    topic: str | None = None,
    repo_id: str = DATASET_REPO_ID,
    revision: str = DATASET_REVISION,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> tuple[GuidedBenchCase, ...]:
    """Load a pinned GuidedBench release from the Hugging Face Hub cache."""

    if subset not in {"all", "core", "additional"}:
        raise ValueError("subset must be one of: all, core, additional")
    cases = _load_source(
        repo_id=repo_id,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    if subset != "all":
        cases = tuple(case for case in cases if case.subset == subset)
    if topic is not None:
        cases = tuple(case for case in cases if case.topic == topic)
    return cases


def get_case(
    identifier: int | str,
    *,
    repo_id: str = DATASET_REPO_ID,
    revision: str = DATASET_REVISION,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> GuidedBenchCase:
    """Find a case by integer index, stable id, or exact question text."""

    cases = _load_source(
        repo_id=repo_id,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    for case in cases:
        if isinstance(identifier, int) and case.index == identifier:
            return case
        if isinstance(identifier, str) and identifier in {case.id, case.question}:
            return case
    raise KeyError(f"No GuidedBench case matches {identifier!r}")


def export_jsonl(
    path: str | Path,
    *,
    subset: Subset = "all",
    topic: str | None = None,
    repo_id: str = DATASET_REPO_ID,
    revision: str = DATASET_REVISION,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> Path:
    """Export records downloaded from Hugging Face as JSON Lines."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for case in load_cases(
            subset=subset,
            topic=topic,
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        ):
            handle.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")
    return destination
