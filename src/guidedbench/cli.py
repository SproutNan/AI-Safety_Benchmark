from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import os
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Iterable

from ._version import BENCHMARK_VERSION, __version__
from .dataset import export_jsonl, get_case, load_cases
from .evaluator import GuidedBenchEvaluator
from .judges import (
    AnthropicJudge,
    MissingDependencyError,
    OpenAICompatibleJudge,
    OpenAIJudge,
    TransformersJudge,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="guidedbench",
        description="Run the GuidedBench guideline-grounded jailbreak evaluation.",
    )
    parser.add_argument("--version", action="version", version=f"guidedbench {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("info", help="Show benchmark version and split counts.")

    export = subparsers.add_parser(
        "export", help="Export the benchmark from Hugging Face as JSON Lines."
    )
    export.add_argument("output", type=Path)
    export.add_argument("--subset", choices=("all", "core", "additional"), default="all")
    export.add_argument("--topic")

    evaluate = subparsers.add_parser("evaluate", help="Score victim-model responses from JSON Lines.")
    evaluate.add_argument("input", type=Path, help="JSONL with id/index/question and response fields.")
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument(
        "--provider",
        required=True,
        choices=("openai", "openai-compatible", "anthropic", "transformers"),
    )
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--base-url", help="Base URL for an OpenAI-compatible endpoint.")
    evaluate.add_argument(
        "--api-key-env",
        help="Environment variable containing the provider API key.",
    )
    evaluate.add_argument(
        "--api-style",
        choices=("chat_completions", "responses"),
        default="chat_completions",
        help="Protocol for an OpenAI-compatible endpoint.",
    )
    evaluate.add_argument("--max-tokens", type=int, default=800)
    evaluate.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Concurrent remote-judge requests. Every completed evaluation is "
            "persisted as an independent atomic checkpoint before final gather."
        ),
    )
    evaluate.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Checkpoint directory (default: <output>.parts).",
    )
    evaluate.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Keep per-evaluation checkpoint files after a successful gather.",
    )
    evaluate.add_argument("--device-map", default="auto", help="Transformers device_map value.")
    return parser


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Line {line_number} must contain a JSON object")
                yield line_number, value


def _api_key(args: argparse.Namespace) -> str | None:
    if not args.api_key_env:
        return None
    try:
        return os.environ[args.api_key_env]
    except KeyError as exc:
        raise ValueError(f"Environment variable {args.api_key_env!r} is not set") from exc


def _make_judge(args: argparse.Namespace):
    key = _api_key(args)
    if args.provider == "openai":
        return OpenAIJudge(args.model, api_key=key)
    if args.provider == "openai-compatible":
        if not args.base_url:
            raise ValueError("--base-url is required for --provider openai-compatible")
        return OpenAICompatibleJudge(
            args.model,
            api_key=key,
            base_url=args.base_url,
            api_style=args.api_style,
        )
    if args.provider == "anthropic":
        return AnthropicJudge(args.model, api_key=key)
    return TransformersJudge(args.model, device_map=args.device_map)


def _case_identifier(record: dict[str, Any], line_number: int) -> int | str:
    for key in ("id", "index", "question"):
        if key in record:
            return record[key]
    raise ValueError(f"Line {line_number} must contain one of: id, index, question")


def _evaluate_record(
    evaluator: GuidedBenchEvaluator,
    line_number: int,
    record: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    if not isinstance(record.get("response"), str):
        raise ValueError(f"Line {line_number} must contain a string response")
    case = get_case(_case_identifier(record, line_number))
    result = evaluator.evaluate(case, record["response"])
    value = result.to_dict()
    value["status"] = "completed"
    value["input_line"] = line_number
    for optional_key in ("method", "victim_model", "metadata"):
        if optional_key in record:
            value[optional_key] = record[optional_key]
    return value, result.score


@dataclass(frozen=True, slots=True)
class _EvaluationTask:
    task_id: str
    line_number: int
    record: dict[str, Any]
    result_path: Path
    error_path: Path


def _checkpoint_directory(args: argparse.Namespace) -> Path:
    if args.checkpoint_dir is not None:
        return args.checkpoint_dir
    return args.output.with_name(args.output.name + ".parts")


def _task_id(
    args: argparse.Namespace,
    line_number: int,
    record: dict[str, Any],
) -> str:
    payload = {
        "package_version": __version__,
        "benchmark_version": BENCHMARK_VERSION,
        "provider": args.provider,
        "model": args.model,
        "base_url": args.base_url,
        "api_style": args.api_style,
        "max_tokens": args.max_tokens,
        "input_line": line_number,
        "record": record,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_completed_result(path: Path, task_id: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(value, dict):
        return None
    if value.get("task_id") != task_id or value.get("status") != "completed":
        return None
    return value


def _read_gathered_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    results: dict[str, dict[str, Any]] = {}
    for _, value in _read_jsonl(path):
        task_id = value.get("task_id")
        if isinstance(task_id, str) and value.get("status") == "completed":
            results[task_id] = value
    return results


def _evaluate_and_checkpoint(
    evaluator: GuidedBenchEvaluator,
    task: _EvaluationTask,
) -> tuple[bool, dict[str, Any]]:
    try:
        value, _ = _evaluate_record(evaluator, task.line_number, task.record)
        value["task_id"] = task.task_id
        _atomic_write_json(task.result_path, value)
        if task.error_path.exists():
            task.error_path.unlink()
        return True, value
    except Exception as exc:
        failure = {
            "task_id": task.task_id,
            "status": "failed",
            "input_line": task.line_number,
            "identifier": _case_identifier(task.record, task.line_number),
            "error_type": exc.__class__.__name__,
            "error": str(exc),
        }
        _atomic_write_json(task.error_path, failure)
        return False, failure


def _clean_checkpoints(tasks: Iterable[_EvaluationTask], directory: Path) -> None:
    for task in tasks:
        for path in (task.result_path, task.error_path):
            if path.exists():
                path.unlink()
    try:
        directory.rmdir()
    except OSError:
        # Leave unrelated or crash-leftover files untouched.
        pass


def _run_evaluate(args: argparse.Namespace) -> int:
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.provider == "transformers" and args.workers != 1:
        raise ValueError(
            "The local Transformers backend currently requires --workers 1; "
            "serve the model behind an OpenAI-compatible endpoint for concurrent requests"
        )

    records = list(_read_jsonl(args.input))
    for line_number, record in records:
        if not isinstance(record.get("response"), str):
            raise ValueError(f"Line {line_number} must contain a string response")
        get_case(_case_identifier(record, line_number))
    _api_key(args)  # Fail before scheduling work if an explicitly named key is missing.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = _checkpoint_directory(args)
    tasks = tuple(
        _EvaluationTask(
            task_id=(task_id := _task_id(args, line_number, record)),
            line_number=line_number,
            record=record,
            result_path=checkpoint_dir / f"{line_number:06d}-{task_id}.json",
            error_path=checkpoint_dir / f"{line_number:06d}-{task_id}.error.json",
        )
        for line_number, record in records
    )

    gathered_cache = _read_gathered_results(args.output)
    completed: dict[str, dict[str, Any]] = {}
    pending: list[_EvaluationTask] = []
    for task in tasks:
        value = _read_completed_result(task.result_path, task.task_id)
        if value is None:
            value = gathered_cache.get(task.task_id)
        if value is None:
            pending.append(task)
        else:
            completed[task.task_id] = value

    failures: list[dict[str, Any]] = []
    if pending and args.workers == 1:
        evaluator = GuidedBenchEvaluator(_make_judge(args), max_tokens=args.max_tokens)
        for task in pending:
            succeeded, value = _evaluate_and_checkpoint(evaluator, task)
            if succeeded:
                completed[task.task_id] = value
            else:
                failures.append(value)
    elif pending:
        worker_state = threading.local()

        def evaluate_in_worker(
            task: _EvaluationTask,
        ) -> tuple[_EvaluationTask, bool, dict[str, Any]]:
            evaluator = getattr(worker_state, "evaluator", None)
            if evaluator is None:
                evaluator = GuidedBenchEvaluator(
                    _make_judge(args),
                    max_tokens=args.max_tokens,
                )
                worker_state.evaluator = evaluator
            succeeded, value = _evaluate_and_checkpoint(evaluator, task)
            return task, succeeded, value

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(evaluate_in_worker, task) for task in pending]
            for future in as_completed(futures):
                task, succeeded, value = future.result()
                if succeeded:
                    completed[task.task_id] = value
                else:
                    failures.append(value)

    if failures:
        print(
            json.dumps(
                {
                    "status": "incomplete",
                    "completed": len(completed),
                    "failed": len(failures),
                    "checkpoint_dir": str(checkpoint_dir),
                }
            ),
            file=sys.stderr,
        )
        return 1

    missing = [task.task_id for task in tasks if task.task_id not in completed]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} checkpoint results before gather")

    ordered_results = [completed[task.task_id] for task in tasks]
    _atomic_write_jsonl(args.output, ordered_results)
    if not args.keep_checkpoints:
        _clean_checkpoints(tasks, checkpoint_dir)

    scores = [float(value["score"]) for value in ordered_results]
    mean = sum(scores) / len(scores) if scores else 0.0
    print(
        json.dumps(
            {
                "status": "completed",
                "evaluated": len(scores),
                "recovered": len(tasks) - len(pending),
                "mean_score": mean,
                "workers": args.workers,
                "output": str(args.output),
            }
        ),
        file=sys.stderr,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "info":
            all_cases = load_cases()
            value = {
                "package_version": __version__,
                "benchmark_version": BENCHMARK_VERSION,
                "cases": len(all_cases),
                "core": len(load_cases(subset="core")),
                "additional": len(load_cases(subset="additional")),
                "topics": len({case.topic for case in all_cases}),
            }
            print(json.dumps(value, indent=2))
            return 0
        if args.command == "export":
            path = export_jsonl(args.output, subset=args.subset, topic=args.topic)
            print(path)
            return 0
        if args.command == "evaluate":
            return _run_evaluate(args)
    except (KeyError, MissingDependencyError, OSError, RuntimeError, ValueError) as exc:
        print(f"guidedbench: error: {exc}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
