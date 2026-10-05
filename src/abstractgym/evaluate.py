"""Evaluation harness for Gym-generated CFG trace instances."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
import json
from typing import Callable, Iterable

from abstractgym.cfg import Grammar
from abstractgym.oracle import accepts
from abstractgym.trace import check_compact_trace, extract_cot_and_answer, trace_prefix_match


@dataclass(frozen=True)
class RecordScore:
    """Per-prediction scores; aggregated into EvaluationResult."""

    task_id: str
    format_valid: bool
    verdict_correct: bool
    trace_valid: bool
    trace_exact_match: bool
    verdict_trace_consistent: bool
    oracle_trace_agreement: bool
    prefix_match_len: int
    target_trace_len: int
    trace_failure: str | None
    first_invalid_step: int | None
    trace_message: str

    @property
    def verdict_correct_trace_invalid(self) -> bool:
        return self.verdict_correct and not self.trace_valid

    @property
    def prefix_match_ratio(self) -> float:
        return _rate(self.prefix_match_len, self.target_trace_len)

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["verdict_correct_trace_invalid"] = self.verdict_correct_trace_invalid
        data["prefix_match_ratio"] = self.prefix_match_ratio
        return data


@dataclass(frozen=True)
class EvaluationResult:
    total: int
    verdict_correct: int
    trace_valid: int
    trace_exact_match: int
    format_valid: int
    oracle_trace_agreement: int
    verdict_trace_consistent: int = 0
    verdict_correct_trace_invalid: int = 0
    prefix_match_ratio_sum: float = 0.0
    trace_failures: tuple[tuple[str, int], ...] = ()
    buckets: tuple[tuple[str, tuple[tuple[str, "EvaluationResult"], ...]], ...] = ()
    records: tuple[RecordScore, ...] = ()

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "total": self.total,
            "verdict_accuracy": _rate(self.verdict_correct, self.total),
            "trace_validity": _rate(self.trace_valid, self.total),
            "trace_exact_match": _rate(self.trace_exact_match, self.total),
            "format_validity": _rate(self.format_valid, self.total),
            "oracle_trace_agreement": _rate(self.oracle_trace_agreement, self.total),
            "verdict_trace_consistency": _rate(self.verdict_trace_consistent, self.total),
            "verdict_correct_trace_invalid_rate": _rate(self.verdict_correct_trace_invalid, self.total),
            "mean_prefix_match_ratio": self.prefix_match_ratio_sum / self.total if self.total else 0.0,
            "counts": {
                "verdict_correct": self.verdict_correct,
                "trace_valid": self.trace_valid,
                "trace_exact_match": self.trace_exact_match,
                "format_valid": self.format_valid,
                "oracle_trace_agreement": self.oracle_trace_agreement,
                "verdict_trace_consistent": self.verdict_trace_consistent,
                "verdict_correct_trace_invalid": self.verdict_correct_trace_invalid,
            },
            "trace_failures": dict(self.trace_failures),
        }
        if self.buckets:
            data["buckets"] = {
                key: {value: result.to_dict() for value, result in groups} for key, groups in self.buckets
            }
        return data


# Bucket keys map to a function extracting a bucket label from an instance.
BucketFn = Callable[[dict[str, object]], object]


def _difficulty(key: str) -> BucketFn:
    return lambda instance: instance.get("difficulty", {}).get(key)  # type: ignore[union-attr]


def _range_bucket(key: str, edges: tuple[int, ...]) -> BucketFn:
    def bucket(instance: dict[str, object]) -> object:
        value = _difficulty(key)(instance)
        if not isinstance(value, int):
            return None
        lower = 0
        for edge in edges:
            if value < edge:
                return f"{lower}-{edge - 1}"
            lower = edge
        return f"{lower}+"

    return bucket


DEFAULT_BUCKETS: dict[str, BucketFn] = {
    "split": lambda instance: instance.get("split"),
    "family": lambda instance: instance.get("family"),
    "target_answer": lambda instance: instance.get("target_answer"),
    "derivation_depth": _difficulty("derivation_depth"),
    "invalidity_type": _difficulty("invalidity_type"),
    "backtrack_count": _range_bucket("backtrack_count", (1, 4, 8, 16, 32)),
    "prune_count": _range_bucket("prune_count", (1, 4, 8, 16, 32)),
    "trace_length": _range_bucket("search_steps", (16, 32, 64, 128, 256)),
}


def score_prediction(instance: dict[str, object], prediction: dict[str, object]) -> RecordScore:
    output = str(prediction.get("output", prediction.get("completion", "")))
    cot, answer = extract_cot_and_answer(output)

    target_answer = str(instance["target_answer"])
    grammar = Grammar.from_dict(instance["grammar"])
    tokens = tuple(instance["input"])
    oracle_answer = "accept" if accepts(grammar, tokens) else "reject"
    expected_trace = str(instance["target_trace_compact"])
    search_config = instance.get("search_config") or {}
    max_depth = search_config.get("max_depth") if isinstance(search_config, dict) else None
    max_steps = search_config.get("max_steps") if isinstance(search_config, dict) else None

    if cot is not None:
        check = check_compact_trace(grammar, tokens, cot, expected_trace, max_depth=max_depth, max_steps=max_steps)
        prefix_len, target_len = trace_prefix_match(cot, expected_trace)
        trace_valid, exact, conclusion = check.valid, check.exact_match, check.conclusion
        failure, first_invalid, message = check.failure, check.first_invalid_step, check.message
        if check.limit_hit:
            trace_valid = False
            failure = "budget_exhausted"
            message = "valid bounded prefix, but membership remains unknown"
    else:
        prefix_len, target_len = trace_prefix_match("", expected_trace)
        trace_valid, exact, conclusion = False, False, None
        failure, first_invalid, message = "missing_cot", None, "no <cot> block"

    return RecordScore(
        task_id=str(instance["task_id"]),
        format_valid=cot is not None and answer is not None,
        verdict_correct=answer == target_answer,
        trace_valid=trace_valid,
        trace_exact_match=exact,
        verdict_trace_consistent=trace_valid and answer is not None and conclusion == answer,
        oracle_trace_agreement=oracle_answer == target_answer,
        prefix_match_len=prefix_len,
        target_trace_len=target_len,
        trace_failure=failure,
        first_invalid_step=first_invalid,
        trace_message=message,
    )


def aggregate_scores(scores: Iterable[RecordScore]) -> EvaluationResult:
    records = tuple(scores)
    failures = Counter(score.trace_failure for score in records if score.trace_failure is not None)
    return EvaluationResult(
        total=len(records),
        verdict_correct=sum(score.verdict_correct for score in records),
        trace_valid=sum(score.trace_valid for score in records),
        trace_exact_match=sum(score.trace_exact_match for score in records),
        format_valid=sum(score.format_valid for score in records),
        oracle_trace_agreement=sum(score.oracle_trace_agreement for score in records),
        verdict_trace_consistent=sum(score.verdict_trace_consistent for score in records),
        verdict_correct_trace_invalid=sum(score.verdict_correct_trace_invalid for score in records),
        prefix_match_ratio_sum=sum(score.prefix_match_ratio for score in records),
        trace_failures=tuple(sorted(failures.items())),
        records=records,
    )


def evaluate_predictions(
    instances: Iterable[dict[str, object]],
    predictions: Iterable[dict[str, object]],
    *,
    buckets: dict[str, BucketFn] | None = None,
) -> EvaluationResult:
    instance_by_id = {str(instance["task_id"]): instance for instance in instances}
    pairs = [
        (instance_by_id[str(prediction.get("task_id"))], prediction)
        for prediction in predictions
        if str(prediction.get("task_id")) in instance_by_id
    ]
    scores = [score_prediction(instance, prediction) for instance, prediction in pairs]
    result = aggregate_scores(scores)
    if not buckets:
        return result

    bucketed = []
    for key, bucket_fn in buckets.items():
        groups: dict[str, list[RecordScore]] = {}
        for (instance, _), score in zip(pairs, scores):
            groups.setdefault(str(bucket_fn(instance)), []).append(score)
        bucketed.append((key, tuple((value, aggregate_scores(group)) for value, group in sorted(groups.items()))))
    return replace(result, buckets=tuple(bucketed))


def evaluate_targets(
    instances: Iterable[dict[str, object]],
    *,
    buckets: dict[str, BucketFn] | None = None,
) -> EvaluationResult:
    records = list(instances)
    predictions = [{"task_id": record["task_id"], "output": record["completion"]} for record in records]
    return evaluate_predictions(records, predictions, buckets=buckets)


def read_jsonl(path: str) -> list[dict[str, object]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: str, payload: dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: str, rows: Iterable[dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
