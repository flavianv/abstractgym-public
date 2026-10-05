"""Data contracts for the Gym-first CFG environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Literal

from abstractgym.cfg import Grammar


Split = Literal["train", "validation", "test"]
Validity = Literal["accepted", "rejected", "mixed"]
InvalidityType = Literal["wrong_prefix", "wrong_suffix", "too_short", "too_long", "exhaustive_near_miss"]
RecursionProfile = Literal["none", "direct", "mutual", "nested", "mixed"]


@dataclass(frozen=True)
class SearchConfig:
    policy: str = "leftmost_dfs"
    max_depth: int = 32
    max_steps: int = 512

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GrammarComplexityConfig:
    family: str = "recursive_growth"
    nonterminal_count: int = 3
    terminal_vocabulary: tuple[str, ...] = ("a", "b", "c")
    alternatives_per_nonterminal: tuple[int, int] = (1, 3)
    production_length: tuple[int, int] = (1, 2)
    recursion_profile: RecursionProfile = "direct"
    ambiguity: int = 0
    distractor_rule_count: int = 1
    dead_end_rule_count: int = 1
    production_order_policy: str = "adversarial"

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["terminal_vocabulary"] = list(self.terminal_vocabulary)
        data["alternatives_per_nonterminal"] = list(self.alternatives_per_nonterminal)
        data["production_length"] = list(self.production_length)
        return data


@dataclass(frozen=True)
class InstanceComplexityConfig:
    derivation_depth: int = 3
    string_length: int | None = None
    validity: Validity = "mixed"
    invalidity_type: InvalidityType = "wrong_suffix"
    failed_branch_depth: int = 1
    backtrack_count: int | None = None
    prune_count: int | None = None
    instance_recursion_depth: int | None = None
    search_steps: int | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SplitConfig:
    train_depth_ceiling: int = 4
    validation_depth_range: tuple[int, int] = (1, 4)
    eval_depth_range: tuple[int, int] = (5, 8)
    heldout_complexity_combinations: tuple[dict[str, object], ...] = field(default_factory=tuple)
    split_seed: int = 0

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["validation_depth_range"] = list(self.validation_depth_range)
        data["eval_depth_range"] = list(self.eval_depth_range)
        data["heldout_complexity_combinations"] = list(self.heldout_complexity_combinations)
        return data


@dataclass(frozen=True)
class GeneratedGrammar:
    grammar_id: str
    family: str
    grammar: Grammar
    grammar_complexity: GrammarComplexityConfig
    grammar_constraints: dict[str, object]
    grammar_stats: dict[str, object]
    sampling_seed: int

    def to_dict(self) -> dict[str, object]:
        return {
            "grammar_id": self.grammar_id,
            "family": self.family,
            "grammar": self.grammar.to_dict(),
            "grammar_complexity": self.grammar_complexity.to_dict(),
            "grammar_constraints": self.grammar_constraints,
            "grammar_stats": self.grammar_stats,
            "sampling_seed": self.sampling_seed,
        }


@dataclass(frozen=True)
class TaskInstance:
    task_id: str
    grammar_id: str
    family: str
    split: Split
    grammar: Grammar
    input: tuple[str, ...]
    search_config: SearchConfig
    difficulty: dict[str, object]
    prompt: str
    target_trace_compact: str
    target_trace_full: tuple[dict[str, Any], ...] | None
    target_answer: Literal["accept", "reject"]
    oracle_answer: Literal["accept", "reject"]

    def to_dict(self, *, include_full_trace: bool = False) -> dict[str, object]:
        data: dict[str, object] = {
            "task": "cfg_acceptance_trace",
            "task_id": self.task_id,
            "grammar_id": self.grammar_id,
            "family": self.family,
            "split": self.split,
            "grammar": self.grammar.to_dict(),
            "input": list(self.input),
            "search_config": self.search_config.to_dict(),
            "difficulty": self.difficulty,
            "prompt": self.prompt,
            "target_trace_compact": self.target_trace_compact,
            "target_answer": self.target_answer,
            "oracle_answer": self.oracle_answer,
            "completion": f"<cot>\n{self.target_trace_compact}\n</cot>\n<answer>{self.target_answer}</answer>",
        }
        if include_full_trace and self.target_trace_full is not None:
            data["target_trace_full"] = list(self.target_trace_full)
        return data


@dataclass(frozen=True)
class QuarantineRecord:
    grammar_id: str
    input: tuple[str, ...]
    reason: str
    oracle_answer: str
    trace_answer: str

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["input"] = list(self.input)
        return data


def stable_id(prefix: str, payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:12]
    return f"{prefix}_{digest}"
