"""Gym-first CFG data generation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import random
from typing import Iterable

from abstractgym.cfg import Grammar, Production
from abstractgym.oracle import accepts
from abstractgym.schema import (
    GeneratedGrammar,
    GrammarComplexityConfig,
    InstanceComplexityConfig,
    QuarantineRecord,
    SearchConfig,
    Split,
    SplitConfig,
    TaskInstance,
    stable_id,
)
from abstractgym.trace import generate_trace


@dataclass(frozen=True)
class GeneratedBatch:
    instances: tuple[TaskInstance, ...]
    grammars: tuple[GeneratedGrammar, ...]
    quarantine: tuple[QuarantineRecord, ...]


def toy_backtracking_grammar() -> Grammar:
    """Return a tiny v0 grammar that forces simple backtracking."""

    return Grammar(
        start="S",
        nonterminals=frozenset({"S", "D"}),
        terminals=frozenset({"a", "b", "c"}),
        productions=(
            Production("S", ("a", "D")),
            Production("S", ("a", "S")),
            Production("S", ("b",)),
            Production("D", ("c",)),
        ),
    )


def generate_grammar(config: GrammarComplexityConfig | None = None, *, seed: int = 0) -> GeneratedGrammar:
    config = config or GrammarComplexityConfig()
    if config.family not in {"recursive_growth", "prefix_decoy"}:
        raise ValueError(f"unsupported v0 grammar family {config.family!r}")
    if config.nonterminal_count < 2:
        raise ValueError("nonterminal_count must be at least 2")
    if len(config.terminal_vocabulary) < 3:
        raise ValueError("terminal_vocabulary must contain at least three symbols for v0 decoys")

    grammar = _build_right_recursive_grammar(config)
    grammar.validate_v0_constraints()
    # Identity describes content, never the RNG seed. The legacy generator is
    # deterministic for a fixed config; its depth splits share this grammar.
    payload = grammar.to_dict()
    return GeneratedGrammar(
        grammar_id=stable_id("grammar", payload),
        family=config.family,
        grammar=grammar,
        grammar_complexity=config,
        grammar_constraints={
            "no_epsilon": True,
            "no_left_recursion": True,
            "bounded_production_length": True,
            "bounded_generation_recursion": True,
        },
        grammar_stats=grammar_stats(grammar),
        sampling_seed=seed,
    )


def make_instance(
    generated: GeneratedGrammar,
    instance_config: InstanceComplexityConfig,
    *,
    split: Split = "train",
    search_config: SearchConfig | None = None,
    include_full_trace: bool = False,
    index: int = 0,
) -> tuple[TaskInstance | None, QuarantineRecord | None]:
    search_config = search_config or SearchConfig()
    tokens = sample_input(generated, instance_config)
    oracle_accepts = accepts(generated.grammar, tokens)
    trace = generate_trace(
        generated.grammar,
        tokens,
        max_depth=search_config.max_depth,
        max_steps=search_config.max_steps,
        include_full_trace=True,
    )
    oracle_answer = "accept" if oracle_accepts else "reject"
    trace_answer = "accept" if trace.accepted else "reject"
    if trace.limit_hit or oracle_answer != trace_answer:
        return None, QuarantineRecord(
            grammar_id=generated.grammar_id,
            input=tokens,
            reason="search_budget_exhausted" if trace.limit_hit else "oracle_trace_disagreement",
            oracle_answer=oracle_answer,
            trace_answer=trace_answer,
        )

    difficulty = trace_difficulty(trace.target_trace_full, tokens, instance_config)
    task_payload = {
        "grammar_id": generated.grammar_id,
        "split": split,
        "input": list(tokens),
        "instance_config": instance_config.to_dict(),
        "index": index,
    }
    return TaskInstance(
        task_id=stable_id("task", task_payload),
        grammar_id=generated.grammar_id,
        family=generated.family,
        split=split,
        grammar=generated.grammar,
        input=tokens,
        search_config=search_config,
        difficulty=difficulty,
        prompt=_prompt(generated, tokens),
        target_trace_compact=trace.target_trace_compact,
        target_trace_full=trace.target_trace_full if include_full_trace else None,
        target_answer=oracle_answer,
        oracle_answer=oracle_answer,
    ), None


def generate_dataset(
    *,
    grammar_config: GrammarComplexityConfig | None = None,
    split_config: SplitConfig | None = None,
    search_config: SearchConfig | None = None,
    train_count: int = 20,
    validation_count: int = 10,
    test_count: int = 10,
    instances_per_grammar: int = 4,
    seed: int = 0,
    include_full_trace: bool = False,
) -> GeneratedBatch:
    grammar_config = grammar_config or GrammarComplexityConfig()
    split_config = split_config or SplitConfig()
    search_config = search_config or SearchConfig()
    rng = random.Random(seed)
    instances: list[TaskInstance] = []
    grammars: list[GeneratedGrammar] = []
    quarantine: list[QuarantineRecord] = []

    if instances_per_grammar < 1 or min(train_count, validation_count, test_count) < 0:
        raise ValueError("counts must be nonnegative and instances_per_grammar positive")

    for split, count in (("train", train_count), ("validation", validation_count), ("test", test_count)):
        generated_count = 0
        attempts = 0
        while generated_count < count:
            attempts += 1
            if attempts > max(100, count * 100):
                raise ValueError("could not fill split within sampling budget; inspect search bounds/configuration")
            generated = generate_grammar(grammar_config, seed=rng.randrange(1_000_000_000))
            grammars.append(generated)
            for _ in range(instances_per_grammar):
                if generated_count >= count:
                    break
                instance_config = _sample_instance_config(split, split_config, rng)
                _ensure_no_split_leakage(split, grammar_config, instance_config, split_config)
                instance, record = make_instance(
                    generated,
                    instance_config,
                    split=split,
                    search_config=search_config,
                    include_full_trace=include_full_trace,
                    index=generated_count,
                )
                if record is not None:
                    quarantine.append(record)
                    continue
                instances.append(instance)
                generated_count += 1

    return GeneratedBatch(instances=tuple(instances), grammars=tuple(grammars), quarantine=tuple(quarantine))


def generate_examples(count: int, *, seed: int = 0) -> Iterable[dict[str, object]]:
    batch = generate_dataset(train_count=count, validation_count=0, test_count=0, seed=seed)
    for instance in batch.instances:
        yield instance.to_dict()


def write_jsonl(path: str, examples: Iterable[dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, sort_keys=True) + "\n")


def write_batch(
    batch: GeneratedBatch,
    *,
    instances_path: str,
    grammars_path: str | None = None,
    quarantine_path: str | None = None,
    include_full_trace: bool = False,
) -> None:
    write_jsonl(instances_path, (instance.to_dict(include_full_trace=include_full_trace) for instance in batch.instances))
    if grammars_path:
        write_jsonl(grammars_path, (grammar.to_dict() for grammar in batch.grammars))
    if quarantine_path:
        write_jsonl(quarantine_path, (record.to_dict() for record in batch.quarantine))


def sample_input(generated: GeneratedGrammar, config: InstanceComplexityConfig) -> tuple[str, ...]:
    terminals = tuple(generated.grammar_complexity.terminal_vocabulary)
    grow, final = terminals[0], terminals[1]
    wrong = "__bad__"
    depth = max(1, config.derivation_depth)
    grow_count = depth - 1
    target_length = config.string_length if config.string_length is not None else grow_count + 1
    if config.validity == "accepted":
        return tuple([grow] * max(0, target_length - 1) + [final])
    if config.validity == "rejected":
        return _invalid_tokens(grow, final, wrong, grow_count, config)
    if depth % 2 == 0:
        return tuple([grow] * grow_count + [final])
    return _invalid_tokens(grow, final, wrong, grow_count, config)


def grammar_stats(grammar: Grammar) -> dict[str, object]:
    alternatives = [len(grammar.productions_for(nonterminal)) for nonterminal in grammar.nonterminals]
    return {
        "num_nonterminals": len(grammar.nonterminals),
        "num_terminals": len(grammar.terminals),
        "num_productions": len(grammar.productions),
        "max_rhs_length": max((len(production.rhs) for production in grammar.productions), default=0),
        "choice_points": sum(1 for count in alternatives if count > 1),
        "max_branching_factor": max(alternatives, default=0),
        "recursive_nonterminals": len(_right_recursive_nonterminals(grammar)),
        "recursion_shape": "direct" if _right_recursive_nonterminals(grammar) else "none",
        "ambiguity_level": 0,
        "decoy_density": _decoy_density(grammar),
    }


def trace_difficulty(
    trace: tuple[dict[str, object], ...],
    tokens: tuple[str, ...],
    config: InstanceComplexityConfig,
) -> dict[str, object]:
    ops = [step["op"] for step in trace]
    depths = [int(step.get("depth", 0)) for step in trace]
    return {
        "derivation_depth": config.derivation_depth,
        "string_length": len(tokens),
        "validity": config.validity,
        "invalidity_type": config.invalidity_type if config.validity != "accepted" else None,
        "failed_branch_depth": config.failed_branch_depth,
        "backtrack_count": ops.count("backtrack"),
        "prune_count": ops.count("prune"),
        "instance_recursion_depth": config.instance_recursion_depth,
        "search_steps": len(trace),
        "max_observed_depth": max(depths, default=0),
        "acceptance": "accept" if "accept" in ops else "reject",
    }


def _build_right_recursive_grammar(config: GrammarComplexityConfig) -> Grammar:
    grow, final, wrong = config.terminal_vocabulary[:3]
    nonterminals = ["S"] + [f"D{i}" for i in range(1, config.nonterminal_count)]
    productions: list[Production] = []
    decoy_count = min(config.distractor_rule_count, max(0, len(nonterminals) - 1))

    for index in range(decoy_count):
        decoy = nonterminals[index + 1]
        productions.append(Production("S", (grow, decoy)))
        if index + 2 < len(nonterminals):
            productions.append(Production(decoy, (grow, nonterminals[index + 2])))
        else:
            productions.append(Production(decoy, (wrong,)))

    if config.recursion_profile == "none":
        productions.append(Production("S", (grow, final)))
    else:
        productions.append(Production("S", (grow, "S")))
    productions.append(Production("S", (final,)))

    for dead_index in range(config.dead_end_rule_count):
        if dead_index + 1 < len(nonterminals):
            productions.append(Production(nonterminals[dead_index + 1], (wrong,)))

    return Grammar(
        start="S",
        nonterminals=frozenset(nonterminals),
        terminals=frozenset(config.terminal_vocabulary),
        productions=tuple(productions),
    )


def _sample_instance_config(split: Split, split_config: SplitConfig, rng: random.Random) -> InstanceComplexityConfig:
    if split == "train":
        depth = rng.randint(1, split_config.train_depth_ceiling)
    elif split == "validation":
        depth = rng.randint(*split_config.validation_depth_range)
    else:
        depth = rng.randint(*split_config.eval_depth_range)
    validity = "accepted" if rng.random() < 0.5 else "rejected"
    invalidity_type = rng.choice(["wrong_prefix", "wrong_suffix", "too_short", "too_long", "exhaustive_near_miss"])
    return InstanceComplexityConfig(
        derivation_depth=depth,
        validity=validity,
        invalidity_type=invalidity_type,
        failed_branch_depth=max(1, min(depth, 3)),
        instance_recursion_depth=depth - 1,
    )


def _invalid_tokens(
    grow: str,
    final: str,
    wrong: str,
    grow_count: int,
    config: InstanceComplexityConfig,
) -> tuple[str, ...]:
    if config.invalidity_type == "wrong_prefix":
        return tuple([wrong] + [grow] * grow_count + [final])
    if config.invalidity_type == "too_short":
        return tuple([grow] * max(0, grow_count - 1))
    if config.invalidity_type == "too_long":
        return tuple([grow] * grow_count + [final, wrong])
    if config.invalidity_type == "exhaustive_near_miss":
        return tuple([grow] * (grow_count + max(1, config.failed_branch_depth)))
    return tuple([grow] * grow_count + [wrong])


def _ensure_no_split_leakage(
    split: Split,
    grammar_config: GrammarComplexityConfig,
    instance_config: InstanceComplexityConfig,
    split_config: SplitConfig,
) -> None:
    if split == "train" and instance_config.derivation_depth > split_config.train_depth_ceiling:
        raise ValueError("training instance exceeds train_depth_ceiling")
    if split == "test":
        low, high = split_config.eval_depth_range
        if not low <= instance_config.derivation_depth <= high:
            raise ValueError("test instance falls outside eval_depth_range")
    if split == "train":
        for combo in split_config.heldout_complexity_combinations:
            if _matches_combo(grammar_config, instance_config, combo):
                raise ValueError(f"training instance matches held-out complexity combination: {combo}")


def _matches_combo(
    grammar_config: GrammarComplexityConfig,
    instance_config: InstanceComplexityConfig,
    combo: dict[str, object],
) -> bool:
    for key, expected in combo.items():
        if key.startswith("grammar."):
            actual = getattr(grammar_config, key.removeprefix("grammar."))
        elif key.startswith("instance."):
            actual = getattr(instance_config, key.removeprefix("instance."))
        else:
            raise ValueError(f"held-out combo keys must start with grammar. or instance.: {key}")
        if isinstance(expected, list) and len(expected) == 2 and all(isinstance(value, int) for value in expected):
            if not expected[0] <= actual <= expected[1]:
                return False
        elif actual != expected:
            return False
    return True


def _right_recursive_nonterminals(grammar: Grammar) -> set[str]:
    recursive = set()
    for production in grammar.productions:
        if production.lhs in production.rhs[1:]:
            recursive.add(production.lhs)
    return recursive


def _decoy_density(grammar: Grammar) -> float:
    decoys = sum(1 for production in grammar.productions if any(symbol.startswith("D") for symbol in production.rhs))
    if not grammar.productions:
        return 0.0
    return decoys / len(grammar.productions)


def _prompt(generated: GeneratedGrammar, tokens: tuple[str, ...]) -> str:
    return (
        "Decide whether the CFG accepts the input. Produce only the compact "
        "canonical DFS trace actions inside <cot> tags, then give "
        "<answer>accept</answer> or <answer>reject</answer>.\n\n"
        f"Grammar ID: {generated.grammar_id}\n"
        f"Grammar:\n{generated.grammar.format()}\n\n"
        f"Input tokens: {list(tokens)}"
    )
