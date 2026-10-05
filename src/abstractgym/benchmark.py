"""Frozen, contrastive CFG benchmark with explicit structural split identities.

This is separate from the legacy v0 generator, whose splits share a grammar.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import random

from abstractgym.cfg import Grammar, Production
from abstractgym.oracle import accepts
from abstractgym.schema import stable_id
from abstractgym.trace import generate_trace

VERSION = "cfg-controller-v1"


def _grammar(family, shape, vocabulary, center, rng, swap_closes=False):
    # Ordered widths are canonicalized by _shapes; symbol renaming cannot move
    # a structural template into a different split.
    w0, w1, base_width, decoys = shape
    a, b, c, d, _, poison, _ = vocabulary
    pairs = [(a, b, w0), (c, d, w1)]
    if swap_closes:
        pairs = [(a, d, w0), (c, b, w1)]
    rules = []
    for i in range(decoys):
        name = f"D{i}"
        rules.append(Production("S", (name,)))
        for left, right, width in pairs:
            rhs = (left,) * width + (name,)
            if family == "nested":
                rhs += (right,) * width
            rules.append(Production(name, rhs))
        rules.append(Production(name, (center,) * base_width + (poison,) * (i + 1)))
    recursive = []
    for left, right, width in pairs:
        rhs = (left,) * width + ("S",)
        if family == "nested":
            rhs += (right,) * width
        recursive.append(Production("S", rhs))
    rng.shuffle(recursive)
    # Deliberately explore decoys before productive alternatives.
    rules.extend(recursive)
    rules.append(Production("S", (center,) * base_width))
    grammar = Grammar("S", frozenset(["S"] + [f"D{i}" for i in range(decoys)]),
                      frozenset(vocabulary), tuple(rules))
    grammar.validate_v0_constraints()
    return grammar


def _shapes():
    return [(a, b, c, d) for a, b, c, d in itertools.product(range(1, 5), range(1, 5), range(1, 5), range(1, 4)) if a <= b]


def build_suite(*, seed=17, structures_per_split=4, pairs_per_structure=4,
                train_depth=4, test_depth=(5, 8), max_steps=4096, max_depth=64):
    if structures_per_split < 1 or pairs_per_structure < 1:
        raise ValueError("structure and pair counts must be positive")
    if train_depth < 2 or not train_depth < test_depth[0] <= test_depth[1]:
        raise ValueError("training ceiling must be >=2 and test depths strictly greater")
    if max_steps < 1 or max_depth < 1:
        raise ValueError("search budgets must be positive")
    shapes = _shapes()
    if 3 * structures_per_split > len(shapes):
        raise ValueError("requested more disjoint structural templates than available")
    rng = random.Random(seed)
    rng.shuffle(shapes)
    rows, quarantine = [], []
    for family in ("regular", "nested"):
        for split_index, split in enumerate(("train", "validation", "test")):
            selected = shapes[split_index * structures_per_split:(split_index + 1) * structures_per_split]
            for shape in selected:
                structure_id = stable_id("structure", [family, shape])
                vocabulary = list("abcdefg")
                rng.shuffle(vocabulary)
                # Contrastive variants stay in the same structural split.
                # Nested variants swap closing types; regular variants swap the base.
                grammars = [_grammar(family, shape, vocabulary,
                                     vocabulary[4] if family == "nested" else vocabulary[4 if variant == 0 else 6],
                                     rng, swap_closes=family == "nested" and variant == 1)
                            for variant in range(2)]
                for pair_index in range(pairs_per_structure * (2 if split == "test" else 1)):
                    depth_band = "deep" if split == "test" and pair_index >= pairs_per_structure else "shallow"
                    depth = rng.randint(*(test_depth if depth_band == "deep" else (2 if family == "nested" else 1, train_depth)))
                    source = rng.randrange(2)
                    path = [rng.randrange(2) for _ in range(depth - 1)]
                    if family == "nested":
                        # Balanced wrapper types make a subset resistant to a
                        # terminal-count-only recognizer as well as label priors.
                        path = [i % 2 for i in range(depth - 1)]
                        rng.shuffle(path)
                    tokens = (vocabulary[4 if family == "nested" or source == 0 else 6],) * shape[2]
                    for choice in reversed(path):
                        left, right = vocabulary[2 * choice:2 * choice + 2]
                        if family == "nested" and source == 1:
                            right = vocabulary[2 * (1 - choice) + 1]
                        tokens = (left,) * shape[choice] + tokens + ((right,) * shape[choice] if family == "nested" else ())
                    pair_id = stable_id("pair", [seed, structure_id, pair_index, tokens])
                    candidates = []
                    for variant, grammar in enumerate(grammars):
                        answer = "accept" if accepts(grammar, tokens) else "reject"
                        trace = generate_trace(grammar, tokens, max_depth=max_depth, max_steps=max_steps)
                        task_id = stable_id("case", [pair_id, variant])
                        if trace.limit_hit or trace.accepted != (answer == "accept"):
                            quarantine.append({"pair_id": pair_id, "task_id": task_id,
                                               "reason": "budget_exhausted" if trace.limit_hit else "oracle_disagreement"})
                            continue
                        candidates.append({"task_id": task_id, "pair_id": pair_id, "split": split,
                            "family": family, "depth_band": depth_band, "structure_id": structure_id,
                            "grammar_id": stable_id("grammar", grammar.to_dict()), "grammar": grammar.to_dict(),
                            "input": list(tokens), "target_answer": answer,
                            "search_config": {"max_steps": max_steps, "max_depth": max_depth},
                            "difficulty": {"derivation_depth": depth, "nesting_depth": depth - 1 if family == "nested" else 0,
                                "string_length": len(tokens), "decoys": shape[3],
                                "balanced_pair_counts": family == "nested" and path.count(0) * shape[0] == path.count(1) * shape[1],
                                "search_steps": len(trace.trace),
                                "backtracks": sum(s["op"] == "backtrack" for s in trace.trace)},
                            "target_trace_compact": trace.target_trace_compact})
                    if len(candidates) == 2:
                        assert {r["target_answer"] for r in candidates} == {"accept", "reject"}
                        rows.extend(candidates)
                    elif candidates:
                        quarantine.append({"pair_id": pair_id, "task_id": candidates[0]["task_id"], "reason": "paired_case_quarantined"})
    manifest = {"version": VERSION, "seed": seed, "structures_per_split": structures_per_split,
                "pairs_per_structure": pairs_per_structure, "train_depth": train_depth,
                "test_depth": list(test_depth), "max_steps": max_steps, "max_depth": max_depth,
                "requested_rows": 2 * 4 * structures_per_split * pairs_per_structure * 2,
                "retained_rows": len(rows), "quarantined_rows": len(quarantine),
                "split_policy": "disjoint structural templates; both contrastive variants in same split",
                "test_shift": "held-out structures have shallow and deep bands; compare depth within each structure. Neither band shares training structures.",
                "limits": "two controlled grammar families, not arbitrary CFGs; no universal-computation claim"}
    manifest["instances_sha256"] = hashlib.sha256(encode_rows(rows).encode()).hexdigest()
    return rows, manifest, quarantine


def encode_rows(rows):
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def write_suite(directory, **kwargs):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows, manifest, quarantine = build_suite(**kwargs)
    (directory / "instances.jsonl").write_text(encode_rows(rows))
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (directory / "quarantine.jsonl").write_text(encode_rows(quarantine))
    return manifest
