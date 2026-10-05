from abstractgym.dataset import generate_dataset
from abstractgym.cfg import Grammar
from abstractgym.evaluate import DEFAULT_BUCKETS, evaluate_predictions, evaluate_targets
from abstractgym.trace import _apply_production, _leftmost_nonterminal, _prune_reason, _short_reason


def test_evaluate_targets_reports_verdict_and_trace_separately() -> None:
    batch = generate_dataset(train_count=3, validation_count=0, test_count=0, seed=2)
    records = [instance.to_dict() for instance in batch.instances]

    result = evaluate_targets(records).to_dict()

    assert result["verdict_accuracy"] == 1.0
    assert result["trace_validity"] == 1.0
    assert result["trace_exact_match"] == 1.0


def test_correct_verdict_does_not_require_exact_trace() -> None:
    batch = generate_dataset(train_count=6, validation_count=0, test_count=0, seed=2)
    record = next(instance for instance in batch.instances if instance.target_answer == "accept").to_dict()
    # Trace a non-canonical production order: shortcut straight to the final derivation.
    noncanonical = _reordered_accepting_trace(record)
    prediction = {
        "task_id": record["task_id"],
        "output": f"<cot>\n{noncanonical}\n</cot>\n<answer>accept</answer>",
    }

    result = evaluate_predictions([record], [prediction]).to_dict()

    assert result["verdict_accuracy"] == 1.0
    assert result["trace_validity"] == 1.0
    assert result["trace_exact_match"] == 0.0
    assert result["verdict_trace_consistency"] == 1.0


def test_locally_plausible_nonsense_is_invalid_but_verdict_can_be_right() -> None:
    batch = generate_dataset(train_count=6, validation_count=0, test_count=0, seed=2)
    record = next(instance for instance in batch.instances if instance.target_answer == "reject").to_dict()
    first_try = record["target_trace_compact"].split("T:\n", 1)[1].splitlines()[0]
    states = record["target_trace_compact"].split("T:\n", 1)[0]
    root = first_try.split()[3]
    nonsense = f"{states}T:\n{first_try}\n1 DONE d7 {root} exhausted"
    prediction = {"task_id": record["task_id"], "output": f"<cot>\n{nonsense}\n</cot>\n<answer>reject</answer>"}

    result = evaluate_predictions([record], [prediction]).to_dict()

    assert result["verdict_accuracy"] == 1.0
    assert result["trace_validity"] == 0.0
    assert result["verdict_correct_trace_invalid_rate"] == 1.0
    assert result["trace_failures"] == {"wrong_depth": 1}
    assert result["mean_prefix_match_ratio"] > 0.0


def test_valid_trace_with_contradicting_answer_is_inconsistent() -> None:
    batch = generate_dataset(train_count=6, validation_count=0, test_count=0, seed=2)
    record = next(instance for instance in batch.instances if instance.target_answer == "accept").to_dict()
    prediction = {
        "task_id": record["task_id"],
        "output": f"<cot>\n{record['target_trace_compact']}\n</cot>\n<answer>reject</answer>",
    }

    result = evaluate_predictions([record], [prediction]).to_dict()

    assert result["trace_validity"] == 1.0
    assert result["verdict_accuracy"] == 0.0
    assert result["verdict_trace_consistency"] == 0.0


def test_missing_cot_scores_zero_prefix() -> None:
    batch = generate_dataset(train_count=2, validation_count=0, test_count=0, seed=2)
    record = batch.instances[0].to_dict()
    prediction = {"task_id": record["task_id"], "output": f"<answer>{record['target_answer']}</answer>"}

    result = evaluate_predictions([record], [prediction]).to_dict()

    assert result["format_validity"] == 0.0
    assert result["trace_failures"] == {"missing_cot": 1}
    assert result["mean_prefix_match_ratio"] == 0.0


def test_buckets_partition_records() -> None:
    batch = generate_dataset(train_count=8, validation_count=4, test_count=4, seed=3)
    records = [instance.to_dict() for instance in batch.instances]

    result = evaluate_targets(records, buckets=dict(DEFAULT_BUCKETS)).to_dict()

    for key in DEFAULT_BUCKETS:
        groups = result["buckets"][key]
        assert sum(group["total"] for group in groups.values()) == result["total"]
    assert set(result["buckets"]["split"]) == {"train", "validation", "test"}


def _reordered_accepting_trace(record: dict) -> str:
    """Re-run DFS with productions for each nonterminal in reverse order, keeping production IDs."""

    grammar = Grammar.from_dict(record["grammar"])
    tokens = tuple(record["input"])
    states: dict[tuple[str, ...], str] = {}
    actions: list[str] = []

    def sid(sentential: tuple[str, ...]) -> str:
        return states.setdefault(sentential, f"s{len(states)}")

    def search(sentential: tuple[str, ...], depth: int) -> bool:
        current = sid(sentential)
        if _prune_reason(grammar, sentential, tokens) is not None:
            actions.append(f"PRN d{depth} {current} {_short_reason(_prune_reason(grammar, sentential, tokens))}")
            return False
        index = _leftmost_nonterminal(grammar, sentential)
        if index is None:
            accepted = sentential == tokens
            actions.append(f"ACC d{depth} {current}" if accepted else f"REJ d{depth} {current} term")
            return accepted
        productions = grammar.productions_for(sentential[index])
        if not productions:
            actions.append(f"REJ d{depth} {current} no_prod")
            return False
        for production in reversed(productions):
            after = _apply_production(sentential, index, production)
            actions.append(f"TRY d{depth} {current} {grammar.production_id(production)} {sid(after)}")
            if search(after, depth + 1):
                return True
            actions.append(f"BT d{depth} {sid(after)} {current}")
        actions.append(f"DONE d{depth} {current} exhausted")
        return False

    assert search((grammar.start,), 0)
    lines = ["S:", *(f"{state_id}:{' '.join(sentential)}" for sentential, state_id in states.items()), "T:"]
    lines.extend(f"{number} {action}" for number, action in enumerate(actions))
    return "\n".join(lines)
