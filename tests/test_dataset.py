from abstractgym.dataset import generate_dataset, generate_grammar, make_instance
from abstractgym.schema import GrammarComplexityConfig, InstanceComplexityConfig, SplitConfig


def test_generated_grammar_exposes_v0_constraints_and_knobs() -> None:
    generated = generate_grammar(
        GrammarComplexityConfig(
            nonterminal_count=4,
            terminal_vocabulary=("x", "y", "z"),
            distractor_rule_count=2,
            dead_end_rule_count=1,
        ),
        seed=3,
    )

    assert generated.grammar_id.startswith("grammar_")
    assert generated.grammar_constraints["no_epsilon"] is True
    assert generated.grammar_constraints["no_left_recursion"] is True
    assert generated.grammar_complexity.nonterminal_count == 4
    assert generated.grammar_stats["num_nonterminals"] == 4


def test_one_grammar_can_generate_multiple_string_instances() -> None:
    generated = generate_grammar(seed=5)
    first, first_quarantine = make_instance(
        generated,
        InstanceComplexityConfig(derivation_depth=2, validity="accepted"),
        index=1,
    )
    second, second_quarantine = make_instance(
        generated,
        InstanceComplexityConfig(derivation_depth=4, validity="rejected", invalidity_type="wrong_suffix"),
        index=2,
    )

    assert first_quarantine is None
    assert second_quarantine is None
    assert first is not None and second is not None
    assert first.grammar_id == second.grammar_id == generated.grammar_id
    assert first.input != second.input
    assert first.target_answer == "accept"
    assert second.target_answer == "reject"


def test_dataset_respects_depth_split_boundaries() -> None:
    batch = generate_dataset(
        split_config=SplitConfig(train_depth_ceiling=3, validation_depth_range=(1, 3), eval_depth_range=(4, 6)),
        train_count=8,
        validation_count=4,
        test_count=6,
        seed=11,
    )

    train_depths = [item.difficulty["derivation_depth"] for item in batch.instances if item.split == "train"]
    test_depths = [item.difficulty["derivation_depth"] for item in batch.instances if item.split == "test"]
    assert max(train_depths) <= 3
    assert min(test_depths) >= 4
    assert not batch.quarantine
