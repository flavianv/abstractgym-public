from abstractgym.cfg import parse_grammar
from abstractgym.dataset import make_instance, toy_backtracking_grammar
from abstractgym.schema import GeneratedGrammar, GrammarComplexityConfig, InstanceComplexityConfig
from abstractgym.trace import check_compact_trace, generate_trace, verify_example


def test_generate_trace_accepts_with_backtracking() -> None:
    grammar = toy_backtracking_grammar()
    result = generate_trace(grammar, ("a", "b"))

    assert result.accepted is True
    ops = [step["op"] for step in result.trace]
    assert ops.count("try_production") >= 4
    assert "prune" in ops
    assert "backtrack" in ops
    assert ops[-1] == "accept"
    assert "TRY" in result.target_trace_compact
    assert "BT" in result.target_trace_compact


def test_generate_trace_rejects_after_exhausting_search() -> None:
    grammar = toy_backtracking_grammar()
    result = generate_trace(grammar, ("b", "b"))

    assert result.accepted is False
    assert result.trace[-1]["op"] == "reject_node"


def test_verify_example_requires_canonical_trace() -> None:
    grammar = toy_backtracking_grammar()
    generated = GeneratedGrammar(
        grammar_id="g",
        family="recursive_growth",
        grammar=grammar,
        grammar_complexity=GrammarComplexityConfig(nonterminal_count=2),
        grammar_constraints={"no_epsilon": True, "no_left_recursion": True},
        grammar_stats={},
        sampling_seed=0,
    )
    instance, record = make_instance(generated, InstanceComplexityConfig(derivation_depth=2, validity="accepted"))
    assert record is None
    assert instance is not None
    example = instance.to_dict()

    ok, message = verify_example(example)
    assert ok, message

    example["target_trace_compact"] = example["target_trace_compact"].replace("TRY", "NOPE", 1)
    ok, message = verify_example(example)
    assert not ok
    assert "expected TRY" in message or "must be expanded with TRY" in message


def test_trace_validity_is_separate_from_exact_match() -> None:
    grammar = toy_backtracking_grammar()
    result = generate_trace(grammar, ("a", "b"))
    submitted = result.target_trace_compact.replace("0 TRY", "0 TRY")

    check = check_compact_trace(grammar, ("a", "b"), submitted, result.target_trace_compact + "\n")
    assert check.valid
    assert check.exact_match


def test_v0_trace_solver_rejects_epsilon_grammar() -> None:
    grammar = parse_grammar(
        """
        %start S
        S -> A
        A -> epsilon
        """
    )

    try:
        generate_trace(grammar, ())
    except ValueError as exc:
        assert "epsilon" in str(exc)
    else:
        raise AssertionError("expected v0 trace solver to reject epsilon grammar")
