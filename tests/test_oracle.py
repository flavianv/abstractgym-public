from abstractgym.cfg import parse_grammar
from abstractgym.oracle import accepts


def test_earley_oracle_accepts_epsilon_even_when_trace_solver_v0_does_not() -> None:
    grammar = parse_grammar(
        """
        %start S
        S -> A
        A -> epsilon
        """
    )

    assert accepts(grammar, ())
    assert not accepts(grammar, ("a",))


def test_earley_oracle_handles_right_recursion() -> None:
    grammar = parse_grammar(
        """
        %start S
        S -> "a" S | "b"
        """
    )

    assert accepts(grammar, ("a", "a", "b"))
    assert not accepts(grammar, ("a", "a"))
