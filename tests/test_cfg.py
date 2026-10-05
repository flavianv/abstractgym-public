from abstractgym.cfg import Grammar, parse_grammar, tokenize_input


def test_parse_grammar_round_trip_dict() -> None:
    grammar = parse_grammar(
        """
        %start S
        S -> A B | "a"
        A -> "a" | epsilon
        B -> "b"
        """
    )

    assert grammar.start == "S"
    assert grammar.terminals == frozenset({"a", "b"})
    assert grammar.nonterminals == frozenset({"S", "A", "B"})
    assert Grammar.from_dict(grammar.to_dict()) == grammar


def test_tokenize_input_uses_chars_or_spaces() -> None:
    assert tokenize_input("ab") == ("a", "b")
    assert tokenize_input("a b") == ("a", "b")
    assert tokenize_input("") == ()


def test_v0_constraint_detection_finds_left_recursion() -> None:
    grammar = parse_grammar(
        """
        %start S
        S -> A "a" | "b"
        A -> S "c" | "d"
        """
    )

    assert grammar.left_recursive_nonterminals() == frozenset({"S", "A"})
