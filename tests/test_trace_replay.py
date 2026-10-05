"""Adversarial traces that are locally plausible but not a faithful DFS execution."""

import pytest

from abstractgym.dataset import toy_backtracking_grammar
from abstractgym.trace import check_compact_trace, generate_trace, trace_prefix_match

GRAMMAR = toy_backtracking_grammar()
TOKENS = ("a", "b")
CANONICAL = generate_trace(GRAMMAR, TOKENS).target_trace_compact
STATES, ACTIONS = CANONICAL.split("T:\n")
ACTION_LINES = ACTIONS.splitlines()


def _trace(*lines: str) -> str:
    return STATES + "T:\n" + "\n".join(lines)


def _replace(index: int, line: str) -> str:
    lines = list(ACTION_LINES)
    lines[index] = line
    return _trace(*lines)


def _check(trace: str, **bounds: int):
    return check_compact_trace(GRAMMAR, TOKENS, trace, CANONICAL, **bounds)


def test_canonical_trace_replays_to_its_conclusion() -> None:
    check = _check(CANONICAL)
    assert check.valid and check.exact_match
    assert check.conclusion == "accept"
    rejected = generate_trace(GRAMMAR, ("b", "b")).target_trace_compact
    assert check_compact_trace(GRAMMAR, ("b", "b"), rejected, rejected).conclusion == "reject"


def test_locally_valid_steps_with_no_search_structure_are_rejected() -> None:
    # Every line passes a per-step check; together they are not an execution.
    nonsense = _trace(
        "0 TRY d0 s0 p0 s1",
        "1 DONE d7 s0 exhausted",
        "2 BT d3 s1 s1",
        "3 REJ d9 s0 whatever",
    )
    check = _check(nonsense)
    assert not check.valid
    assert check.failure == "wrong_depth"
    assert check.first_invalid_step == 1


@pytest.mark.parametrize(
    ("index", "line", "failure"),
    [
        (1, "1 TRY d2 s1 p3 s2", "wrong_depth"),
        (2, "2 PRN d2 s2 long", "false_reason"),
        (2, "2 REJ d2 s2 term", "wrong_opcode"),
        (3, "3 BT d1 s2 s0", "bad_backtrack"),
        (4, "4 TRY d1 s1 p3 s2", "repeated_production"),
        (6, "6 TRY d0 s0 p1 s4", "wrong_after_state"),
        (6, "6 TRY d0 s0 p3 s2", "bad_production"),
        (7, "7 TRY d1 s0 p0 s1", "wrong_node"),
        (14, "14 ACC d2 s5", "wrong_node"),
        (5, "6 BT d0 s1 s0", "numbering"),
        (5, "5 BT d0 s1", "malformed_action"),
    ],
)
def test_single_step_corruptions_are_localized(index: int, line: str, failure: str) -> None:
    check = _check(_replace(index, line))
    assert not check.valid
    assert check.failure == failure
    assert check.first_invalid_step == index


def test_done_before_all_productions_tried_is_premature() -> None:
    # After backtracking from s4, s3 still has p1 and p2 untried.
    lines = ACTION_LINES[:10] + ["10 DONE d1 s3 exhausted"]
    check = _check(_trace(*lines))
    assert check.failure == "premature_done"
    assert check.first_invalid_step == 10


def test_root_must_be_start_symbol() -> None:
    shortcut = "S:\ns0:a b\nT:\n0 ACC d0 s0"
    check = _check(shortcut)
    assert check.failure == "root_not_start"


def test_truncated_and_trailing_traces_are_invalid() -> None:
    assert _check(_trace(*ACTION_LINES[:-1])).failure == "truncated"
    trailing = _check(_trace(*ACTION_LINES, "15 BT d1 s6 s3"))
    assert trailing.failure == "trailing_actions"
    assert trailing.first_invalid_step == 15


def test_step_limit_truncation_concludes_unknown() -> None:
    limited = generate_trace(GRAMMAR, TOKENS, max_steps=6)
    assert limited.limit_hit
    check = check_compact_trace(GRAMMAR, TOKENS, limited.target_trace_compact, limited.target_trace_compact, max_steps=6)
    assert check.valid and check.limit_hit
    assert check.conclusion == "unknown"
    # Same prefix without the step bound is just an unfinished search.
    assert _check(limited.target_trace_compact).failure == "truncated"
    assert _check(CANONICAL, max_steps=6).failure == "over_step_limit"


def test_depth_cutoff_must_be_emitted_when_max_depth_exceeded() -> None:
    tokens = ("a", "a", "a", "b")
    cut = generate_trace(GRAMMAR, tokens, max_depth=2).target_trace_compact
    assert "CUT" in cut
    check = check_compact_trace(GRAMMAR, tokens, cut, cut, max_depth=2)
    assert check.valid and check.conclusion == "unknown" and check.limit_hit
    # Without the bound, a CUT is not a legal classification.
    assert check_compact_trace(GRAMMAR, tokens, cut, cut).failure == "wrong_opcode"


def test_prefix_match_ignores_state_numbering() -> None:
    renamed = CANONICAL.replace("s6", "s99")
    assert trace_prefix_match(renamed, CANONICAL) == (15, 15)
    assert trace_prefix_match(_replace(3, "3 BT d1 s2 s0"), CANONICAL) == (3, 15)
    assert trace_prefix_match("garbage", CANONICAL) == (0, 15)
