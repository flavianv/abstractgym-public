"""Deterministic DFS trace solver and compact trace validation."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from abstractgym.cfg import Grammar, Production


TraceStep = dict[str, Any]
CompactAction = tuple[str, ...]


@dataclass(frozen=True)
class TraceResult:
    accepted: bool
    target_trace_compact: str
    target_trace_full: tuple[TraceStep, ...]
    limit_hit: bool = False

    @property
    def trace(self) -> tuple[TraceStep, ...]:
        return self.target_trace_full

    def to_dict(self, *, include_full_trace: bool = False) -> dict[str, object]:
        data: dict[str, object] = {
            "accepted": self.accepted,
            "target_trace_compact": self.target_trace_compact,
            "limit_hit": self.limit_hit,
        }
        if include_full_trace:
            data["target_trace_full"] = list(self.target_trace_full)
        return data


@dataclass(frozen=True)
class TraceCheck:
    valid: bool
    exact_match: bool
    message: str
    conclusion: str | None = None
    failure: str | None = None
    first_invalid_step: int | None = None
    limit_hit: bool = False


class TraceReplayError(ValueError):
    """A submitted trace diverges from a faithful leftmost DFS execution."""

    def __init__(self, failure: str, message: str, step: int | None = None) -> None:
        super().__init__(message)
        self.failure = failure
        self.step = step


@dataclass(frozen=True)
class ReplayResult:
    conclusion: str
    limit_hit: bool


def generate_trace(
    grammar: Grammar,
    tokens: tuple[str, ...],
    *,
    max_depth: int = 32,
    max_steps: int = 512,
    include_full_trace: bool = True,
) -> TraceResult:
    """Generate the canonical bounded leftmost DFS derivation trace."""

    grammar.validate_v0_constraints()
    full_trace: list[TraceStep] = []
    state_ids: dict[tuple[str, ...], str] = {}
    states: list[tuple[str, ...]] = []
    actions: list[str] = []
    limit_hit = False
    depth_cut = False

    def state_id(sentential: tuple[str, ...]) -> str:
        if sentential not in state_ids:
            state_ids[sentential] = f"s{len(states)}"
            states.append(sentential)
        return state_ids[sentential]

    def emit(step: TraceStep, action: str) -> bool:
        nonlocal limit_hit
        if len(actions) >= max_steps:
            limit_hit = True
            return False
        if include_full_trace:
            full_trace.append(step)
        actions.append(f"{len(actions)} {action}")
        return True

    def search(sentential: tuple[str, ...], depth: int) -> bool:
        nonlocal depth_cut
        if limit_hit:
            return False
        current_id = state_id(sentential)
        if depth > max_depth:
            depth_cut = True
            return emit(
                {"op": "cutoff", "depth": depth, "sentential": list(sentential), "reason": "max_depth"},
                f"CUT d{depth} {current_id} max_depth",
            ) and False

        prune_reason = _prune_reason(grammar, sentential, tokens)
        if prune_reason is not None:
            return emit(
                {"op": "prune", "depth": depth, "sentential": list(sentential), "reason": prune_reason},
                f"PRN d{depth} {current_id} {_short_reason(prune_reason)}",
            ) and False

        next_index = _leftmost_nonterminal(grammar, sentential)
        if next_index is None:
            accepted = sentential == tokens
            if accepted:
                return emit(
                    {"op": "accept", "depth": depth, "sentential": list(sentential)},
                    f"ACC d{depth} {current_id}",
                ) and True
            return emit(
                {"op": "reject_leaf", "depth": depth, "sentential": list(sentential), "reason": "terminal_mismatch"},
                f"REJ d{depth} {current_id} term",
            ) and False

        nonterminal = sentential[next_index]
        productions = grammar.productions_for(nonterminal)
        if not productions:
            return emit(
                {"op": "reject_leaf", "depth": depth, "sentential": list(sentential), "reason": "no_productions"},
                f"REJ d{depth} {current_id} no_prod",
            ) and False

        for production in productions:
            after = _apply_production(sentential, next_index, production)
            after_id = state_id(after)
            production_id = grammar.production_id(production)
            if not emit(
                {
                    "op": "try_production",
                    "depth": depth,
                    "before": list(sentential),
                    "nonterminal_index": next_index,
                    "production_id": production_id,
                    "production": production.to_dict(),
                    "after": list(after),
                },
                f"TRY d{depth} {current_id} {production_id} {after_id}",
            ):
                return False
            if search(after, depth + 1):
                return True
            if not emit(
                {"op": "backtrack", "depth": depth, "from": list(after), "to": list(sentential)},
                f"BT d{depth} {after_id} {current_id}",
            ):
                return False

        return emit(
            {
                "op": "reject_node",
                "depth": depth,
                "sentential": list(sentential),
                "reason": "all_productions_exhausted",
            },
            f"DONE d{depth} {current_id} exhausted",
        ) and False

    accepted = search((grammar.start,), 0)
    if limit_hit:
        accepted = False
    return TraceResult(
        accepted=accepted,
        target_trace_compact=_render_compact_trace(states, actions),
        target_trace_full=tuple(full_trace),
        limit_hit=limit_hit or (depth_cut and not accepted),
    )


def trace_to_cot(compact_trace: str) -> str:
    return f"<cot>\n{compact_trace.strip()}\n</cot>"


def check_compact_trace(
    grammar: Grammar,
    tokens: tuple[str, ...],
    submitted_trace: str,
    expected_trace: str,
    *,
    max_depth: int | None = None,
    max_steps: int | None = None,
) -> TraceCheck:
    """Validate a submitted compact trace separately from exact canonical match.

    Validity means the trace replays as a faithful leftmost DFS execution:
    correct node classifications, stack-consistent depths and backtracks,
    exhaustive DONE, and a single root-level conclusion. Production order may
    differ from the canonical grammar order; exact match is the stricter check.
    """

    try:
        states, actions = parse_compact_trace(submitted_trace)
    except ValueError as exc:
        return TraceCheck(valid=False, exact_match=False, message=str(exc), failure="parse_error", first_invalid_step=0)
    try:
        replay = replay_compact_trace(grammar, tokens, states, actions, max_depth=max_depth, max_steps=max_steps)
    except TraceReplayError as exc:
        return TraceCheck(
            valid=False,
            exact_match=False,
            message=str(exc),
            failure=exc.failure,
            first_invalid_step=exc.step,
        )

    exact = _normalize_compact_trace(submitted_trace) == _normalize_compact_trace(expected_trace)
    message = "ok" if exact else "valid trace, not exact canonical trace"
    return TraceCheck(
        valid=True,
        exact_match=exact,
        message=message,
        conclusion=replay.conclusion,
        limit_hit=replay.limit_hit,
    )


def trace_prefix_match(submitted_trace: str, expected_trace: str) -> tuple[int, int]:
    """Return (matching leading actions, expected action count).

    State IDs are resolved to their sentential forms so that a trace with a
    different state numbering is not penalized. Unparseable submissions score 0.
    """

    expected = _resolved_actions(expected_trace)
    try:
        submitted = _resolved_actions(submitted_trace)
    except ValueError:
        return 0, len(expected)
    matched = 0
    for got, want in zip(submitted, expected):
        if got != want:
            break
        matched += 1
    return matched, len(expected)


def _resolved_actions(text: str) -> list[tuple[str, ...]]:
    states, actions = parse_compact_trace(text)
    return [
        tuple("[" + " ".join(states[part]) + "]" if part in states else part for part in action[1:])
        for action in actions
    ]


def parse_compact_trace(text: str) -> tuple[dict[str, tuple[str, ...]], list[CompactAction]]:
    states: dict[str, tuple[str, ...]] = {}
    actions: list[CompactAction] = []
    section: str | None = None
    for raw_line in text.strip().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line in {"S:", "T:"}:
            section = line[0]
            continue
        if section == "S":
            state_id, sep, symbols_text = line.partition(":")
            if sep != ":" or not re.fullmatch(r"s\d+", state_id):
                raise ValueError(f"invalid state line: {line!r}")
            symbols = tuple(symbols_text.split())
            if not symbols:
                raise ValueError(f"state {state_id} is empty; v0 traces do not use epsilon sentential forms")
            states[state_id] = symbols
        elif section == "T":
            parts = tuple(line.split())
            if len(parts) < 3 or not parts[0].isdigit():
                raise ValueError(f"invalid action line: {line!r}")
            actions.append(parts)
        else:
            raise ValueError(f"line outside S/T section: {line!r}")
    if not states:
        raise ValueError("compact trace has no state table")
    if not actions:
        raise ValueError("compact trace has no actions")
    return states, actions


def extract_cot_and_answer(output: str) -> tuple[str | None, str | None]:
    cot_match = re.search(r"<cot>\s*(.*?)\s*</cot>", output, flags=re.DOTALL | re.IGNORECASE)
    answer_match = re.search(r"<answer>\s*(accept|reject)\s*</answer>", output, flags=re.IGNORECASE)
    cot = cot_match.group(1).strip() if cot_match else None
    answer = answer_match.group(1).lower() if answer_match else None
    return cot, answer


def verify_example(example: dict[str, Any], *, max_depth: int = 32, max_steps: int = 512) -> tuple[bool, str]:
    """Verify that a generated example matches the canonical DFS trace."""

    try:
        grammar = Grammar.from_dict(example["grammar"])
        tokens = tuple(example["input"])
        search_config = example.get("search_config", {})
        if isinstance(search_config, dict):
            max_depth = int(search_config.get("max_depth", max_depth))
            max_steps = int(search_config.get("max_steps", max_steps))
        expected = generate_trace(grammar, tokens, max_depth=max_depth, max_steps=max_steps)
    except (KeyError, TypeError, ValueError) as exc:
        return False, f"invalid example: {exc}"

    trace = example.get("target_trace_compact", example.get("trace"))
    if not isinstance(trace, str):
        return False, "target_trace_compact must be a string"
    check = check_compact_trace(
        grammar, tokens, trace, expected.target_trace_compact, max_depth=max_depth, max_steps=max_steps
    )
    if not check.exact_match:
        return False, check.message
    if check.limit_hit:
        return False, "search budget exhausted; membership is unknown"
    expected_answer = "accept" if expected.accepted else "reject"
    if example.get("target_answer", "accept" if example.get("accepted") else "reject") != expected_answer:
        return False, f"target_answer mismatch: expected {expected_answer}"
    return True, "ok"


def _render_compact_trace(states: list[tuple[str, ...]], actions: list[str]) -> str:
    lines = ["S:"]
    for index, sentential in enumerate(states):
        lines.append(f"s{index}:{_format_sentential(sentential)}")
    lines.append("T:")
    lines.extend(actions)
    return "\n".join(lines)


def _format_sentential(sentential: tuple[str, ...]) -> str:
    return " ".join(sentential)


def _normalize_compact_trace(text: str) -> str:
    return "\n".join(line.strip() for line in text.strip().splitlines() if line.strip())


@dataclass
class _Frame:
    state_id: str
    sentential: tuple[str, ...]
    depth: int
    tried: set[str]
    phase: str = "entered"  # entered -> (child running) -> await_bt -> expanding
    child_id: str | None = None


def replay_compact_trace(
    grammar: Grammar,
    tokens: tuple[str, ...],
    states: dict[str, tuple[str, ...]],
    actions: list[CompactAction],
    *,
    max_depth: int | None = None,
    max_steps: int | None = None,
) -> ReplayResult:
    """Replay compact actions against a leftmost DFS stack machine.

    Raises TraceReplayError at the first action that a faithful execution could
    not have emitted.
    """

    if max_steps is not None and len(actions) > max_steps:
        raise TraceReplayError("over_step_limit", f"trace has {len(actions)} actions, max_steps is {max_steps}", max_steps)

    stack: list[_Frame] = []
    conclusion: str | None = None
    depth_cut = False

    for step, action in enumerate(actions):
        def fail(failure: str, detail: str) -> TraceReplayError:
            return TraceReplayError(failure, f"step {step}: {detail}: {' '.join(action)}", step)

        if int(action[0]) != step:
            raise fail("numbering", "action numbers must be contiguous from 0")
        if conclusion is not None:
            raise fail("trailing_actions", f"search already concluded {conclusion}")
        opcode = action[1]

        if step == 0:
            root_id = action[3] if len(action) > 3 else None
            if root_id not in states:
                raise fail("bad_state_ref", "unknown root state")
            if states[root_id] != (grammar.start,):
                raise fail("root_not_start", f"root state must be the start symbol {grammar.start!r}")
            stack.append(_Frame(root_id, states[root_id], 0, set()))

        frame = stack[-1]
        if len(action) < 4 or action[2] != f"d{frame.depth}":
            raise fail("wrong_depth", f"expected depth d{frame.depth}")

        if frame.phase == "await_bt":
            if opcode != "BT":
                raise fail("wrong_opcode", "expected BT after failed child")
            _expect_fields(action, 5, step)
            if action[3] != frame.child_id or action[4] != frame.state_id:
                raise fail("bad_backtrack", f"expected BT from {frame.child_id} to {frame.state_id}")
            frame.phase = "expanding"
            continue

        if action[3] != frame.state_id:
            raise fail("wrong_node", f"expected current node {frame.state_id}")

        if frame.phase == "entered":
            expected = _classify_node(grammar, frame.sentential, tokens, frame.depth, max_depth)
            if expected is None:
                if opcode != "TRY":
                    raise fail("wrong_opcode", "node must be expanded with TRY")
            else:
                expected_opcode, expected_reason = expected
                if opcode != expected_opcode:
                    raise fail("wrong_opcode", f"expected {expected_opcode}")
                if opcode == "ACC":
                    _expect_fields(action, 4, step)
                    conclusion = "accept"
                    continue
                _expect_fields(action, 5, step)
                if action[4] != expected_reason:
                    raise fail("false_reason", f"expected reason {expected_reason}")
                depth_cut |= opcode == "CUT"
                conclusion = _pop_failed(stack)
                continue
        elif frame.phase == "expanding":
            if opcode == "DONE":
                _expect_fields(action, 5, step)
                untried = [
                    grammar.production_id(p)
                    for p in grammar.productions_for(frame.sentential[_leftmost_nonterminal(grammar, frame.sentential)])
                    if grammar.production_id(p) not in frame.tried
                ]
                if untried:
                    raise fail("premature_done", f"untried productions {' '.join(untried)}")
                if action[4] != "exhausted":
                    raise fail("false_reason", "expected reason exhausted")
                conclusion = _pop_failed(stack)
                continue
            if opcode != "TRY":
                raise fail("wrong_opcode", "expected TRY or DONE")

        # TRY from an entered or expanding node.
        _expect_fields(action, 6, step)
        production_id, after_id = action[4], action[5]
        if after_id not in states:
            raise fail("bad_state_ref", "unknown after state")
        try:
            production = grammar.production_by_id(production_id)
        except (KeyError, ValueError, IndexError):
            raise fail("bad_production", "unknown production") from None
        index = _leftmost_nonterminal(grammar, frame.sentential)
        assert index is not None
        if production.lhs != frame.sentential[index]:
            raise fail("bad_production", f"production lhs must be {frame.sentential[index]}")
        if production_id in frame.tried:
            raise fail("repeated_production", "production already tried at this node")
        if _apply_production(frame.sentential, index, production) != states[after_id]:
            raise fail("wrong_after_state", "after state does not match production application")
        frame.tried.add(production_id)
        stack.append(_Frame(after_id, states[after_id], frame.depth + 1, set()))

    if conclusion is not None:
        limited = depth_cut and conclusion != "accept"
        return ReplayResult(conclusion="unknown" if limited else conclusion, limit_hit=limited)
    if max_steps is not None and len(actions) == max_steps:
        return ReplayResult(conclusion="unknown", limit_hit=True)
    raise TraceReplayError("truncated", "trace ends before the search concludes", len(actions))


def _classify_node(
    grammar: Grammar,
    sentential: tuple[str, ...],
    tokens: tuple[str, ...],
    depth: int,
    max_depth: int | None,
) -> tuple[str, str | None] | None:
    """Return the (opcode, reason) a freshly entered node must emit, or None to expand."""

    if max_depth is not None and depth > max_depth:
        return "CUT", "max_depth"
    prune_reason = _prune_reason(grammar, sentential, tokens)
    if prune_reason is not None:
        return "PRN", _short_reason(prune_reason)
    index = _leftmost_nonterminal(grammar, sentential)
    if index is None:
        return ("ACC", None) if sentential == tokens else ("REJ", "term")
    if not grammar.productions_for(sentential[index]):
        return "REJ", "no_prod"
    return None


def _pop_failed(stack: list[_Frame]) -> str | None:
    failed = stack.pop()
    if not stack:
        return "reject"
    parent = stack[-1]
    parent.phase = "await_bt"
    parent.child_id = failed.state_id
    return None


def _expect_fields(action: CompactAction, expected: int, step: int) -> None:
    if len(action) != expected:
        raise TraceReplayError(
            "malformed_action",
            f"step {step}: {action[1]} expects {expected} fields: {' '.join(action)}",
            step,
        )


def _leftmost_nonterminal(grammar: Grammar, sentential: tuple[str, ...]) -> int | None:
    for index, symbol in enumerate(sentential):
        if symbol in grammar.nonterminals:
            return index
    return None


def _apply_production(
    sentential: tuple[str, ...],
    nonterminal_index: int,
    production: Production,
) -> tuple[str, ...]:
    return sentential[:nonterminal_index] + production.rhs + sentential[nonterminal_index + 1 :]


def _prune_reason(grammar: Grammar, sentential: tuple[str, ...], tokens: tuple[str, ...]) -> str | None:
    terminal_count = 0
    for symbol in sentential:
        if symbol in grammar.terminals:
            if terminal_count >= len(tokens) or symbol != tokens[terminal_count]:
                return "terminal_prefix_mismatch"
            terminal_count += 1
        else:
            break

    fixed_terminals = sum(1 for symbol in sentential if symbol in grammar.terminals)
    if fixed_terminals > len(tokens):
        return "too_many_terminals"
    return None


def _short_reason(reason: str) -> str:
    return {
        "terminal_prefix_mismatch": "pref",
        "too_many_terminals": "long",
    }.get(reason, reason)
