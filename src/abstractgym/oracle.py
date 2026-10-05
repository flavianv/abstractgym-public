"""Authoritative CFG membership oracle."""

from __future__ import annotations

from dataclasses import dataclass

from abstractgym.cfg import Grammar


@dataclass(frozen=True)
class EarleyItem:
    lhs: str
    rhs: tuple[str, ...]
    dot: int
    start: int

    @property
    def complete(self) -> bool:
        return self.dot >= len(self.rhs)

    def next_symbol(self) -> str | None:
        if self.complete:
            return None
        return self.rhs[self.dot]

    def advance(self) -> "EarleyItem":
        return EarleyItem(self.lhs, self.rhs, self.dot + 1, self.start)


def accepts(grammar: Grammar, tokens: tuple[str, ...]) -> bool:
    """Return whether ``grammar`` accepts ``tokens`` using Earley recognition."""

    augmented_start = "__START__"
    start_item = EarleyItem(augmented_start, (grammar.start,), 0, 0)
    chart: list[set[EarleyItem]] = [set() for _ in range(len(tokens) + 1)]
    chart[0].add(start_item)

    for position in range(len(chart)):
        changed = True
        while changed:
            changed = False
            for item in list(chart[position]):
                next_symbol = item.next_symbol()
                if next_symbol is None:
                    changed |= _complete(chart, position, item)
                elif next_symbol in grammar.nonterminals:
                    changed |= _predict(grammar, chart, position, next_symbol)
        if position < len(tokens):
            for item in list(chart[position]):
                next_symbol = item.next_symbol()
                if next_symbol == tokens[position]:
                    chart[position + 1].add(item.advance())

    return EarleyItem(augmented_start, (grammar.start,), 1, 0) in chart[len(tokens)]


def _predict(grammar: Grammar, chart: list[set[EarleyItem]], position: int, nonterminal: str) -> bool:
    changed = False
    for production in grammar.productions_for(nonterminal):
        item = EarleyItem(production.lhs, production.rhs, 0, position)
        if item not in chart[position]:
            chart[position].add(item)
            changed = True
    return changed


def _complete(chart: list[set[EarleyItem]], position: int, completed: EarleyItem) -> bool:
    changed = False
    for item in list(chart[completed.start]):
        if item.next_symbol() == completed.lhs:
            advanced = item.advance()
            if advanced not in chart[position]:
                chart[position].add(advanced)
                changed = True
    return changed
