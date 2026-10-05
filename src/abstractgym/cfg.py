"""Context-free grammar representation and parsing."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


EPSILON_ALIASES = {"", "eps", "epsilon", "ε"}
TOKEN_RE = re.compile(r'"([^"]*)"|(\S+)')


@dataclass(frozen=True)
class Production:
    """A single CFG production."""

    lhs: str
    rhs: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"lhs": self.lhs, "rhs": list(self.rhs)}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Production":
        rhs = data.get("rhs", [])
        if not isinstance(data.get("lhs"), str):
            raise ValueError("production lhs must be a string")
        if not isinstance(rhs, list) or not all(isinstance(item, str) for item in rhs):
            raise ValueError("production rhs must be a list of strings")
        return cls(lhs=data["lhs"], rhs=tuple(rhs))

    def format(self) -> str:
        rhs = " ".join(_format_symbol(symbol) for symbol in self.rhs) or "epsilon"
        return f"{self.lhs} -> {rhs}"


@dataclass(frozen=True)
class Grammar:
    """A context-free grammar with explicit terminal and nonterminal sets."""

    start: str
    nonterminals: frozenset[str]
    terminals: frozenset[str]
    productions: tuple[Production, ...]

    def __post_init__(self) -> None:
        if self.start not in self.nonterminals:
            raise ValueError(f"start symbol {self.start!r} is not a nonterminal")
        for production in self.productions:
            if production.lhs not in self.nonterminals:
                raise ValueError(f"production lhs {production.lhs!r} is not a nonterminal")
            unknown = set(production.rhs) - self.nonterminals - self.terminals
            if unknown:
                raise ValueError(f"unknown symbols in {production.format()}: {sorted(unknown)}")

    def productions_for(self, nonterminal: str) -> tuple[Production, ...]:
        return tuple(production for production in self.productions if production.lhs == nonterminal)

    def production_id(self, production: Production) -> str:
        return f"p{self.productions.index(production)}"

    def production_by_id(self, production_id: str) -> Production:
        if not production_id.startswith("p"):
            raise ValueError(f"invalid production id {production_id!r}")
        try:
            index = int(production_id[1:])
        except ValueError as exc:
            raise ValueError(f"invalid production id {production_id!r}") from exc
        try:
            return self.productions[index]
        except IndexError as exc:
            raise ValueError(f"unknown production id {production_id!r}") from exc

    def has_epsilon_productions(self) -> bool:
        return any(not production.rhs for production in self.productions)

    def left_recursive_nonterminals(self) -> frozenset[str]:
        """Return nonterminals involved in direct or indirect left recursion."""

        edges: dict[str, set[str]] = {nonterminal: set() for nonterminal in self.nonterminals}
        for production in self.productions:
            if production.rhs and production.rhs[0] in self.nonterminals:
                edges[production.lhs].add(production.rhs[0])

        recursive: set[str] = set()
        for start in self.nonterminals:
            stack = list(edges[start])
            seen: set[str] = set()
            while stack:
                current = stack.pop()
                if current == start:
                    recursive.add(start)
                    break
                if current in seen:
                    continue
                seen.add(current)
                stack.extend(edges[current])
        return frozenset(recursive)

    def validate_v0_constraints(self) -> None:
        if self.has_epsilon_productions():
            raise ValueError("v0 generated grammars must not contain epsilon productions")
        left_recursive = self.left_recursive_nonterminals()
        if left_recursive:
            raise ValueError(f"v0 generated grammars must not be left recursive: {sorted(left_recursive)}")

    def to_dict(self) -> dict[str, object]:
        return {
            "start": self.start,
            "nonterminals": sorted(self.nonterminals),
            "terminals": sorted(self.terminals),
            "productions": [production.to_dict() for production in self.productions],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Grammar":
        start = data.get("start")
        nonterminals = data.get("nonterminals")
        terminals = data.get("terminals")
        productions = data.get("productions")
        if not isinstance(start, str):
            raise ValueError("grammar start must be a string")
        if not isinstance(nonterminals, list) or not all(isinstance(item, str) for item in nonterminals):
            raise ValueError("grammar nonterminals must be a list of strings")
        if not isinstance(terminals, list) or not all(isinstance(item, str) for item in terminals):
            raise ValueError("grammar terminals must be a list of strings")
        if not isinstance(productions, list):
            raise ValueError("grammar productions must be a list")
        return cls(
            start=start,
            nonterminals=frozenset(nonterminals),
            terminals=frozenset(terminals),
            productions=tuple(Production.from_dict(item) for item in productions),
        )

    def format(self) -> str:
        lines = [f"%start {self.start}"]
        by_lhs: dict[str, list[Production]] = {}
        for production in self.productions:
            by_lhs.setdefault(production.lhs, []).append(production)
        for lhs in sorted(by_lhs):
            alternatives = []
            for production in by_lhs[lhs]:
                alternatives.append(" ".join(_format_symbol(symbol) for symbol in production.rhs) or "epsilon")
            lines.append(f"{lhs} -> {' | '.join(alternatives)}")
        return "\n".join(lines)


def parse_grammar(text: str) -> Grammar:
    """Parse the small grammar format used by examples and CLI input."""

    start: str | None = None
    raw_productions: list[Production] = []
    quoted_terminals: set[str] = set()
    nonterminals: set[str] = set()

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("%start"):
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"line {line_number}: expected '%start SYMBOL'")
            start = parts[1]
            nonterminals.add(start)
            continue
        if "->" not in line:
            raise ValueError(f"line {line_number}: expected a production with '->'")
        lhs, rhs_text = [part.strip() for part in line.split("->", 1)]
        if not lhs:
            raise ValueError(f"line {line_number}: missing production lhs")
        nonterminals.add(lhs)
        for alternative in _split_alternatives(rhs_text):
            rhs, terminals = _parse_rhs(alternative)
            quoted_terminals.update(terminals)
            raw_productions.append(Production(lhs=lhs, rhs=rhs))

    if start is None:
        if not raw_productions:
            raise ValueError("grammar has no productions")
        start = raw_productions[0].lhs
        nonterminals.add(start)

    rhs_symbols = {symbol for production in raw_productions for symbol in production.rhs}
    inferred_nonterminals = rhs_symbols - quoted_terminals
    nonterminals.update(inferred_nonterminals)
    terminals = quoted_terminals

    return Grammar(
        start=start,
        nonterminals=frozenset(nonterminals),
        terminals=frozenset(terminals),
        productions=tuple(raw_productions),
    )


def tokenize_input(value: str | Iterable[str]) -> tuple[str, ...]:
    """Tokenize a CLI input string or normalize an existing token sequence."""

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ()
        if " " in stripped:
            return tuple(stripped.split())
        return tuple(stripped)
    return tuple(value)


def _split_alternatives(rhs_text: str) -> list[str]:
    alternatives: list[str] = []
    current: list[str] = []
    in_quote = False
    for char in rhs_text:
        if char == '"':
            in_quote = not in_quote
            current.append(char)
        elif char == "|" and not in_quote:
            alternatives.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    alternatives.append("".join(current).strip())
    return alternatives


def _parse_rhs(text: str) -> tuple[tuple[str, ...], set[str]]:
    if text.lower() in EPSILON_ALIASES:
        return (), set()
    symbols: list[str] = []
    terminals: set[str] = set()
    for match in TOKEN_RE.finditer(text):
        quoted, bare = match.groups()
        if quoted is not None:
            symbols.append(quoted)
            terminals.add(quoted)
        elif bare.lower() not in EPSILON_ALIASES:
            symbols.append(bare)
    return tuple(symbols), terminals


def _format_symbol(symbol: str) -> str:
    if re.fullmatch(r"[A-Z][A-Za-z0-9_]*", symbol):
        return symbol
    escaped = symbol.replace('"', '\\"')
    return f'"{escaped}"'
