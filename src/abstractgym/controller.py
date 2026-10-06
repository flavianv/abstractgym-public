"""Small-step DFS environment. State storage and validation do not pick actions.

The reference policy is used only to create teacher states and test fixtures.
The model receives observation(), never reference_action().
"""
from __future__ import annotations
from copy import deepcopy

from abstractgym.cfg import Grammar
from abstractgym.trace import _apply_production, _classify_node, _leftmost_nonterminal

HINT_FIELDS = ("untried_rules", "prefix_matches", "complete_match", "next_untried")

PROTOCOL = '''Execute deterministic leftmost DFS in grammar production order.
Return actions as JSON objects: {"op":"TRY","production":"p0"},
{"op":"PRN","reason":"pref"}, {"op":"REJ","reason":"term"},
{"op":"ACC"}, {"op":"BT"}, {"op":"DONE"}, or {"op":"CUT","reason":"max_depth"}.
At an entered node: CUT if depth exceeds max_depth; otherwise PRN pref if the
terminal prefix before the first nonterminal mismatches the input; otherwise
PRN long if the total fixed terminals exceed input length; otherwise ACC for
an exact terminal match, REJ term for any other terminal leaf, REJ no_prod for
a nonterminal with no productions, or TRY the first untried applicable rule.
TRY replaces the leftmost nonterminal and pushes a child with empty tried list.
After any failed child, emit BT at its parent, then TRY the next untried rule
or DONE when exhausted. DONE fails that node. A failed root rejects unless
any branch was cut, in which case the outcome is unknown. ACC accepts globally.
No actions may follow completion. A step budget exhaustion is unknown.
PRN also permits reason "long"; REJ also permits reason "no_prod".
Production IDs p0, p1, ... are indices in the observation's productions array.
'''


class ControllerEnv:
    def __init__(self, row):
        self.grammar = Grammar.from_dict(row["grammar"])
        self.tokens = tuple(row["input"])
        self.max_steps = row["search_config"]["max_steps"]
        self.max_depth = row["search_config"]["max_depth"]
        if self.max_steps < 1 or self.max_depth < 0:
            raise ValueError("invalid search budget")
        self.stack = [{"form": [self.grammar.start], "tried": [], "phase": "entered"}]
        self.steps = 0
        self.outcome = None
        self.cut = False

    def observation(self, hints=None):
        selected = set(HINT_FIELDS if hints is True else
                       () if hints is None or hints is False else
                       (hints,) if isinstance(hints, str) else hints)
        if selected - set(HINT_FIELDS):
            raise ValueError("unknown hint field")
        observation = deepcopy({"grammar": self.grammar.to_dict(), "input": list(self.tokens),
                         "stack": self.stack, "steps": self.steps,
                         "max_steps": self.max_steps, "max_depth": self.max_depth,
                         "cut_seen": self.cut, "outcome": self.outcome})
        if selected and observation["stack"]:
            frame = observation["stack"][-1]
            form = tuple(frame["form"])
            index = _leftmost_nonterminal(self.grammar, form)
            untried = ([] if index is None else
                       [self.grammar.production_id(rule)
                        for rule in self.grammar.productions_for(form[index])
                        if self.grammar.production_id(rule) not in frame["tried"]])
            # Ignore the depth cutoff when exposing a comparison, not an action.
            classification = _classify_node(self.grammar, form, self.tokens, 0, None)
            values = {"untried_rules": untried,
                      "prefix_matches": classification != ("PRN", "pref"),
                      "complete_match": index is None and form == self.tokens,
                      "next_untried": untried[0] if untried else None}
            frame.update({name: values[name] for name in HINT_FIELDS if name in selected})
        return observation

    def reference_action(self):
        if self.outcome is not None:
            raise ValueError("execution already completed")
        frame = self.stack[-1]
        if frame["phase"] == "await_bt":
            return {"op": "BT"}
        form = tuple(frame["form"])
        if frame["phase"] == "entered":
            classification = _classify_node(self.grammar, form, self.tokens, len(self.stack) - 1, self.max_depth)
            if classification:
                op, reason = classification
                return {"op": op, **({"reason": reason} if reason else {})}
        index = _leftmost_nonterminal(self.grammar, form)
        for rule in self.grammar.productions_for(form[index]):
            pid = self.grammar.production_id(rule)
            if pid not in frame["tried"]:
                return {"op": "TRY", "production": pid}
        return {"op": "DONE"}

    def step(self, action):
        # A strict validator stops on the first error. It supplies no correction,
        # candidate actions, retries, or gold state to the model after failure.
        if self.outcome is not None:
            raise ValueError("execution already completed")
        if action != self.reference_action():
            raise ValueError("invalid_action")
        frame = self.stack[-1]
        op = action["op"]
        if op == "TRY":
            rule = self.grammar.production_by_id(action["production"])
            form = tuple(frame["form"])
            after = _apply_production(form, _leftmost_nonterminal(self.grammar, form), rule)
            frame["tried"].append(action["production"])
            self.stack.append({"form": list(after), "tried": [], "phase": "entered"})
        elif op == "BT":
            frame["phase"] = "expanding"
        elif op == "ACC":
            self.outcome = "accept"
        else:
            self.cut |= op == "CUT"
            self.stack.pop()
            if self.stack:
                self.stack[-1]["phase"] = "await_bt"
            else:
                self.outcome = "unknown" if self.cut else "reject"
        self.steps += 1
        if self.outcome is None and self.steps >= self.max_steps:
            self.outcome = "unknown"
        return self.observation()


def teacher_trajectory(row, *, hints=None):
    env = ControllerEnv(row)
    trajectory = []
    while env.outcome is None:
        observation = env.observation(hints=hints)
        action = env.reference_action()
        trajectory.append({"observation": observation, "action": action})
        env.step(action)
    if env.outcome != row["target_answer"]:
        raise ValueError("teacher did not complete with the membership oracle answer")
    return trajectory
