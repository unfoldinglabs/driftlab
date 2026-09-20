"""Conditional policy rules: the shared piece of "hidden rules as programs".

RuleWorld already keeps its policy as a program (nested exceptions with a
`depth` knob). This module brings the same idea to worlds whose hidden state
is a schema or a convention set: a seeded list of IF-THEN rules over task
features ("when the order is large, priority is required"; "when the function
takes several parameters, type hints are enforced"). The world supplies the
predicates and the possible effects; the policy owns sampling, resolution and
mutation, so drift can rewire one conditional instead of flipping a flat fact.

Ground truth stays exact: the policy is the program the world executes.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ConditionalPolicy:
    """`n` seeded rules pairing a named predicate with an effect. `active(ctx)`
    returns the effects whose predicates hold for the current task context."""
    seed: int
    n: int
    predicates: dict          # name -> callable(ctx) -> bool
    effects: list             # effect names the world knows how to enforce
    rules: list = field(init=False, default_factory=list)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed + 77_777)
        pairs = [(w, e) for w in self.predicates for e in self.effects]
        self.rng.shuffle(pairs)
        self.rules = [{"when": w, "then": e} for w, e in pairs[: self.n]]

    def active(self, ctx) -> list:
        return [r["then"] for r in self.rules if self.predicates[r["when"]](ctx)]

    def mutate(self) -> str:
        """Rewire one rule (new predicate or new effect); returns a description."""
        if not self.rules:
            return "no conditional rules to change"
        r = self.rules[int(self.rng.integers(len(self.rules)))]
        old = dict(r)
        if self.rng.random() < 0.5 and len(self.predicates) > 1:
            r["when"] = str(self.rng.choice([w for w in self.predicates if w != r["when"]]))
        elif len(self.effects) > 1:
            r["then"] = str(self.rng.choice([e for e in self.effects if e != r["then"]]))
        return f"conditional rule changed: when '{r['when']}' then '{r['then']}' (was: when '{old['when']}' then '{old['then']}')"

    def describe(self) -> list:
        return [f"when {r['when']} then {r['then']}" for r in self.rules]
