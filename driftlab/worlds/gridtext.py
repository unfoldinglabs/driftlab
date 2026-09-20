"""RuleWorld: an intake desk routing requests under procedures nobody wrote down.

Each step presents a request (type, region, size); the agent assigns it to one
of four handling options. Which is correct follows hidden rules with nested
exceptions:

    depth 1:  type                  -> option      (base rule)
    depth 2:  (type, region)        -> option      (exception)
    depth 3:  (type, region, size)  -> option      (exception to the exception)

Change capabilities: change_latent (a rule flips), change_surface (option
descriptions rotate; letters and rules fixed), introduce_novelty (a new request
type with fresh rules), probe, sample_task (by depth), sample_contrast_pair,
ask_confidence, task_key, consult. Rules can also flip on the world's own
schedule (`n_mutations`/`schedule`) for experiments that don't schedule changes.

With `transition_window=N` a latent change is gradual instead of instantaneous:
the affected keys migrate to the new desk a few per step over ~N steps (via
temporary deepest-level exceptions), so the truth stays deterministic at every
step while the world is genuinely mid-transition.

Epistemic action: with `consult=True` the agent may end a reply with `ASK: ...`
instead of a CHOICE; the floor manager names the right desk for the request in
front of it, and the step's reward is forfeited — information at a price.

Endogenous change: with `endogenous=True` the back office reacts to load. If
the agent routes more than `capacity_share` of recent accepted requests to one
option, that desk is over capacity and one of its base rules is moved to
another option: a rule change the agent caused.
"""

import re
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .base import CONFIDENCE_SUFFIX, flavor_rng, person

# signal-free requester chatter: realistic surface, no information about the hidden rules
CHATTER = ["Hoping this can go out before end of day.", "Same as the ones I sent last month, I think.",
           "Our client has been asking about this twice already.", "No rush, but please confirm receipt.",
           "Flagging that the paperwork was scanned, not typed.", "Let me know if anything is missing.",
           "This one came in through the regional office.", "Second attempt, the first form bounced."]
DEPTS = ["Client Services", "Regional Ops", "Partner Accounts", "Field Sales", "Finance Desk"]

TYPES = ["invoice", "refund", "shipment", "claim", "renewal", "transfer", "audit", "onboarding", "cancellation"]
REGIONS = ["north", "south", "east"]
SIZES = ["small", "large"]
OPTIONS = ["A", "B", "C", "D"]
LABEL_SETS = [
    {"A": "standard queue", "B": "priority desk", "C": "regional office", "D": "external partner"},
    {"A": "front desk", "B": "escalation team", "C": "field branch", "D": "outsourced vendor"},
    {"A": "tier 1", "B": "tier 2", "C": "local unit", "D": "third party"},
]
KEY_FIELDS = ("type", "region", "size")
# structural surface variants: the same request rendered in different formats and
# attribute orders — what is correct never changes, only how the task reads
RENDER_FORMATS = [
    "We have received a {size} {type} originating from the {region} region and need it assigned to the right desk.",
    "REQUEST DETAILS — region: {region} | size class: {size} | category: {type}",
    "Incoming case. Origin: {region} region. Volume: {size}. Classification: {type}.",
]

SYSTEM = ("You work on the intake desk at Meridian Logistics. Requests arrive one at a time; for each one, "
          "assign it to the handling option you believe is right. The back office tells you whether the "
          "assignment was accepted. Nobody has written the routing procedures down, so you will have to "
          "work them out on the job. End every reply with a line of the form:\nCHOICE: <A|B|C|D>")
ASK_SUFFIX = ("\nIf you are unsure, you may spend the step asking instead of routing: end your reply with the line\n"
              "ASK: <your question>\nThe floor manager will tell you the right option for the request in front of "
              "you, but that request goes back into the queue unprocessed — you get no credit for the step.")


def key_of(task: dict) -> tuple:
    return tuple(task[f] for f in KEY_FIELDS)


def task_of(key: tuple) -> dict:
    return dict(zip(KEY_FIELDS, key))


@dataclass
class RuleWorld:
    seed: int
    T: int = 120
    depth: int = 2
    n_mutations: int = 0
    exception_rate: float = 0.35
    warmup: int = 20
    n_types: int = 6
    schedule: object = "jittered"
    mutation_scope: str = "base"     # "base": flip only depth-1 rules | "any"
    feedback_noise: float = 0.0      # probability that a step's feedback is misleadingly inverted
    transition_window: int = 0       # >0: change_latent migrates a rule gradually over this many steps
    surface_strength: str = "label"  # label: descriptions rotate | structural: the request is also re-rendered
    ask_confidence: bool = False
    consult: bool = False            # epistemic action: ASK: forfeits the step, the floor manager answers
    task_fn: object = None           # (t, world) -> task; default: seeded random
    endogenous: bool = False
    capacity_share: float = 0.5      # endogenous: max share of recent accepted volume one desk tolerates
    capacity_window: int = 15
    name: str = "rule_world"

    system_prompt: str = field(init=False)
    rules: dict = field(init=False)
    mutation_steps: list = field(init=False)
    changes: list = field(init=False, default_factory=list)
    label_set: int = field(init=False, default=0)
    _applied_until: int = field(init=False, default=-1)
    _current: dict = field(init=False, default=None)
    _new: list = field(init=False, default_factory=list)
    _last_conf: object = field(init=False, default=None)
    _recent_accepted: deque = field(init=False, default_factory=deque)
    _unresolved: dict = field(init=False, default_factory=dict)
    _migrations: list = field(init=False, default_factory=list)
    _temp_rules: list = field(init=False, default_factory=list)
    _asked: bool = field(init=False, default=False)
    n_asks: int = field(init=False, default=0)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)
        self._task_rng = np.random.default_rng(self.seed + 10_000)
        self._noise_rng = np.random.default_rng(self.seed + 20_000)  # separate stream: noise never shifts tasks or schedules
        self.system_prompt = SYSTEM + (ASK_SUFFIX if self.consult else "") \
            + (CONFIDENCE_SUFFIX if self.ask_confidence else "")
        self._recent_accepted = deque(maxlen=self.capacity_window)
        self.types = TYPES[: self.n_types]
        self.rules = {}
        for ty in self.types:
            self._add_type_rules(ty)
        if isinstance(self.schedule, list):
            self.mutation_steps = sorted(self.schedule)
        elif self.n_mutations > 0:
            span = np.linspace(self.warmup, self.T - 5, self.n_mutations + 1)[:-1]
            jitter = self._rng.integers(0, 5, size=len(span)) if self.schedule == "jittered" else np.zeros(len(span), int)
            self.mutation_steps = sorted(int(s + j) for s, j in zip(span, jitter))
        else:
            self.mutation_steps = []

    def _add_type_rules(self, ty: str):
        self.rules[(ty,)] = str(self._rng.choice(OPTIONS))
        if self.depth >= 2:
            for rg in REGIONS:
                if self._rng.random() < self.exception_rate:
                    self.rules[(ty, rg)] = self._other(self.rules[(ty,)])
                    if self.depth >= 3:
                        for sz in SIZES:
                            if self._rng.random() < self.exception_rate:
                                self.rules[(ty, rg, sz)] = self._other(self.rules[(ty, rg)])

    def _other(self, opt: str, avoid: set | None = None) -> str:
        pool = [o for o in OPTIONS if o != opt and (not avoid or o not in avoid)]
        return str(self._rng.choice(pool or [o for o in OPTIONS if o != opt]))

    def _record(self, t, kind, desc, affected=None, **extra) -> dict:
        for a in affected or []:
            self._unresolved[tuple(a)] = t
        ch = {"t": t, "kind": kind, "desc": desc, "affected": [list(a) for a in (affected or [])], **extra}
        self.changes.append(ch)
        self._new.append(ch)
        return ch

    # ---- rules -----------------------------------------------------------------
    def resolve(self, task: dict) -> tuple[str, int]:
        ty, rg, sz = key_of(task)
        for key, d in (((ty, rg, sz), 3), ((ty, rg), 2), ((ty,), 1)):
            if key in self.rules:
                return self.rules[key], d
        raise KeyError(task)

    def correct(self, task: dict) -> str:
        return self.resolve(task)[0]

    def task_depth(self, task: dict) -> int:
        return self.resolve(task)[1]

    def truth_table(self) -> dict:
        return {(ty, rg, sz): self.correct(task_of((ty, rg, sz))) for ty in self.types for rg in REGIONS for sz in SIZES}

    def keys_affected_by(self, key: tuple) -> list[tuple]:
        out = []
        for full in self.truth_table():
            if full[: len(key)] == key:
                deeper = [k for k in self.rules if len(k) > len(key) and full[: len(k)] == k]
                if not deeper:
                    out.append(full)
        return out

    def mutable_keys(self) -> list[tuple]:
        keys = list(self.rules)
        return [k for k in keys if len(k) == 1] if self.mutation_scope == "base" else keys

    # ---- change capabilities -----------------------------------------------------
    def mutate_now(self, t: int, key: tuple | None = None, kind: str = "latent") -> tuple:
        if key is None:
            pool = self.mutable_keys()
            key = pool[self._rng.integers(len(pool))]
        old = self.rules[key]
        self.rules[key] = self._other(old)
        affected = self.keys_affected_by(key)
        self._record(t, kind, f"rule flip {key} {old} -> {self.rules[key]} (affects {len(affected)} keys)",
                     affected, key=list(key), old=old, new=self.rules[key])
        return (t, key, old, self.rules[key])

    def change_latent(self, t: int) -> str:
        if not self.transition_window:
            self.mutate_now(t)
            return self.changes[-1]["desc"]
        # gradual: the affected keys migrate to the new desk a few at a time over the window,
        # via temporary deepest-level exceptions; the truth stays deterministic throughout
        pool = self.mutable_keys()
        key = pool[self._rng.integers(len(pool))]
        old = self.rules[key]
        new = self._other(old)
        queue = list(self.keys_affected_by(key))
        self._rng.shuffle(queue)
        per = max(1, -(-len(queue) // self.transition_window))
        self._migrations.append({"key": key, "old": old, "new": new, "queue": queue, "per": per})
        self._record(t, "latent", f"rule change begins: {key} migrating {old} -> {new} "
                                  f"over ~{self.transition_window} steps ({len(queue)} keys)",
                     [], key=list(key), old=old, new=new)
        return self.changes[-1]["desc"]

    def _advance_migrations(self, t: int) -> list[str]:
        out = []
        for m in list(self._migrations):
            chunk, m["queue"] = m["queue"][:m["per"]], m["queue"][m["per"]:]
            for full in chunk:
                if full not in self.rules:
                    self._temp_rules.append((m["key"], full))
                self.rules[full] = m["new"]
            if chunk:
                out.append(self._record(t, "latent", f"migration {m['key']}: {len(chunk)} more "
                                                     f"keys now route to {m['new']}",
                                        chunk, key=list(m["key"]), old=m["old"], new=m["new"])["desc"])
            if not m["queue"]:  # finalize: flip the base rule, drop the temporary exceptions
                self.rules[m["key"]] = m["new"]
                for pair in [x for x in self._temp_rules if x[0] == m["key"]]:
                    self.rules.pop(pair[1], None)
                    self._temp_rules.remove(pair)
                self._migrations.remove(m)
                out.append(self._record(t, "latent", f"migration complete: {m['key']} now {m['new']}",
                                        [], key=list(m["key"]), old=m["old"], new=m["new"])["desc"])
        return out

    def change_surface(self, t: int) -> str:
        self.label_set = (self.label_set + 1) % len(LABEL_SETS)
        if self.surface_strength == "structural":
            self.render_set = (getattr(self, "render_set", 0) + 1) % len(RENDER_FORMATS)
            return self._record(t, "surface", f"intake form restructured and options relabeled: {self.options_text()}")["desc"]
        return self._record(t, "surface", f"option descriptions relabeled: {self.options_text()}")["desc"]

    relabel = change_surface

    def introduce_novelty(self, t: int) -> str | None:
        pool = [ty for ty in TYPES if ty not in self.types]
        if not pool:
            return None
        ty = pool[0]
        self.types.append(ty)
        self._add_type_rules(ty)
        return self._record(t, "novelty", f"new request type appears: {ty}")["desc"]

    introduce_type = introduce_novelty

    def apply_intent(self, t: int, intent: dict) -> str:
        """Coherent drift from an OrganizationProcess. A 'reorg' dissolves the desk with the
        most base rules — every one of them moves at once, one cause. 'tooling' re-words the
        surface and flips one rule. Anything else is a burst of related flips."""
        n = max(1, int(intent.get("n", 1)))
        if intent.get("channel") == "reorg":
            base = [k for k in self.rules if len(k) == 1]
            by_opt: dict = {}
            for k in base:
                by_opt.setdefault(self.rules[k], []).append(k)
            desk, keys = max(by_opt.items(), key=lambda kv: len(kv[1]))
            for key in keys[:n]:
                self.mutate_now(t, key=key)
            return f"reorg: desk {desk} dissolved; {min(n, len(keys))} request types reassigned"
        if intent.get("channel") == "tooling":
            self.change_surface(t)
            self.mutate_now(t)
            return "tooling change: relabel plus one rule flip"
        for _ in range(n):
            self.mutate_now(t)
        return f"{intent.get('channel', 'change')}: {n} related rule flips"

    def advance_to(self, t: int) -> list[str]:
        due = [ms for ms in self.mutation_steps if self._applied_until < ms <= t]
        self._applied_until = t
        return [self.change_latent(ms) for ms in due] + self._advance_migrations(t)

    def _endogenous_check(self, t: int):
        if not self.endogenous or len(self._recent_accepted) < self.capacity_window:
            return
        n = len(self._recent_accepted)
        counts = {o: sum(1 for x in self._recent_accepted if x == o) for o in OPTIONS}
        over = [o for o, c in counts.items() if c / n > self.capacity_share]
        if not over:
            return
        desk = over[0]
        base_keys = [k for k in self.rules if len(k) == 1 and self.rules[k] == desk]
        if not base_keys:
            return
        key = base_keys[self._rng.integers(len(base_keys))]
        old = self.rules[key]
        self.rules[key] = self._other(old, avoid=set(over))
        affected = self.keys_affected_by(key)
        self._record(t, "endogenous", f"desk {desk} over capacity; {key[0]} requests moved {old} -> {self.rules[key]} "
                                      f"(affects {len(affected)} keys)", affected, key=list(key), old=old, new=self.rules[key])
        self._recent_accepted.clear()

    # ---- tasks ---------------------------------------------------------------------
    def task(self, t: int) -> dict:
        rng = np.random.default_rng(self.seed * 100_003 + t)
        return {"type": str(rng.choice(self.types)), "region": str(rng.choice(REGIONS)), "size": str(rng.choice(SIZES))}

    def sample_task(self, depth: int | None = None, avoid: set | None = None, types: list | None = None) -> dict:
        pool = [k for k in self.truth_table()
                if (depth is None or self.task_depth(task_of(k)) == depth) and (types is None or k[0] in types)]
        if avoid:
            pool = [k for k in pool if k not in avoid] or pool
        if not pool:
            pool = list(self.truth_table())
        return task_of(pool[self._task_rng.integers(len(pool))])

    def sample_contrast_pair(self) -> tuple[dict, dict] | None:
        tt = self.truth_table()
        keys = list(tt)
        for _ in range(200):
            k = keys[self._task_rng.integers(len(keys))]
            i = int(self._task_rng.integers(3))
            values = {0: self.types, 1: REGIONS, 2: SIZES}[i]
            alts = [v for v in values if v != k[i]]
            v = alts[self._task_rng.integers(len(alts))]
            k2 = k[:i] + (v,) + k[i + 1:]
            if tt[k2] != tt[k]:
                return task_of(k), task_of(k2)
        return None

    def task_key(self, task: dict | None = None) -> list:
        return list(key_of(task or self._current))

    # ---- rendering -----------------------------------------------------------------
    def render(self, task: dict) -> str:
        return RENDER_FORMATS[getattr(self, "render_set", 0)].format(**task)

    def labels(self) -> dict:
        return LABEL_SETS[self.label_set]

    def options_text(self) -> str:
        return "Options: " + ", ".join(f"{k} ({v})" for k, v in self.labels().items())

    def probe(self, task: dict) -> str:
        return (f"Quick check from your supervisor (nothing will be processed and you will get no feedback; "
                f"just say how you would assign it):\n{self.render(task)}\n{self.options_text()}\nWhich option?")

    # ---- World protocol ------------------------------------------------------------
    def observe(self, t: int) -> str:
        self._current = self.task_fn(t, self) if self.task_fn else self.task(t)
        rng = flavor_rng(self.seed, t)
        intro = (f"Ticket REQ-{2400 + t} · from {person(rng)} ({rng.choice(DEPTS)})\n"
                 f"Subject: {self._current['type']} request — {self._current['region']} region\n"
                 f"\"{rng.choice(CHATTER)}\"")
        return f"{intro}\n{self.render(self._current)}\n{self.options_text()}\nWhich option do you choose?"

    def parse(self, text: str):
        c = re.search(r"CONFIDENCE:\s*(\d{1,3})", text)
        self._last_conf = min(float(c.group(1)), 100.0) / 100 if c else None
        m = re.search(r"CHOICE:\s*([ABCD])", text)
        if m:
            return m.group(1)
        if self.consult and re.search(r"^ASK:", text, re.M):
            return "ASK"
        return None

    def default_action(self, rng):
        return str(rng.choice(OPTIONS))

    def describe_action(self, action) -> str:
        return str(action)

    def feedback(self, task: dict, choice: str) -> tuple[int, str]:
        ok = choice == self.correct(task)
        return int(ok), ("Accepted: the request was processed successfully." if ok
                         else f"Rejected: option {choice} is not valid for this request.")

    def act(self, t: int, action: str) -> tuple[float, str]:
        if action == "ASK":
            self._asked = True
            self.n_asks += 1
            desk = self.correct(self._current)
            return 0.0, (f"Floor manager: this one goes to {desk} ({self.labels()[desk]}). "
                         "The request went back into the queue; take the next one.")
        reward, msg = self.feedback(self._current, action)
        if reward:
            self._unresolved.pop(key_of(self._current), None)
            self._recent_accepted.append(action)
            self._endogenous_check(t)
        if self.feedback_noise and self._noise_rng.random() < self.feedback_noise:
            # the world lies about this step's outcome; privileged() still logs the truth
            reward = 1 - reward
            msg = ("Accepted: the request was processed successfully." if reward
                   else f"Rejected: option {action} is not valid for this request.")
        return float(reward), msg

    def privileged(self, t: int) -> dict:
        task = self._current
        new, self._new = self._new, []
        asked, self._asked = self._asked, False
        return {"task": task, "key": list(key_of(task)), "task_key": list(key_of(task)), "correct": self.correct(task),
                "depth": self.task_depth(task), "confidence": self._last_conf, "asked": asked,
                "affected_now": key_of(task) in self._unresolved, "changes": new}

    def summary(self) -> dict:
        return {"mutation_steps": self.mutation_steps, "changes": self.changes, "n_asks": self.n_asks,
                "n_endogenous": sum(1 for c in self.changes if c["kind"] == "endogenous")}
