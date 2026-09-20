"""ClaimsWorld: a claims desk where the fraud adapts to the reviewer.

Each step one insurance claim arrives, described by three attributes (type,
filing channel, amount). A fraud ring operates with a current signature: a
pair of attribute values its claims always carry. Legitimate claims never
fully match the signature, so the mapping from features to the right decision
is deterministic — the game is tracking the signature as it moves.

What makes this world different from every other: the drift is adversarial.
With `endogenous=True` (opt-in, like every world's self-caused mechanism),
the ring watches what gets through.
When too many of its claims are being rejected, it re-styles its claims to
mimic the traffic the agent has recently been approving — drift aims exactly
where the agent is weakest, and it is the agent's own decision pattern that
directs it.

Change capabilities: change_latent (the signature moves to a random new pair:
the scheduled, non-adversarial control), change_surface (the claim template is
re-worded; nothing about fraud changes), endogenous (the adversarial shift
described above), ask_confidence, task_key (claims repeat, so latencies count
encounters), consult.

With `transition_window=N` a scheduled tactic change is a gradual handover:
the ring starts the new playbook while the old pattern still circulates, the
fraud mix shifts toward the new signature over ~N steps, and the old pattern
is retired at the end (a second logged change). Both patterns are fraudulent
during the window, so the feature-to-decision truth stays deterministic.

Epistemic action: with `consult=True` the agent may reply `ASK: AUDIT` instead
of deciding; the audit desk reveals whether the current claim is fraudulent, a
colleague settles it, and the step's reward is forfeited.
"""

import itertools
import re
from dataclasses import dataclass, field

import numpy as np

from .base import CONFIDENCE_SUFFIX, flavor_rng, person

SYSTEM = ("You review insurance claims at Meridian Assurance. Claims arrive one at a time; approve the legitimate "
          "ones and reject the fraudulent ones. Fraud exists and fraud tactics change over time; you are never told "
          "the current pattern. End your reply with a line:\nDECISION: APPROVE or DECISION: REJECT")
ASK_SUFFIX = ("\nIf you are unsure, you may hold the claim and request a pre-decision audit instead: end your reply "
              "with the line\nASK: AUDIT\nThe audit desk will tell you whether this claim is fraudulent, but the "
              "decision passes to a colleague — you get no credit for the step.")

TYPES = ["auto", "home", "health", "travel", "gadget"]
CHANNELS = ["online", "phone", "paper"]
AMOUNTS = ["small", "large"]
ATTRS = {"type": TYPES, "channel": CHANNELS, "amount": AMOUNTS}
# narrative incident lines per claim type; the {} takes a seeded detail. Signal-free by
# construction: fraud is decided by the attribute signature, never by the story.
INCIDENTS = {
    "auto": ["Rear-ended at a stoplight on {}; bumper and sensor damage.", "Windscreen cracked by road debris on {}.",
             "Parked vehicle sideswiped overnight on {}; mirror and panel damage.", "Battery and electronics failed after flooding on {}."],
    "home": ["Burst pipe in the utility room on {}; water damage to flooring.", "Storm took several roof tiles on {}.",
             "Break-in through the back door on {}; locks and a window replaced.", "Power surge on {} damaged kitchen appliances."],
    "health": ["Outpatient treatment on {} following a fall.", "Physiotherapy sessions after an injury on {}.",
               "Emergency dental work on {} after an accident.", "Specialist consultation on {} following a referral."],
    "travel": ["Trip cancelled on {} after an airline schedule change.", "Checked luggage lost on the {} connection.",
               "Medical treatment abroad on {}; receipts attached.", "Missed connection on {} due to a delayed inbound flight."],
    "gadget": ["Phone screen shattered on {}; repair quote attached.", "Laptop stopped charging on {}; assessed as liquid damage.",
               "Camera stolen from a bag on {}; police report filed.", "Tablet dropped on {}; digitizer replacement quoted."],
}
# signal-free adjuster remarks: realistic file chatter, no information about fraud
ADJUSTER_NOTES = ["Documents scanned and attached to the file.", "Claimant reachable in the afternoons.",
                  "First claim on this policy.", "Photos pending from the claimant.",
                  "Policy premiums are up to date.", "Call notes filed under the claim id."]
DAYS = ["Mar 3", "Mar 14", "Apr 2", "Apr 19", "May 7", "May 26", "Jun 11", "Jun 30", "Jul 8", "Jul 22"]
CHANNEL_PHRASE = {"online": "submitted through the customer portal (online)",
                  "phone": "taken down by the call center (phone)",
                  "paper": "received by post and scanned (paper)"}
TEMPLATES = [  # {header} carries the narrative; the explicit category line keeps the features unambiguous
    "{header}\n{incident}\nFiling: {channel_phrase}. Requested payout: ${figure:,} ({amount} band).\n"
    "Adjuster note: {adjuster}\n"
    "Category: {type} · Channel: {channel} · Amount: {amount}",
    "{header}\n{incident}\nCame in {channel_phrase}; claimed amount ${figure:,}, {amount} band.\n"
    "Adjuster note: {adjuster}\n"
    "Category: {type} · Channel: {channel} · Amount: {amount}"]


def key_of(claim: dict) -> tuple:
    return (claim["type"], claim["channel"], claim["amount"])


@dataclass
class ClaimsWorld:
    seed: int
    T: int = 120
    n_types: int = 5                 # reduced modes shrink this for denser task encounters
    fraud_share: float = 0.4
    adapt_every: int = 20            # how often the ring reconsiders its tactics
    adapt_window: int = 20           # how much recent traffic it watches
    warmup: int = 15
    transition_window: int = 0       # >0: a scheduled tactic change is a gradual handover, not a jump
    ask_confidence: bool = False
    consult: bool = False            # epistemic action: ASK: AUDIT forfeits the step, the audit desk answers
    endogenous: bool = False         # the adversary; off by default so scheduled experiments stay controlled
    name: str = "claims_desk"

    system_prompt: str = field(init=False)
    changes: list = field(init=False, default_factory=list)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.system_prompt = SYSTEM + (ASK_SUFFIX if self.consult else "") \
            + (CONFIDENCE_SUFFIX if self.ask_confidence else "")
        self.types = TYPES[: self.n_types]
        self.attrs = {"type": self.types, "channel": CHANNELS, "amount": AMOUNTS}
        self.signature = self._random_signature()
        self.template = 0
        self.recent: list = []       # (claim, action, is_fraud) the ring can observe
        self._new: list = []
        self._current: dict = None
        self._last_conf = None
        self._n = 0
        self._old_sig: dict | None = None   # still-circulating signature during a gradual handover
        self._handover: tuple | None = None  # (start, end) of the handover window
        self._asked = False
        self.n_asks = 0

    # ---- fraud machinery ------------------------------------------------------------
    def _random_signature(self, avoid: dict | None = None) -> dict:
        while True:
            attrs = list(self.rng.choice(list(self.attrs), size=2, replace=False))
            sig = {a: str(self.rng.choice(self.attrs[a])) for a in sorted(attrs)}
            if sig != (avoid or {}):
                return sig

    def _matches(self, claim: dict, sig: dict | None = None) -> bool:
        sig = sig or self.signature
        return all(claim.get(a) == v for a, v in sig.items())

    def _is_fraud_pattern(self, claim: dict) -> bool:
        """During a gradual handover both the new and the still-circulating old signature are
        fraudulent, so the feature-to-decision mapping stays deterministic mid-transition."""
        return self._matches(claim) or (self._old_sig is not None and self._matches(claim, self._old_sig))

    def _draw_claim(self, t: int) -> dict:
        fraud = bool(self.rng.random() < self.fraud_share)
        sig = self.signature
        if fraud and self._handover:
            start, end = self._handover
            p_new = min(1.0, (t - start + 1) / max(1, end - start))  # traffic mixes from old to new
            if self.rng.random() >= p_new:
                sig = self._old_sig
        while True:
            claim = {a: str(self.rng.choice(vals)) for a, vals in self.attrs.items()}
            if fraud:
                claim.update(sig)
                return claim
            if not self._is_fraud_pattern(claim):
                return claim         # legitimate claims never match a fraudulent signature

    def _affected(self, *sigs) -> list:
        combos = itertools.product(self.types, CHANNELS, AMOUNTS)
        return [list(k) for k in combos
                if any(all(dict(zip(("type", "channel", "amount"), k)).get(a) == v for a, v in s.items()) for s in sigs)]

    def _record(self, t, kind, desc, affected=None, **extra):
        ch = {"t": t, "kind": kind, "desc": desc, "affected": affected or [], **extra}
        self.changes.append(ch)
        self._new.append(ch)
        return desc

    def _shift_signature(self, t: int, new: dict, kind: str, why: str):
        sigs = [self.signature, new] + ([self._old_sig] if self._old_sig else [])
        self._old_sig, self._handover = None, None  # an abrupt shift supersedes any handover in progress
        old, self.signature = self.signature, new
        sig_txt = ", ".join(f"{a}={v}" for a, v in new.items())
        return self._record(t, kind, f"the fraud ring shifted tactics: fraudulent claims now look like ({sig_txt}) — {why}",
                            affected=self._affected(*sigs), old_sig=old, new_sig=new)

    def _retire_old_signature(self, t: int):
        if not self._handover or t < self._handover[1]:
            return
        old, self._old_sig, self._handover = self._old_sig, None, None
        sig_txt = ", ".join(f"{a}={v}" for a, v in old.items())
        self._record(t, "latent", f"the old fraud pattern ({sig_txt}) has stopped circulating; "
                                  "only the new one remains", affected=self._affected(old), old_sig=old)

    def _adversary_check(self, t: int):
        if not self.endogenous or t < self.warmup or (t + 1) % self.adapt_every:
            return
        window = self.recent[-self.adapt_window:]
        frauds = [r for r in window if r[2]]
        caught = sum(1 for r in frauds if r[1] == "REJECT")
        if not frauds or caught / len(frauds) < 0.5:
            return                   # the current tactic still works; no reason to move
        approved = [r[0] for r in window if r[1] == "APPROVE" and not r[2]]
        if approved:
            mimic = approved[int(self.rng.integers(len(approved)))]
            attrs = list(self.rng.choice(list(self.attrs), size=2, replace=False))
            new = {a: mimic[a] for a in sorted(attrs)}
            why = "they are mimicking the traffic you approve"
        else:
            new, why = self._random_signature(avoid=self.signature), "their claims stopped getting through"
        if new != self.signature:
            self._shift_signature(t, new, "endogenous", why)

    # ---- change capabilities ----------------------------------------------------------
    def change_latent(self, t: int) -> str:
        new = self._random_signature(avoid=self.signature)
        if not self.transition_window:
            return self._shift_signature(t, new, "latent", "a scheduled tactic change")
        # gradual handover: the ring starts running the new playbook while the old one
        # still circulates; the fraud mix shifts toward the new signature over the window
        old, self.signature = self.signature, new
        self._old_sig, self._handover = old, (t, t + self.transition_window)
        new_txt = ", ".join(f"{a}={v}" for a, v in new.items())
        old_txt = ", ".join(f"{a}={v}" for a, v in old.items())
        self._record(t, "latent", f"the fraud ring is migrating tactics: new-style claims look like ({new_txt}); "
                                  f"the old pattern ({old_txt}) still circulates for a while",
                     affected=self._affected(old, new), old_sig=old, new_sig=new)
        return self.changes[-1]["desc"]

    def change_surface(self, t: int) -> str:
        self.template = (self.template + 1) % len(TEMPLATES)
        return self._record(t, "surface", "claim intake re-worded (the fraud pattern is unchanged)")

    def correct(self, claim: dict) -> str:
        return "REJECT" if self._is_fraud_pattern(claim) else "APPROVE"

    def task_key(self, claim: dict) -> tuple:
        return key_of(claim)

    # ---- World protocol ------------------------------------------------------------------
    def observe(self, t: int) -> str:
        self._retire_old_signature(t)
        self._current = self._draw_claim(t)
        self._n += 1
        rng = flavor_rng(self.seed, t)
        figure = int(rng.integers(120, 950)) if self._current["amount"] == "small" else int(rng.integers(3800, 24000))
        header = (f"Claim CLM-{7000 + self._n} — claimant {person(rng)}, "
                  f"policy MA-{int(rng.integers(10_000, 99_999))}, {self._current['type']} cover.")
        incident = str(rng.choice(INCIDENTS[self._current["type"]])).format(rng.choice(DAYS))
        text = TEMPLATES[self.template].format(header=header, incident=incident, figure=figure,
                                               adjuster=rng.choice(ADJUSTER_NOTES),
                                               channel_phrase=CHANNEL_PHRASE[self._current["channel"]],
                                               **self._current)
        return f"{text}\nDo you approve or reject this claim?"

    def parse(self, text: str):
        c = re.search(r"CONFIDENCE:\s*(\d{1,3})", text)
        self._last_conf = min(float(c.group(1)), 100.0) / 100 if c else None
        m = re.search(r"DECISION:\s*(APPROVE|REJECT)", text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        if self.consult and re.search(r"^ASK:", text, re.M):
            return "ASK"
        return None

    def default_action(self, rng):
        return str(rng.choice(["APPROVE", "REJECT"]))

    def describe_action(self, action) -> str:
        return str(action)

    def act(self, t: int, action: str) -> tuple[float, str]:
        fraud = self._is_fraud_pattern(self._current)
        if action == "ASK":
            self._asked = True
            self.n_asks += 1
            # a colleague settles it correctly; the ring still sees the outcome as traffic
            self.recent.append((self._current, self.correct(self._current), fraud))
            self._adversary_check(t)
            return 0.0, ("Audit desk: this claim is fraudulent. A colleague took over and rejected it; "
                         "no credit for the step." if fraud else
                         "Audit desk: this claim is legitimate. A colleague took over and approved it; "
                         "no credit for the step.")
        ok = action == self.correct(self._current)
        self.recent.append((self._current, action, fraud))
        self._adversary_check(t)
        if ok:
            fb = ("Rejected; the audit confirmed the claim was fraudulent." if fraud
                  else "Approved; the claim settled without issue.")
        else:
            fb = ("Approved, but the audit later found the claim fraudulent." if fraud
                  else "Rejected, but the customer appealed and the claim was valid.")
        return float(ok), fb

    def privileged(self, t: int) -> dict:
        claim = self._current
        new, self._new = self._new, []
        asked, self._asked = self._asked, False
        return {"task": claim, "key": list(key_of(claim)), "task_key": list(key_of(claim)),
                "correct": self.correct(claim), "is_fraud": self._is_fraud_pattern(claim),
                "signature": dict(self.signature), "confidence": self._last_conf, "asked": asked,
                "retiring_signature": dict(self._old_sig) if self._old_sig else None,
                "affected_now": any(key_of(claim) in [tuple(a) for a in c["affected"]] for c in self.changes[-2:]),
                "changes": new}

    def summary(self) -> dict:
        frauds = [r for r in self.recent if r[2]]
        return {"changes": self.changes, "n_asks": self.n_asks,
                "n_endogenous": sum(1 for c in self.changes if c["kind"] == "endogenous"),
                "fraud_caught": sum(1 for r in frauds if r[1] == "REJECT") / len(frauds) if frauds else None}
