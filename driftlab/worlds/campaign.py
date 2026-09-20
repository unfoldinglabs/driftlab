"""CampaignWorld: a campaign desk scored only by the weekly report.

Each day the agent picks one angle for that day's push (price, quality,
urgency, testimonial). Every push either lands or falls flat according to a
hidden per-angle engagement rate — but the agent never sees per-push outcomes.
All it ever gets is a pooled report every `report_every` days: how many of the
period's pushes landed, with no breakdown. Credit assignment over pooled
outcomes is the whole game, which is how most real work is scored.

Per-step feedback carries no signal, so reward-based metrics dilute to noise
here (a report lands on one step in `report_every`). The profile scores this
world against the privileged truth instead: `correct` logs the best angle each
day and success = picking it, and `hit` logs the truth of every push.

With `weekly=True` (the default) audience activity follows a weekday cycle
that scales all angles equally: report totals gain realistic weekly texture
while the best angle is unchanged.

Change capabilities: change_latent (an audience shift: the engagement rates
are re-drawn), change_surface (the angle descriptions are re-worded),
endogenous (fatigue: an angle used relentlessly stops working — the agent's
own repetition degrades it, and rest restores it), ask_confidence.
"""

import re
from dataclasses import dataclass, field

import numpy as np

from .base import CONFIDENCE_SUFFIX, flavor_rng

# signal-free desk chatter: realistic surface, no information about engagement rates
DESK_NOTES = ["Design refreshed the banner art overnight.", "The send window is 10am as usual.",
              "Two unsubscribes came in via support, already processed.", "The list grew by a handful of signups.",
              "No deliverability complaints this morning.", "Legal signed off on all four running angles."]

SYSTEM = ("You run the outreach campaign at Meridian Media. Each day you pick the angle for that day's push. "
          "You never see how a single push performs: results arrive only in a pooled report every week or so, "
          "with no per-day breakdown. Audience tastes change over time. End your reply with a line:\nANGLE: <letter>")

ANGLE_NAMES = [["a price cut", "a quality story", "limited-time urgency", "a customer testimonial"],
               ["a discount offer", "a craftsmanship piece", "an act-now deadline", "a happy-customer quote"]]
LETTERS = ["A", "B", "C", "D"]


@dataclass
class CampaignWorld:
    seed: int
    T: int = 90
    report_every: int = 10
    regime: str = "static"           # static | walk (rates random-walk a little every day)
    fatigue_step: float = 0.12       # endogenous: repetition cost per consecutive use
    fatigue_recovery: float = 0.04   # per day of rest
    fatigue_cap: float = 0.7
    weekly: bool = True              # audience activity follows a weekday cycle (scales all angles equally,
                                     # so the best angle is unchanged; report totals gain realistic texture)
    ask_confidence: bool = False
    endogenous: bool = False         # fatigue; off by default so scheduled experiments stay controlled
    name: str = "campaign_desk"

    system_prompt: str = field(init=False)
    changes: list = field(init=False, default_factory=list)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.system_prompt = SYSTEM + (CONFIDENCE_SUFFIX if self.ask_confidence else "")
        self.rates = self.rng.permutation([0.15, 0.35, 0.55, 0.8])
        self.fatigue = np.zeros(4)
        self.names = 0
        self.period: list = []       # hits since the last report
        self.history: list = []
        self._fatigued: set = set()
        self._new: list = []
        self._last_conf = None
        self._chose: int = 0

    def _record(self, t, kind, desc, **extra):
        ch = {"t": t, "kind": kind, "desc": desc, "affected": [], **extra}
        self.changes.append(ch)
        self._new.append(ch)
        return desc

    def _effective(self, i: int) -> float:
        return float(np.clip(self.rates[i] * (1.0 - self.fatigue[i]), 0.02, 0.98))

    # ---- change capabilities ----------------------------------------------------------
    def change_latent(self, t: int) -> str:
        self.rates = self.rng.permutation([0.15, 0.35, 0.55, 0.8])
        return self._record(t, "latent", "audience shift: what engages people has changed", rates=[float(r) for r in self.rates])

    def change_surface(self, t: int) -> str:
        self.names = (self.names + 1) % len(ANGLE_NAMES)
        return self._record(t, "surface", "the angle descriptions were re-worded (audience behavior is unchanged)")

    def _walk(self):
        if self.regime == "walk":
            self.rates = np.clip(self.rates + self.rng.normal(0, 0.02, size=4), 0.05, 0.95)

    def _update_fatigue(self, chosen: int, t: int):
        if not self.endogenous:
            return
        for i in range(4):
            if i == chosen:
                self.fatigue[i] = min(self.fatigue[i] + self.fatigue_step, self.fatigue_cap)
            else:
                self.fatigue[i] = max(self.fatigue[i] - self.fatigue_recovery, 0.0)
        letter = LETTERS[chosen]
        if self.fatigue[chosen] >= 0.5 and letter not in self._fatigued:
            self._fatigued.add(letter)
            self._record(t, "endogenous", f"the audience is tiring of angle {letter}: repetition has worn it out")
        if self.fatigue[chosen] < 0.3 and letter in self._fatigued:
            self._fatigued.discard(letter)

    # ---- World protocol ------------------------------------------------------------------
    def observe(self, t: int) -> str:
        names = ANGLE_NAMES[self.names]
        menu = "  ".join(f"{letter}) {name}" for letter, name in zip(LETTERS, names))
        note = flavor_rng(self.seed, t).choice(DESK_NOTES)
        weekday = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][t % 7]
        return (f"Day {t + 1}, {weekday}. Morning stand-up note: {note}\n"
                f"Pick today's angle:\n{menu}\nWhich angle do you push today?")

    def parse(self, text: str):
        c = re.search(r"CONFIDENCE:\s*(\d{1,3})", text)
        self._last_conf = min(float(c.group(1)), 100.0) / 100 if c else None
        m = re.search(r"ANGLE:\s*([ABCD])", text, re.IGNORECASE)
        return m.group(1).upper() if m else None

    def default_action(self, rng):
        return str(rng.choice(LETTERS))

    def describe_action(self, action) -> str:
        return str(action)

    WEEK = [1.0, 1.05, 0.95, 1.0, 1.1, 0.7, 0.6]  # Mon..Sun audience activity

    def act(self, t: int, action: str) -> tuple[float, str]:
        self._chose = LETTERS.index(action)
        p = self._effective(self._chose) * (self.WEEK[t % 7] if self.weekly else 1.0)
        hit = bool(self.rng.random() < min(p, 0.98))
        self.period.append(hit)
        self.history.append({"t": t, "angle": action, "hit": hit})
        self._update_fatigue(self._chose, t)
        self._walk()
        if (t + 1) % self.report_every == 0:
            n, k = len(self.period), sum(self.period)
            mean = k / n if n else 0.0
            self.period = []
            return mean, (f"Summary from analytics: {k} of the last {n} pushes landed. "
                          "No per-day breakdown is available.")
        return 0.0, "Push queued. Results arrive in the next report."

    def privileged(self, t: int) -> dict:
        h = self.history[-1]
        new, self._new = self._new, []
        best = int(np.argmax([self._effective(i) for i in range(4)]))
        return {"angle": h["angle"], "hit": bool(h["hit"]), "correct": LETTERS[best],
                "true_rates": [round(self._effective(i), 3) for i in range(4)],
                "confidence": self._last_conf,
                "affected_now": bool(self._fatigued), "changes": new}

    def summary(self) -> dict:
        hits = [h["hit"] for h in self.history]
        return {"changes": self.changes, "mean_engagement": float(np.mean(hits)) if hits else None,
                "n_endogenous": sum(1 for c in self.changes if c["kind"] == "endogenous")}
