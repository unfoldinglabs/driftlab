"""Caused, correlated drift: an organization whose changes have reasons.

Scheduled drift flips independent facts on a timer. Real drift is coherent —
a reorg moves several routing rules at once, a policy push touches related
fields together — and it can respond to the agent. `OrganizationProcess` is a
seeded latent process over named cause channels; pressure on each channel
accumulates stochastically (and through `signal()`, from the agent's own
behavior) until the channel fires and emits a burst of related changes in one
step, all stamped with the same cause id.

Worlds that implement `apply_intent(t, intent)` translate a firing into their
own coherent mechanics (RuleWorld dissolves a desk); any other world gets the
fallback — a burst of its ordinary change_latent/change_surface calls — and
still records the shared cause. Cause ids land in the change records, so
whether an agent infers the common cause behind co-occurring changes is
measurable from the logs like everything else.

    from experiments.common import organization
    Scenario(world, T=T, before_step=organization(rate=0.05), seed=...)
"""

from dataclasses import dataclass, field

import numpy as np

CHANNELS = ("reorg", "policy", "workload", "tooling")


@dataclass
class OrganizationProcess:
    seed: int
    rate: float = 0.04          # mean per-step pressure gain; higher = more frequent firings
    coherence: int = 3          # a firing emits 1..coherence related changes
    surface_share: float = 0.4  # chance a firing also re-words surfaces (same cause)
    warmup: int = 15
    channels: tuple = CHANNELS

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed + 424_242)
        self.pressure = {c: float(self.rng.uniform(0.0, 0.5)) for c in self.channels}
        self.fired = 0

    def signal(self, channel: str, amount: float = 0.1):
        """Agent-driven pressure: worlds or scenarios can report behavior that pushes a channel."""
        if channel in self.pressure:
            self.pressure[channel] += amount

    def step(self, t: int, world) -> str | None:
        """Advance the latent state; maybe fire one cause into the world. Call from before_step."""
        if t < self.warmup:
            return None
        for c in self.channels:
            self.pressure[c] += float(self.rng.exponential(self.rate))
        channel = max(self.pressure, key=self.pressure.get)
        if self.pressure[channel] < 1.0:
            return None
        self.pressure[channel] = float(self.rng.uniform(0.0, 0.3))
        self.fired += 1
        cause = f"{channel}#{self.fired}"
        n = 1 + int(self.rng.integers(self.coherence))
        before = len(world.changes)
        if callable(getattr(world, "apply_intent", None)):
            world.apply_intent(t, {"cause": cause, "channel": channel, "n": n})
        else:
            for _ in range(n):
                world.change_latent(t)
            if self.rng.random() < self.surface_share and callable(getattr(world, "change_surface", None)):
                world.change_surface(t)
        for ch in world.changes[before:]:
            ch.setdefault("cause", cause)
        return cause
