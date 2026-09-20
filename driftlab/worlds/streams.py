"""Structured task streams: what arrives, when — owned by the experiment.

Worlds pick their own tasks by default (seeded i.i.d. draws). Real work
arrives with structure: bursts of the same kind, seasonal mixes, a task
distribution that itself drifts. A stream is a seeded `task_fn(t, world)` the
experiment passes into the world (`world_for(cell, T, task_fn=...)`), keeping
the usual division of labor: the world owns what tasks mean, the stream owns
which one shows up.

    from driftlab.worlds.streams import rule_stream, form_stream
    world_for(cell, T, task_fn=rule_stream("bursty", seed=cell["seed"]))
"""

import numpy as np

from .gridtext import REGIONS, SIZES


def _weights(rng, k: int, sharpness: float) -> np.ndarray:
    w = rng.dirichlet(np.ones(k) / sharpness)
    return w / w.sum()


def rule_stream(kind: str = "mixture_drift", seed: int = 0, period: int = 40,
                burst: int = 6, every: int = 25, sharpness: float = 2.0):
    """RuleWorld streams. mixture_drift: the type distribution is redrawn every
    `period` steps. bursty: every `every` steps, `burst` consecutive requests of
    one type. seasonal: two alternating type mixes with period `period`."""
    rng = np.random.default_rng(seed + 31_337)
    state: dict = {}

    def fn(t: int, world) -> dict:
        types = world.types
        r = np.random.default_rng(seed * 100_003 + t)  # per-step choices stay seed-reproducible
        if kind == "mixture_drift":
            epoch = t // period
            if state.get("epoch") != epoch or len(state.get("w", [])) != len(types):
                state["epoch"], state["w"] = epoch, _weights(np.random.default_rng(seed + epoch), len(types), sharpness)
            ty = str(r.choice(types, p=state["w"]))
        elif kind == "bursty":
            in_burst = (t % every) < burst
            if t % every == 0:
                state["focus"] = str(np.random.default_rng(seed + t).choice(types))
            ty = state.get("focus", types[0]) if in_burst and state.get("focus") in types else str(r.choice(types))
        elif kind == "seasonal":
            half = len(types) // 2 or 1
            pool = types[:half] if (t // period) % 2 == 0 else types[half:] or types
            ty = str(r.choice(pool))
        else:
            raise ValueError(f"unknown rule_stream kind {kind!r}")
        return {"type": ty, "region": str(r.choice(REGIONS)), "size": str(r.choice(SIZES))}

    return fn


def form_stream(kind: str = "bursty", seed: int = 0, every: int = 8, burst: int = 3):
    """FormWorld streams over source records. bursty: runs of orders from the same
    country with similar quantities (one customer's campaign). heavy_tail: mostly
    small orders with occasional very large ones (stress on the quantity cap)."""
    from .formfiller import COUNTRIES

    def fn(task_idx: int, world) -> dict:
        r = np.random.default_rng(seed * 9_973 + task_idx)
        rec = {"customer": f"{r.choice(['Ada', 'Bo', 'Cy', 'Dee', 'Eli'])} {r.choice(['Marsh', 'Okafor', 'Reyes', 'Sato'])}",
               "email": f"user{r.integers(100, 999)}@example.com",
               "order_date": f"2026-{r.integers(1, 13):02d}-{r.integers(1, 29):02d}",
               "priority_level": int(r.integers(3))}
        if kind == "bursty":
            block = task_idx // every
            b = np.random.default_rng(seed + block)
            focus_country, focus_qty = str(b.choice(list(COUNTRIES))), int(b.integers(1, 120))
            in_burst = (task_idx % every) < burst
            rec["shipping_country"] = focus_country if in_burst else str(r.choice(list(COUNTRIES)))
            rec["quantity"] = max(1, focus_qty + int(r.integers(-3, 4))) if in_burst else int(r.integers(1, 120))
        elif kind == "heavy_tail":
            rec["shipping_country"] = str(r.choice(list(COUNTRIES)))
            rec["quantity"] = int(r.integers(1, 15)) if r.random() < 0.85 else int(r.integers(150, 600))
        else:
            raise ValueError(f"unknown form_stream kind {kind!r}")
        return rec

    return fn
