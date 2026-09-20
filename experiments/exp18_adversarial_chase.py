"""Experiment 18 — Is recovery slower when an adversary aims the drift?

The claims desk with three sources of change: none (stable control), tactic
shifts on a timer (scheduled control), and tactic shifts driven by the fraud
ring's best response to the agent's own decisions (adversarial). Scheduled and
adversarial regimes see a similar number of shifts; only who aims them differs.

Hypothesis: recovery from an aimed shift is slower than from a random one,
and it does not improve with repetition — the k-th adversarial shift hurts
like the first, because the ring always moves to wherever the agent has just
become confident. If an agent ever starves the signal (varying its approvals
so the ring cannot read them), that shows up as fewer endogenous shifts.
Works on: claims_desk (pinned; the adversary lives there).

    python -m experiments.exp18_adversarial_chase
    python -m experiments.exp18_adversarial_chase --analyze
"""

from collections import defaultdict

import numpy as np

from experiments.common import NOTES, TRANSCRIPT, q, ref, run_experiment, scheduled, spaced, world_for
from driftlab.core import Scenario
from driftlab.metrics import change_events, collect_steps, lags_generic
from driftlab.runner import env_cells

T = q(150)
N_SCHEDULED = 4
REGIMES = ["stable", "scheduled", "adversarial"]
REFERENCE_AGENTS = [ref("llm_transcript", TRANSCRIPT), ref("llm_notes", NOTES)]


def cells(seeds):
    return env_cells(REGIMES, seeds, world="claims_desk")


def scenario(cell, ctx):
    regime = cell["regime"]
    world = world_for(cell, T, needs=("change_latent", "task_key"), endogenous=(regime == "adversarial"))
    before = scheduled(latent=spaced(N_SCHEDULED, T, q(20))) if regime == "scheduled" else None
    return Scenario(world, T=T, before_step=before, seed=cell["seed"])


def analyze(run_dir):
    groups = defaultdict(list)
    for h, steps in collect_steps(run_dir):
        if not h or not steps or "reward" not in steps[0]:
            continue
        lags = [lags_generic(steps, m)["recovery_lag"] for m in change_events(steps)]
        half = len(lags) // 2
        groups[(h["cell"]["agent"]["name"], h["cell"]["regime"])].append({
            "acc": float(np.mean([s["reward"] for s in steps[q(20):]])),
            "n_shifts": len(lags),
            "recov": float(np.nanmean(lags)) if lags else float("nan"),
            "recov_early": float(np.nanmean(lags[:half])) if half else float("nan"),
            "recov_late": float(np.nanmean(lags[half:])) if half else float("nan")})
    fmt = lambda v: "     —" if v is None or np.isnan(v) else f"{v:6.2f}"  # noqa: E731
    print(f"{'agent':<16}{'regime':<13}{'acc':>7}{'shifts':>8}{'recovery':>10}{'early':>8}{'late':>8}")
    print("-" * 70)
    for (a, r), rs in sorted(groups.items()):
        m = lambda k: float(np.nanmean([x[k] for x in rs]))  # noqa: E731
        print(f"{a:<16}{r:<13}{m('acc'):>7.3f}{m('n_shifts'):>8.1f}{fmt(m('recov')):>10}"
              f"{fmt(m('recov_early')):>8}{fmt(m('recov_late')):>8}")
    print("\n(a chase that never ends shows as 'late' recovery no better than 'early';"
          "\n an agent that starves the ring's signal shows as fewer shifts in the adversarial regime)")


if __name__ == "__main__":
    run_experiment("exp18", __doc__, cells, scenario, REFERENCE_AGENTS, analyze,
                   config={"T": T, "n_scheduled": N_SCHEDULED, "regimes": REGIMES})
