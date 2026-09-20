"""Experiment 19 — Can agents assign credit through pooled, delayed reports?

The campaign desk pools all feedback: no push ever gets an individual outcome,
only a periodic total. The regimes vary how often the report arrives (every 5,
10 or 20 pushes); the agents vary only in memory (none, transcript, notes).
Two audience shifts land mid-run, so the agent must re-learn under pooling.

Hypothesis: with only pooled reports, memory is not an optimization but the
precondition for learning anything: a memoryless agent has nothing to connect
a report to and stays at chance, while an agent with notes can correlate its
own recorded choices with the totals. The gap should widen as reports get
sparser. Works on: campaign_desk (pinned; pooled feedback lives there).

    python -m experiments.exp19_pooled_feedback
    python -m experiments.exp19_pooled_feedback --analyze
"""

from collections import defaultdict

import numpy as np

from experiments.common import NONE, NOTES, TRANSCRIPT, q, ref, run_experiment, scheduled, spaced, world_for
from driftlab.core import Scenario
from driftlab.runner import env_cells
from driftlab.metrics import collect_steps

T = q(120)
REGIMES = {"report_5": 5, "report_10": 10, "report_20": 20}
N_SHIFTS = 2
REFERENCE_AGENTS = [ref("llm_none", NONE), ref("llm_transcript", TRANSCRIPT), ref("llm_notes", NOTES)]


def cells(seeds):
    return env_cells(list(REGIMES), seeds, world="campaign_desk")


def scenario(cell, ctx):
    world = world_for(cell, T, needs=("change_latent",), report_every=q(REGIMES[cell["regime"]]))
    return Scenario(world, T=T, before_step=scheduled(latent=spaced(N_SHIFTS, T, q(15))), seed=cell["seed"])


def analyze(run_dir):
    groups = defaultdict(list)
    for h, steps in collect_steps(run_dir):
        if not h or not steps or "reward" not in steps[0]:
            continue
        tail = steps[q(20):]
        groups[(h["cell"]["agent"]["name"], h["cell"]["regime"])].append({
            "engagement": float(np.mean([1.0 if s.get("hit") else 0.0 for s in tail])),
            "best_pick": float(np.mean([1.0 if s.get("angle") == s.get("correct") else 0.0 for s in tail]))})
    print(f"{'agent':<16}{'regime':<12}{'true engagement':>16}{'picked best':>13}")
    print("-" * 60)
    for (a, r), rs in sorted(groups.items()):
        m = lambda k: float(np.nanmean([x[k] for x in rs]))  # noqa: E731
        print(f"{a:<16}{r:<12}{m('engagement'):>16.3f}{m('best_pick'):>13.3f}")
    print("\n(true engagement comes from the privileged per-push log the agent never saw;"
          "\n random play sits near 0.46 engagement and 0.25 on picking the best angle)")


if __name__ == "__main__":
    run_experiment("exp19", __doc__, cells, scenario, REFERENCE_AGENTS, analyze,
                   config={"T": T, "report_every": dict(REGIMES), "n_shifts": N_SHIFTS})
