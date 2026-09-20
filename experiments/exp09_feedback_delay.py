"""Experiment 9 — Feedback timing (environment engineering).

The outcome of step t is only revealed at step t+k, for k in 0, 1, 3, 6, in a
stable world and a changing one.

Hypothesis: learning speed degrades quickly with delay; how quickly depends on
whether the agent can re-associate a late outcome with the task that caused it.
Works on: any world.

    python -m experiments.exp09_feedback_delay [--world form_filler]
    python -m experiments.exp09_feedback_delay --analyze
"""

import numpy as np

from experiments.common import NOTES, TRANSCRIPT, q, ref, run_experiment, scheduled, spaced, world_for
from driftlab.core import Scenario
from driftlab.metrics import collect_steps, print_table, rolling_mean, summarize_by

T = q(120)
DELAYS = [0, 1, 3, 6]
WORLD_REGIMES = {"stable": 0, "changing": 6}
REFERENCE_AGENTS = [ref("llm_transcript", TRANSCRIPT), ref("llm_notes", NOTES)]


def cells(seeds):
    return [{"regime": f"{wr}_delay{d}", "world_regime": wr, "delay": d, "seed": s}
            for wr in WORLD_REGIMES for d in DELAYS for s in range(seeds)]


def scenario(cell, ctx):
    world = world_for(cell, T, needs=("change_latent",))
    return Scenario(world, T=T, before_step=scheduled(latent=spaced(WORLD_REGIMES[cell["world_regime"]], T, q(20))),
                    feedback_delay=cell["delay"], seed=cell["seed"])


def analyze(run_dir):
    rows = []
    for h, steps in collect_steps(run_dir):
        c = h["cell"]
        r = [s["reward"] for s in steps]
        roll = rolling_mean(r, 10)
        target = 0.7 * max(r) if r else 0.7
        ttc = next((i for i, v in enumerate(roll) if not np.isnan(v) and v >= target), float("nan"))
        rows.append({"agent": c["agent"]["name"], "world": c["world_regime"], "delay": c["delay"],
                     "acc": float(np.mean(r[q(20):])), "auc": float(np.nanmean(roll)), "steps_to_70pct": float(ttc)})
    f = ("acc", "auc", "steps_to_70pct")
    print_table(summarize_by(rows, ("agent", "world", "delay"), f), ("agent", "world", "delay"), f, width=13)


if __name__ == "__main__":
    run_experiment("exp09", __doc__, cells, scenario, REFERENCE_AGENTS, analyze, config={"T": T, "delays": DELAYS})
