"""Experiment 5 — Forgetting on purpose.

Three drift rates (0, 4, 12 substantive changes). The reference agents vary how
much they retain: transcript windows of 5, 15, 40, 120 steps and notes capped
at 300 or 1500 characters. An external harness is one agent here.

Hypothesis: the best retention setting moves with drift. At high drift a short
memory wins because old evidence has expired; at zero drift the longest wins.
Works on: any world.

    python -m experiments.exp05_forgetting_window [--world form_filler]
    python -m experiments.exp05_forgetting_window --analyze
"""

import numpy as np

from experiments.common import q, recurring_caseload, ref, run_experiment, scheduled, spaced, world_for
from driftlab.core import Scenario
from driftlab.metrics import collect_steps, print_table, summarize_by
from driftlab.runner import env_cells

T = q(120)
DRIFT = {"drift_none": 0, "drift_low": 4, "drift_high": 12}
REFERENCE_AGENTS = ([ref(f"transcript_{w}", {"kind": "transcript", "window": w}) for w in (5, 15, 40, 120)]
                    + [ref(f"notes_{b}", {"kind": "notes", "every": 10, "budget_chars": b}) for b in (300, 1500)])


def cells(seeds):
    return env_cells(list(DRIFT), seeds)


def scenario(cell, ctx):
    world = world_for(cell, T, needs=("change_latent",))
    if hasattr(world, "truth_table"):  # v2: a recurring caseload, so a 5-step window still meets repeat cases
        world.task_fn = recurring_caseload(cell["seed"])
    return Scenario(world, T=T, before_step=scheduled(latent=spaced(DRIFT[cell["regime"]], T, q(20))), seed=cell["seed"])


def analyze(run_dir):
    rows = [{"agent": h["cell"]["agent"]["name"], "regime": h["cell"]["regime"],
             "acc": float(np.mean([s["reward"] for s in steps[q(20):]]))} for h, steps in collect_steps(run_dir)]
    summ = summarize_by(rows, ("agent", "regime"), ("acc",))
    print_table(summ, ("agent", "regime"), ("acc",))
    print("\nBest agent setting per drift rate:")
    for regime in DRIFT:
        best = max(((a, s["acc"]) for (a, r), s in summ.items() if r == regime), key=lambda x: x[1], default=None)
        if best:
            print(f"  {regime:<12} {best[0]} ({best[1]:.3f})")


if __name__ == "__main__":
    run_experiment("exp05", __doc__, cells, scenario, REFERENCE_AGENTS, analyze, config={"T": T, "drift": DRIFT})
