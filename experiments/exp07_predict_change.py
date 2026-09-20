"""Experiment 7 — Learning the dynamics of change itself.

Three change schedules with the same number of changes:
    jittered   irregular (control)
    periodic   at exactly regular intervals
    triggered  RuleWorld only: a change right after the fourth "renewal" request since the last one

Hypothesis: with a learnable schedule the dip after later changes is smaller
than after early ones (anticipation).
Works on: any world (jittered, periodic); triggered needs rule_world.

    python -m experiments.exp07_predict_change [--world inventory]
    python -m experiments.exp07_predict_change --analyze
"""

import numpy as np

from experiments.common import NOTES, WORLD, q, ref, run_experiment, scheduled, spaced, world_for
from driftlab.core import Scenario
from driftlab.metrics import change_events, collect_steps, dip_after, print_table, summarize_by
from driftlab.runner import env_cells

T, N_CHANGES = q(160), 7
REFERENCE_AGENTS = [ref("llm_notes", NOTES)]


def cells(seeds):
    regimes = ["jittered", "periodic"] + (["triggered"] if WORLD == "rule_world" else [])
    return env_cells(regimes, seeds)


def scenario(cell, ctx):
    regime = cell["regime"]
    world = world_for(cell, T, needs=("change_latent",))
    if regime == "periodic":
        return Scenario(world, T=T, before_step=scheduled(latent=spaced(N_CHANGES, T, q(20))), seed=cell["seed"])
    if regime == "jittered":
        rng = np.random.default_rng(cell["seed"] + 99)
        steps = sorted(int(s + rng.integers(-q(6), q(6) + 1)) for s in spaced(N_CHANGES, T, q(20)))
        return Scenario(world, T=T, before_step=scheduled(latent=[max(1, s) for s in steps]), seed=cell["seed"])
    state = {"renewals": 0, "changes": 0}

    def before(t, w):
        if state["changes"] < N_CHANGES and state["renewals"] >= q(4):
            w.change_latent(t)
            state["renewals"], state["changes"] = 0, state["changes"] + 1
        if w.task(t)["type"] == "renewal":
            state["renewals"] += 1
    return Scenario(world, T=T, before_step=before, seed=cell["seed"])


def analyze(run_dir, window=6):
    rows = []
    for h, steps in collect_steps(run_dir):
        r = [s["reward"] for s in steps]
        muts = sorted(change_events(steps), key=lambda m: m["t"])
        if len(muts) < 4:
            continue
        dips = [dip_after(r, m["t"], window) for m in muts]
        half = len(dips) // 2
        rows.append({"agent": h["cell"]["agent"]["name"], "regime": h["cell"]["regime"],
                     "dip_early": float(np.mean(dips[:half])), "dip_late": float(np.mean(dips[half:])),
                     "anticipation": float(np.mean(dips[:half]) - np.mean(dips[half:]))})
    f = ("dip_early", "dip_late", "anticipation")
    print("anticipation = early dip - late dip (positive: the agent got better at absorbing changes)")
    print_table(summarize_by(rows, ("agent", "regime"), f), ("agent", "regime"), f)


if __name__ == "__main__":
    run_experiment("exp07", __doc__, cells, scenario, REFERENCE_AGENTS, analyze, config={"T": T, "n_changes": N_CHANGES})
