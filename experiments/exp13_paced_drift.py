"""Experiment 13 — Pacing the drift to the learner (environment engineering).

Same number of substantive changes (6), different timing:
    fixed        evenly spaced, regardless of the learner
    adaptive     a change only once rolling reward is back above its running
                 peak fraction and at least 10 steps have passed
    adversarial  a change as soon as reward recovers (min gap 3)

Hypothesis: adaptive pacing yields a better final learner than fixed pacing
with the same number of changes.
Works on: any world.

    python -m experiments.exp13_paced_drift [--world inventory]
    python -m experiments.exp13_paced_drift --analyze
"""

import numpy as np

from experiments.common import NOTES, q, ref, run_experiment, scheduled, spaced, world_for
from driftlab.core import Scenario
from driftlab.metrics import change_events, collect_steps, lags_generic, print_table, summarize_by
from driftlab.runner import env_cells

T, N_CHANGES = q(150), 6
PACING = {"fixed": None, "adaptive": (0.7, q(10)), "adversarial": (0.7, q(3))}
REFERENCE_AGENTS = [ref("llm_notes", NOTES)]


def cells(seeds):
    return env_cells(list(PACING), seeds)


def scenario(cell, ctx):
    pacing = PACING[cell["regime"]]
    world = world_for(cell, T, needs=("change_latent",))
    if pacing is None:
        return Scenario(world, T=T, before_step=scheduled(latent=spaced(N_CHANGES, T, q(20))), seed=cell["seed"])
    rewards, state = [], {"last": q(15), "changes": 0}

    def before(t, w):
        if state["changes"] < N_CHANGES and t - state["last"] >= pacing[1] and len(rewards) >= q(10):
            recent, peak = np.mean(rewards[-q(10):]), max(np.mean(rewards[i:i + q(10)]) for i in range(0, len(rewards) - q(10) + 1))
            if peak > 0 and recent >= pacing[0] * peak:
                w.change_latent(t)
                state["last"], state["changes"] = t, state["changes"] + 1

    async def after(t, rec, agent):
        rewards.append(rec["reward"])
        return {}
    return Scenario(world, T=T, before_step=before, after_step=after, seed=cell["seed"])


def analyze(run_dir):
    rows = []
    for h, steps in collect_steps(run_dir):
        r = [s["reward"] for s in steps]
        lags = [l for l in (lags_generic(steps, m)["recovery_lag"] for m in change_events(steps)) if l == l]
        rows.append({"agent": h["cell"]["agent"]["name"], "pacing": h["cell"]["regime"],
                     "reward_last20": float(np.mean(r[-q(20):])), "reward_overall": float(np.mean(r[q(20):])),
                     "changes_done": float(len(change_events(steps))),
                     "recovery_lag": float(np.mean(lags)) if lags else float("nan")})
    f = ("reward_last20", "reward_overall", "changes_done", "recovery_lag")
    print_table(summarize_by(rows, ("agent", "pacing"), f), ("agent", "pacing"), f)


if __name__ == "__main__":
    run_experiment("exp13", __doc__, cells, scenario, REFERENCE_AGENTS, analyze, config={"T": T, "n_changes": N_CHANGES})
