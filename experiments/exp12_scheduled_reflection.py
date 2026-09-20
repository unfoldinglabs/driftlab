"""Experiment 12 — Scheduled reflection: when should an agent consolidate?

The variable is the reference agents' reflection schedule: notes rewritten
every 5 / 10 / 20 steps, right after any failure, both, or never. The world
has six substantive changes on a schedule. An external harness reflects
however it likes and gets one row.

Hypothesis: event-triggered reflection beats time-triggered at equal cost,
because failures cluster right after changes. Scored as accuracy per dollar.
Works on: any world.

    python -m experiments.exp12_scheduled_reflection [--world inventory]
    python -m experiments.exp12_scheduled_reflection --analyze
"""

import numpy as np

from experiments.common import q, ref, run_experiment, scheduled, spaced, world_for
from driftlab.core import Scenario
from driftlab.metrics import change_events, collect_steps, lags_generic, print_table, summarize_by
from driftlab.runner import env_cells

T = q(120)
TRIGGERS = {"interval_5": {"trigger": "interval", "every": 5}, "interval_10": {"trigger": "interval", "every": 10},
            "interval_20": {"trigger": "interval", "every": 20}, "failure": {"trigger": "failure", "every": 10},
            "both": {"trigger": "both", "every": 10}, "never": {"trigger": "never", "every": 10}}
REFERENCE_AGENTS = [ref(name, {"kind": "notes", **cfg}) for name, cfg in TRIGGERS.items()]


def cells(seeds):
    return env_cells(["changing"], seeds)


def scenario(cell, ctx):
    world = world_for(cell, T, needs=("change_latent",))
    return Scenario(world, T=T, before_step=scheduled(latent=spaced(6, T, q(20))), seed=cell["seed"])


def analyze(run_dir):
    rows = []
    for h, steps in collect_steps(run_dir):
        r = [s["reward"] for s in steps]
        lags = [l for l in (lags_generic(steps, m)["recovery_lag"] for m in change_events(steps)) if l == l]
        cost = sum(d["cost_usd"] + d["cache_saved_usd"] for d in h.get("cost", {}).values())
        acc = float(np.mean(r[q(20):]))
        rows.append({"agent": h["cell"]["agent"]["name"], "acc": acc,
                     "recovery_lag": float(np.mean(lags)) if lags else float("nan"),
                     "cost_usd": cost, "acc_per_usd": acc / cost if cost else float("nan")})
    f = ("acc", "recovery_lag", "cost_usd", "acc_per_usd")
    print_table(summarize_by(rows, ("agent",), f), ("agent",), f)


if __name__ == "__main__":
    run_experiment("exp12", __doc__, cells, scenario, REFERENCE_AGENTS, analyze, config={"T": T, "triggers": TRIGGERS})
