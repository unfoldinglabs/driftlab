"""Experiment 11 — Warning shots (environment engineering).

Before each substantive change the environment can post a memo ("heads-up: a
procedure change takes effect soon"). Conditions vary lead time (0 = none, 1, 3
steps) and false alarms (about half the changes also come with a memo about
something that will NOT change). On RuleWorld the memo names the request type.

Hypothesis: reliable warnings cut recovery lag; false alarms cost more than
they save because the agent second-guesses things that did not change.
Works on: any world.

    python -m experiments.exp11_warning_shots [--world codebase]
    python -m experiments.exp11_warning_shots --analyze
"""

import numpy as np

from experiments.common import NOTES, q, qlist, ref, run_experiment, world_for
from driftlab.core import Scenario
from driftlab.metrics import change_events, collect_steps, dip_after, lags_generic, print_table, summarize_by
from driftlab.runner import env_cells

T = q(120)
CHANGES = qlist([30, 50, 70, 90, 110])
CONDITIONS = {"no_warning": (0, 0.0), "lead1": (1, 0.0), "lead3": (3, 0.0), "lead3_false50": (3, 0.5)}
REFERENCE_AGENTS = [ref("llm_notes", NOTES)]


def cells(seeds):
    return env_cells(list(CONDITIONS), seeds)


def scenario(cell, ctx):
    lead, false_rate = CONDITIONS[cell["regime"]]
    world = world_for(cell, T, needs=("change_latent",))
    rng = np.random.default_rng(cell["seed"] + 5)
    is_rule = getattr(world, "name", "") == "rule_world"
    planned = {}
    if is_rule:
        keys = world.mutable_keys()
        planned = {ms: keys[rng.integers(len(keys))] for ms in CHANGES}
    false_alarms = {ms - lead - q(5): (str(rng.choice([ty for ty in world.types if ty != planned[ms][0]])) if is_rule else None)
                    for ms in CHANGES if rng.random() < false_rate}

    def before(t, w):
        if t in CHANGES:
            if is_rule:
                w.mutate_now(t, planned[t])
            else:
                w.change_latent(t)

    def memo(subject):
        return (f"Memo from operations: a procedure change affecting {subject} requests takes effect soon."
                if subject else "Memo from operations: a procedure change affecting your work takes effect soon.")

    def notice(t, w):
        msgs = [memo(planned[ms][0] if is_rule else None) for ms in CHANGES if lead and t == ms - lead]
        if t in false_alarms:
            msgs.append(memo(false_alarms[t]))
        return " ".join(msgs) or None

    return Scenario(world, T=T, before_step=before, notice_fn=notice, seed=cell["seed"],
                    tags={"false_alarm_steps": {str(k): v for k, v in false_alarms.items()}})


def analyze(run_dir):
    rows = []
    for h, steps in collect_steps(run_dir):
        r = [s["reward"] for s in steps]
        lags = [l for l in (lags_generic(steps, m)["recovery_lag"] for m in change_events(steps)) if l == l]
        fa = [dip_after(r, int(t_str), 8) for t_str in h.get("false_alarm_steps", {})]
        rows.append({"agent": h["cell"]["agent"]["name"], "condition": h["cell"]["regime"], "acc": float(np.mean(r[q(20):])),
                     "recovery_lag": float(np.mean(lags)) if lags else float("nan"),
                     "false_alarm_dip": float(np.mean(fa)) if fa else float("nan")})
    f = ("acc", "recovery_lag", "false_alarm_dip")
    print_table(summarize_by(rows, ("agent", "condition"), f), ("agent", "condition"), f)


if __name__ == "__main__":
    run_experiment("exp11", __doc__, cells, scenario, REFERENCE_AGENTS, analyze, config={"T": T, "changes": CHANGES, "conditions": CONDITIONS})
