"""Experiment 20 — Does prediction-gated revision reduce over-updating?

The same model plays with three memory designs of increasing structure:
free-form notes, a structured belief file, and an explicit world model — rule
hypotheses with validity conditions, where the agent predicts every outcome
(an EXPECT: line) and a wrong prediction triggers an immediate revision
instead of waiting for the next rewrite interval.

Two regimes make the comparison sharp: rules that really change four times,
and a world that never changes but lies in 15% of its feedback. An agent that
tracks dynamics explicitly should shrug off the lies (a low over-update rate)
without adapting any slower to real change. The wrong-prediction rate itself
is measured from the logged EXPECT lines.

Works on: rule_world (the over-update measure needs per-task ground truth).

    python -m experiments.exp20_world_model
    python -m experiments.exp20_world_model --analyze
"""

from collections import defaultdict

import numpy as np

from experiments.common import NOTES, q, recurring_caseload, ref, run_experiment, scheduled, spaced, world_for
from driftlab.core import Scenario
from driftlab.metrics import collect_steps
from driftlab.profile import run_profile
from driftlab.runner import env_cells

T = q(120)
REGIMES = {"abrupt": {"n": 4, "noise": 0.0}, "noisy_stable": {"n": 0, "noise": 0.15}}
REFERENCE_AGENTS = [ref("llm_notes", NOTES),
                    ref("llm_beliefs", {"kind": "beliefs", "every": 10}),
                    ref("llm_worldmodel", {"kind": "worldmodel", "every": 10})]


def cells(seeds):
    return env_cells(list(REGIMES), seeds)


def scenario(cell, ctx):
    r = REGIMES[cell["regime"]]
    world = world_for(cell, T, needs=("change_latent", "task_key"), feedback_noise=r["noise"])
    world.task_fn = recurring_caseload(cell["seed"])  # repeat cases, so over-update events exist
    return Scenario(world, T=T, before_step=scheduled(latent=spaced(r["n"], T, q(20))), seed=cell["seed"])


def analyze(run_dir):
    groups = defaultdict(list)
    for h, steps in collect_steps(run_dir):
        if not h or not steps or "reward" not in steps[0]:
            continue
        p = run_profile(h, steps)
        groups[(p["agent"], p["regime"])].append(p)
    fmt = lambda v: "     —" if v is None or np.isnan(v) else f"{v:6.3f}"  # noqa: E731
    print(f"{'agent':<18}{'regime':<14}{'final acc':>10}{'over-upd':>10}{'recovery':>10}{'wrong pred':>11}")
    print("-" * 73)
    for (a, r), ps in sorted(groups.items()):
        m = lambda k: float(np.nanmean([p[k] for p in ps]))  # noqa: E731
        print(f"{a:<18}{r:<14}{fmt(m('final_accuracy')):>10}{fmt(m('overupdate_rate')):>10}"
              f"{fmt(m('recovery_lag')):>10}{fmt(m('prediction_wrong_rate')):>11}")
    print("\n(over-upd only exists under lying feedback; recovery only where rules change;"
          "\n wrong pred only for agents that state EXPECT lines — the worldmodel arm)")


if __name__ == "__main__":
    run_experiment("exp20", __doc__, cells, scenario, REFERENCE_AGENTS, analyze,
                   config={"T": T, "regimes": {k: dict(v) for k, v in REGIMES.items()}})
