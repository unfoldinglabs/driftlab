"""Experiment 16 — Do agents over-update on misleading feedback?

Three regimes engineered so a similar surprise means different things:
    abrupt        4 real rule changes at 4 moments; a surprise usually means the world moved
    gradual       12 real rule changes spread evenly; change is constant but small
    noisy_stable  no changes at all, but 15% of feedback is misleadingly inverted;
                  a surprise means nothing

The same model plays with different memories (short and long transcripts, notes,
structured beliefs, and a fast+slow mixture), stating a confidence with every
action. A learner that reads the dynamics updates differently per regime; a
learner that only reads the surprise over-updates under noise and under-updates
under change, and a longer memory makes it more confidently wrong.

Works on: rule_world (the over-update measure needs per-task ground truth).

    python -m experiments.exp16_update_policy
    python -m experiments.exp16_update_policy --analyze
"""

from collections import defaultdict

import numpy as np

from experiments.common import NOTES, q, recurring_caseload, ref, run_experiment, scheduled, spaced, world_for
from driftlab.core import Scenario
from driftlab.metrics import collect_steps
from driftlab.profile import run_profile
from driftlab.runner import env_cells

T = q(120)
REGIMES = {"abrupt": {"n": 4, "noise": 0.0}, "gradual": {"n": 12, "noise": 0.0},
           "noisy_stable": {"n": 0, "noise": 0.15}}
REFERENCE_AGENTS = [ref("transcript_5", {"kind": "transcript", "window": 5}),
                    ref("transcript_120", {"kind": "transcript", "window": 120}),
                    ref("llm_notes", NOTES),
                    ref("llm_beliefs", {"kind": "beliefs", "every": 10}),
                    ref("fast_slow", {"kind": "fastslow", "window": 10, "every": 10})]


def cells(seeds):
    return env_cells(list(REGIMES), seeds)


def scenario(cell, ctx):
    r = REGIMES[cell["regime"]]
    world = world_for(cell, T, needs=("change_latent", "task_key", "ask_confidence"),
                      ask_confidence=True, feedback_noise=r["noise"])
    world.task_fn = recurring_caseload(cell["seed"])  # v2: repeat cases, so short windows and over-update events exist
    return Scenario(world, T=T, before_step=scheduled(latent=spaced(r["n"], T, q(20))), seed=cell["seed"])


def analyze(run_dir):
    groups = defaultdict(list)
    for h, steps in collect_steps(run_dir):
        if not h or not steps or "reward" not in steps[0]:
            continue
        p = run_profile(h, steps)
        groups[(p["agent"], p["regime"])].append(p)
    fmt = lambda v: "     —" if v is None or np.isnan(v) else f"{v:6.3f}"  # noqa: E731
    print(f"{'agent':<16}{'regime':<14}{'final acc':>10}{'over-upd':>10}{'overconf':>10}{'recovery':>10}")
    print("-" * 70)
    for (a, r), ps in sorted(groups.items()):
        m = lambda k: float(np.nanmean([p[k] for p in ps]))  # noqa: E731
        print(f"{a:<16}{r:<14}{fmt(m('final_accuracy')):>10}{fmt(m('overupdate_rate')):>10}"
              f"{fmt(m('overconfidence')):>10}{fmt(m('recovery_lag')):>10}")
    print("\n(final acc counts the world's reported rewards, so the noisy regime tops out at ~0.85;"
          "\n over-upd only exists where feedback can mislead; recovery only exists where rules change)")


if __name__ == "__main__":
    run_experiment("exp16", __doc__, cells, scenario, REFERENCE_AGENTS, analyze,
                   config={"T": T, "regimes": {k: dict(v) for k, v in REGIMES.items()}})
