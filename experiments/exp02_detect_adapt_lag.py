"""Experiment 2 — Detecting a change and adapting to it are two different skills.

Six substantive changes on a schedule (rule flips, schema mutations, demand
jumps, style-guide revisions, depending on the world). After each change,
measure how long until the agent stops doing the old thing (detection), first
does the right thing (recovery), and does it twice in a row (stable). Worlds
that repeat identical tasks (RuleWorld) give encounter-based lags; others give
time-based recovery.

Hypothesis: detection and recovery come apart depending on how the agent remembers.
Works on: any world.

    python -m experiments.exp02_detect_adapt_lag [--world inventory]
    python -m experiments.exp02_detect_adapt_lag --analyze
"""

from experiments.common import NONE, NOTES, SKILLS, TRANSCRIPT, q, ref, run_experiment, scheduled, spaced, world_for
from driftlab.core import Scenario
from driftlab.metrics import change_events, collect_steps, lags_generic, print_table, summarize_by
from driftlab.runner import env_cells

T, N_CHANGES = q(120), 6
REFERENCE_AGENTS = [{"name": "tabular", "type": "tabular"}, ref("llm_none", NONE), ref("llm_transcript", TRANSCRIPT),
                    ref("llm_notes", NOTES), ref("llm_skills", SKILLS)]


def cells(seeds):
    return env_cells(["changing"], seeds)


def scenario(cell, ctx):
    world = world_for(cell, T, needs=("change_latent",))
    return Scenario(world, T=T, before_step=scheduled(latent=spaced(N_CHANGES, T, q(20))), seed=cell["seed"])


def analyze(run_dir):
    rows = []
    for h, steps in collect_steps(run_dir):
        for m in change_events(steps):
            lag = lags_generic(steps, m)
            if lag["n_encounters"] >= 3 or lag["n_encounters"] == 0:
                rows.append({"agent": h["cell"]["agent"]["name"], **lag, "gap": lag["recovery_lag"] - lag["detection_lag"]})
    f = ("detection_lag", "recovery_lag", "stable_lag", "gap")
    print("Lags after a change (encounters on affected tasks where the world has task keys, else steps; lower is better)")
    print_table(summarize_by(rows, ("agent",), f), ("agent",), f)


if __name__ == "__main__":
    run_experiment("exp02", __doc__, cells, scenario, REFERENCE_AGENTS, analyze, config={"T": T})
