"""Experiment 15 — Change the agent causes itself.

Most drift in real work is not scheduled by anyone: it is the environment
reacting to what the agent does. Here the world reacts to the agent
(RuleWorld: a desk the agent overloads sheds work to another desk; Inventory:
big orders strain the supplier and lengthen lead times; Codebase: the reviewer
adopts habits the agent shows consistently; Form: IT adds aliases for field
names the agent keeps using). Three regimes with no scheduled changes:
    stable       no change of any kind
    scheduled    the same expected number of changes, but on a timer, unrelated to the agent
    endogenous   changes only when the agent's own behavior triggers them

Hypotheses: (1) agents recover more slowly from self-caused changes than from
scheduled ones, because nothing external marks the moment of change; (2) an
agent that notices the feedback loop stops triggering it (the count of
endogenous changes falls over the run).
Works on: any world (all four implement endogenous change).

    python -m experiments.exp15_endogenous_drift [--world inventory]
    python -m experiments.exp15_endogenous_drift --analyze
"""

import numpy as np

from experiments.common import NOTES, TRANSCRIPT, q, ref, run_experiment, scheduled, spaced, world_for
from driftlab.core import Scenario
from driftlab.metrics import change_events, collect_steps, lags_generic, print_table, summarize_by
from driftlab.runner import env_cells

T = q(150)
REGIMES = ["stable", "scheduled", "endogenous"]
REFERENCE_AGENTS = [ref("llm_transcript", TRANSCRIPT), ref("llm_notes", NOTES)]


def cells(seeds):
    return env_cells(REGIMES, seeds)


def scenario(cell, ctx):
    regime = cell["regime"]
    world = world_for(cell, T, needs=("change_latent", "endogenous"), endogenous=(regime == "endogenous"))
    before = scheduled(latent=spaced(4, T, q(20))) if regime == "scheduled" else None
    return Scenario(world, T=T, before_step=before, seed=cell["seed"])


def analyze(run_dir):
    rows = []
    for h, steps in collect_steps(run_dir):
        r = [s["reward"] for s in steps]
        ch = change_events(steps)
        lags = [l for l in (lags_generic(steps, m)["recovery_lag"] for m in ch) if l == l]
        endo = [m["t"] for m in ch if m["kind"] == "endogenous"]
        half = T // 2
        rows.append({"agent": h["cell"]["agent"]["name"], "regime": h["cell"]["regime"],
                     "reward": float(np.mean(r[q(20):])), "n_changes": float(len(ch)),
                     "endo_first_half": float(sum(1 for t in endo if t < half)), "endo_second_half": float(sum(1 for t in endo if t >= half)),
                     "recovery_lag": float(np.mean(lags)) if lags else float("nan")})
    f = ("reward", "n_changes", "endo_first_half", "endo_second_half", "recovery_lag")
    print("endo_second_half < endo_first_half means the agent stopped triggering its own changes")
    print_table(summarize_by(rows, ("agent", "regime"), f), ("agent", "regime"), f)


if __name__ == "__main__":
    run_experiment("exp15", __doc__, cells, scenario, REFERENCE_AGENTS, analyze, config={"T": T})
