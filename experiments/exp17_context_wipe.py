"""Experiment 17 — What does consolidation preserve across a context wipe?

Halfway through the episode, some agents lose their raw history: a transcript
agent loses everything it was carrying, a notes agent loses only whatever it
had not yet consolidated (the written notes survive — that is the point of
writing them). The wiped and unwiped versions of each memory play the same
seeded worlds, with six real rule changes on the usual schedule.

Hypothesis: raw transcripts and consolidated notes tie while the context
persists, and come apart the moment it is severed. If true, consolidation's
value is not day-to-day performance but survival across context boundaries.
Works on: any world with real changes.

    python -m experiments.exp17_context_wipe [--world form_filler]
    python -m experiments.exp17_context_wipe --analyze
"""

from collections import defaultdict

import numpy as np

from experiments.common import NOTES, TRANSCRIPT, q, ref, run_experiment, scheduled, spaced, world_for
from driftlab.core import Scenario
from driftlab.metrics import collect_steps
from driftlab.runner import env_cells

T, N_CHANGES = q(120), 6
WIPE_AT = q(60)
REFERENCE_AGENTS = [ref("llm_transcript", TRANSCRIPT),
                    ref("llm_transcript_wiped", {**TRANSCRIPT, "wipe_at": WIPE_AT}),
                    ref("llm_notes", NOTES),
                    ref("llm_notes_wiped", {**NOTES, "wipe_at": WIPE_AT})]


def cells(seeds):
    return env_cells(["changing"], seeds)


def scenario(cell, ctx):
    world = world_for(cell, T, needs=("change_latent",))
    return Scenario(world, T=T, before_step=scheduled(latent=spaced(N_CHANGES, T, q(20))), seed=cell["seed"])


def analyze(run_dir):
    groups = defaultdict(list)
    for h, steps in collect_steps(run_dir):
        if not h or not steps or "reward" not in steps[0]:
            continue
        r = [s["reward"] for s in steps]
        groups[h["cell"]["agent"]["name"]].append({
            "pre": float(np.mean(r[:WIPE_AT])), "post": float(np.mean(r[WIPE_AT:])),
            "dip10": float(np.mean(r[WIPE_AT:WIPE_AT + 10])) if len(r) > WIPE_AT else float("nan")})
    print(f"{'agent':<24}{'acc before wipe':>16}{'acc after':>12}{'10 steps after':>16}")
    print("-" * 68)
    for a, rs in sorted(groups.items()):
        m = lambda k: float(np.nanmean([x[k] for x in rs]))  # noqa: E731
        print(f"{a:<24}{m('pre'):>16.3f}{m('post'):>12.3f}{m('dip10'):>16.3f}")
    print(f"\n(the wipe lands at step {WIPE_AT}; unwiped arms are the controls on the same seeds)")


if __name__ == "__main__":
    run_experiment("exp17", __doc__, cells, scenario, REFERENCE_AGENTS, analyze,
                   config={"T": T, "n_changes": N_CHANGES, "wipe_at": WIPE_AT})
