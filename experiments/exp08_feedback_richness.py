"""Experiment 8 — Feedback richness (environment engineering).

This experiment spans its own set of worlds and ignores --world.

The same agent in three worlds, each at its three feedback levels:
    form_filler   terse / fields / verbose
    inventory     terse / standard / verbose
    codebase      tests_only / names / verbose

Hypothesis: hints about WHY beat hints about WHAT, most of all right after the
world changes (new form version, new season, new session).

    python -m experiments.exp08_feedback_richness
    python -m experiments.exp08_feedback_richness --analyze
"""

import numpy as np

from experiments.common import q, qlist, ref, run_experiment
from driftlab.core import Scenario
from driftlab.metrics import collect_steps, print_table, summarize_by
from driftlab.worlds.codebase import CodebaseWorld
from driftlab.worlds.formfiller import FormWorld
from driftlab.worlds.inventory import InventoryWorld

WORLDS = {
    "form_filler": (FormWorld, ["terse", "fields", "verbose"], "version", {"T": q(90), "version_every": q(15)}),
    "inventory": (InventoryWorld, ["terse", "standard", "verbose"], "season", {"T": q(90), "regime": "seasonal", "season_length": q(30)}),
    "codebase": (CodebaseWorld, ["tests_only", "names", "verbose"], "session", {"T": q(24), "session_length": q(8)}),
}
POST_K = q(8)
REFERENCE_AGENTS = [ref("llm_notes", {"kind": "notes", "every": 8})]


def cells(seeds):
    return [{"regime": f"{w}:{fb}", "world": w, "feedback": fb, "seed": s}
            for w, (_, levels, _, _) in WORLDS.items() for fb in levels for s in range(seeds)]


def scenario(cell, ctx):
    cls, _, _, kw = WORLDS[cell["world"]]
    return Scenario(cls(seed=cell["seed"], feedback=cell["feedback"], **kw), seed=cell["seed"])


def analyze(run_dir):
    rows = []
    for h, steps in collect_steps(run_dir):
        c = h["cell"]
        marker = WORLDS[c["world"]][2]
        r = np.array([s["reward"] for s in steps], dtype=float)
        marks = [s.get(marker) for s in steps]
        changes = [i for i in range(1, len(marks)) if marks[i] is not None and marks[i] != marks[i - 1]]
        post = [r[i:i + POST_K].mean() for i in changes] or [float("nan")]
        rows.append({"agent": c["agent"]["name"], "world": c["world"], "feedback": c["feedback"],
                     "mean_reward": float(r.mean()), "post_change_reward": float(np.nanmean(post)), "n_changes": len(changes)})
    f = ("mean_reward", "post_change_reward", "n_changes")
    print_table(summarize_by(rows, ("agent", "world", "feedback"), f), ("agent", "world", "feedback"), f, width=13)


if __name__ == "__main__":
    run_experiment("exp08", __doc__, cells, scenario, REFERENCE_AGENTS, analyze, config={"worlds": list(WORLDS), "post_k": POST_K})
