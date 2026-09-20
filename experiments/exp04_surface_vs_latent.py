"""Experiment 4 — Surface change vs latent change.

At three points something changes: only the appearance, only the substance (a
rule flips, the schema mutates, demand jumps, the style guide changes), both,
or nothing.

v2: cosmetic change is structural, not just re-worded. Frontier models shrug
off synonym swaps (the v1 result), so on worlds that support it the surface
change now re-renders the task in a different format and field order — the
intake request becomes a fielded block with inverted attribute order, the
source record switches between JSON, key-value lines and a TSV table. The
correct behavior still never changes; only the parse does.

Hypothesis: agents over-react to appearance and under-react to substance.
Over-reaction index = dip(surface) / dip(latent).
Works on: any world.

    python -m experiments.exp04_surface_vs_latent [--world codebase]
    python -m experiments.exp04_surface_vs_latent --analyze
"""

import numpy as np

from experiments.common import NOTES, TRANSCRIPT, q, qlist, ref, run_experiment, scheduled, world_for
from driftlab.core import Scenario
from driftlab.metrics import collect_steps, dip_after, print_table, steps_to_streak, summarize_by
from driftlab.runner import env_cells

T = q(120)
EVENTS = qlist([30, 60, 90])
REGIMES = {"none": (False, False), "surface": (True, False), "latent": (False, True), "both": (True, True)}
REFERENCE_AGENTS = [ref("llm_transcript", TRANSCRIPT), ref("llm_notes", NOTES)]


def cells(seeds):
    return env_cells(list(REGIMES), seeds)


def scenario(cell, ctx):
    surface, latent = REGIMES[cell["regime"]]
    world = world_for(cell, T, needs=("change_latent", "change_surface"))
    if hasattr(world, "surface_strength"):  # v2: structural cosmetic change where the world supports it
        world.surface_strength = "structural"
    return Scenario(world, T=T, before_step=scheduled(latent=EVENTS if latent else (), surface=EVENTS if surface else ()), seed=cell["seed"])


def analyze(run_dir):
    rows = []
    for h, steps in collect_steps(run_dir):
        r = [s["reward"] for s in steps]
        rows.append({"agent": h["cell"]["agent"]["name"], "regime": h["cell"]["regime"],
                     "acc": float(np.mean(r[q(20):])), "dip": float(np.mean([dip_after(r, e) for e in EVENTS])),
                     "steps_to_competence": steps_to_streak(r), "acc_first20": float(np.mean(r[:q(20)]))})
    f = ("acc", "dip", "steps_to_competence", "acc_first20")
    summ = summarize_by(rows, ("agent", "regime"), f)
    print("acc_first20 near 1.0 in an episode other than the first means knowledge carried over from a previous episode")
    print_table(summ, ("agent", "regime"), f)
    print("\nOver-reaction index = dip(surface) / dip(latent)  (>1: reacts more to appearance than to substance)")
    for a in sorted({a for a, _ in summ}):
        s, l = summ.get((a, "surface"), {}).get("dip", np.nan), summ.get((a, "latent"), {}).get("dip", np.nan)
        print(f"  {a:<16} {s / l if l else float('nan'):+.2f}   (surface dip {s:+.3f}, latent dip {l:+.3f})")


if __name__ == "__main__":
    run_experiment("exp04", __doc__, cells, scenario, REFERENCE_AGENTS, analyze, config={"T": T, "events": EVENTS})
