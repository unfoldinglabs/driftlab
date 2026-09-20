"""Experiment 14 — What did the learner actually acquire? Shift the evaluation.

One scenario, four phases on one agent:
    train        100 steps on a depth-3 RuleWorld
    in_dist      day-one probe of every key (no feedback), then 20 steps with feedback
    new_rules    same request types, freshly drawn rules: probe, then 20 steps
    new_types    three never-seen request types: probe, then 20 steps

Hypothesis: day-one accuracy collapses on new_rules and new_types, but the
20-step re-adaptation is faster than a fresh agent's. The fresh-agent control
is run in-process when an agent factory is available (reference agents); for
an external harness the control is skipped and only the trained agent's
re-adaptation is logged.

    python -m experiments.exp14_eval_shift
    python -m experiments.exp14_eval_shift --analyze
"""

import numpy as np

from experiments.common import world_for, NOTES, q, qlist, ref, run_experiment
from driftlab.core import Scenario, run
from driftlab.live import LIVE
from driftlab.metrics import collect_steps, print_table, summarize_by
from driftlab.runner import env_cells
from driftlab.worlds.gridtext import TYPES, RuleWorld, task_of

T_TRAIN, T_ADAPT = q(100), q(20)
SHIFTS = ["in_dist", "new_rules", "new_types"]
T = T_TRAIN + len(SHIFTS) * T_ADAPT
REFERENCE_AGENTS = [ref("learner_notes", NOTES)]


def cells(seeds):
    return env_cells(["shift_suite"], seeds)


def shifted(kind, seed):
    if kind == "in_dist":
        return RuleWorld(seed=seed, T=T, depth=3, n_mutations=0, exception_rate=0.5), None
    if kind == "new_rules":
        return RuleWorld(seed=seed + 1000, T=T, depth=3, n_mutations=0, exception_rate=0.5), None
    return RuleWorld(seed=seed, T=T, depth=3, n_mutations=0, exception_rate=0.5, n_types=9), TYPES[6:9]


def scenario(cell, ctx):
    seed = cell["seed"]
    world = world_for(cell, T, needs=("sample_task", "probe", "task_key"), depth=3, exception_rate=0.5)
    phase = {"name": "train", "types": None}
    boundaries = {T_TRAIN + i * T_ADAPT: s for i, s in enumerate(SHIFTS)}

    def task_fn(t, w):
        return w.sample_task(types=phase["types"])
    world.task_fn = task_fn

    def before(t, w):
        if t in boundaries:
            src, types = shifted(boundaries[t], seed)
            w.rules, w.types = dict(src.rules), list(src.types)   # swap the hidden rule set in place
            phase.update(name=boundaries[t], types=types, probe_due=True)

    async def after(t, rec, agent):
        extra = {"phase": phase["name"]}
        if phase.get("probe_due"):  # day-one probe happens after the first step of the phase
            phase["probe_due"] = False
            keys = [k for k in world.truth_table() if phase["types"] is None or k[0] in phase["types"]]
            hits = [int(world.parse(await agent.act(world.probe(task_of(k)))) == world.correct(task_of(k))) for k in keys]
            extra["day1_acc"] = float(np.mean(hits))
            LIVE.metrics(shift=phase["name"], day1_acc=round(extra["day1_acc"], 2))
            if ctx.get("agent_factory"):  # fresh-agent control, in-process only
                fresh_world, types = shifted(phase["name"], seed)
                fresh_world.task_fn = lambda tt, w: w.sample_task(types=types)
                fresh = ctx["agent_factory"]({**cell, "agent": {**cell["agent"], "name": "fresh"}},
                                             Scenario(fresh_world, T=T_ADAPT), ctx)
                _, recs = await run(Scenario(fresh_world, T=T_ADAPT, seed=seed), fresh)
                extra["fresh_adapt_last10"] = float(np.mean([r["reward"] for r in recs[-10:]]))
        return extra

    return Scenario(world, T=T, before_step=before, after_step=after, seed=seed)


def analyze(run_dir):
    rows = []
    for h, steps in collect_steps(run_dir):
        for shift in SHIFTS:
            ph = [s for s in steps if s.get("phase") == shift]
            if not ph:
                continue
            first = ph[0]
            rows.append({"agent": h["cell"]["agent"]["name"], "shift": shift,
                         "day1_acc": first.get("day1_acc", float("nan")),
                         "adapt_trained": float(np.mean([s["reward"] for s in ph[-10:]])),
                         "adapt_fresh": first.get("fresh_adapt_last10", float("nan"))})
    for r in rows:
        r["transfer_gain"] = r["adapt_trained"] - r["adapt_fresh"]
    f = ("day1_acc", "adapt_trained", "adapt_fresh", "transfer_gain")
    print("adapt_* = accuracy over the last 10 of 20 adaptation steps; transfer_gain = trained - fresh")
    print_table(summarize_by(rows, ("agent", "shift"), f), ("agent", "shift"), f)


if __name__ == "__main__":
    run_experiment("exp14", __doc__, cells, scenario, REFERENCE_AGENTS, analyze, config={"T_train": T_TRAIN, "T_adapt": T_ADAPT})
