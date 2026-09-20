"""Experiment 10 — Teaching by contrast (environment engineering).

The teacher chooses PAIRS of requests that differ in exactly one attribute and
route to different desks, shown back to back, so the learner sees the rule
boundary directly. Compared with random order and the frontier policy on the
same fixed held-out ladder (probed without feedback every EVAL_EVERY steps).

Hypothesis: contrast pairs reach a given held-out accuracy in fewer steps,
with the largest gain on depth-2/3 exceptions.

    python -m experiments.exp10_teaching_by_contrast
    python -m experiments.exp10_teaching_by_contrast --analyze
"""

import numpy as np

from experiments.common import world_for, NOTES, q, qlist, ref, run_experiment
from experiments.exp01_curriculum import DEPTHS, eval_ladder, probe_ladder
from driftlab.agents.teacher import CurriculumTeacher
from driftlab.core import Scenario
from driftlab.live import LIVE
from driftlab.metrics import collect_steps, print_table, summarize_by
from driftlab.runner import env_cells
from driftlab.worlds.gridtext import RuleWorld, key_of

T, EVAL_EVERY = q(150), q(25)
POLICIES = ["random", "frontier", "contrast"]
REFERENCE_AGENTS = [ref("learner_notes", NOTES)]


def cells(seeds):
    return env_cells(POLICIES, seeds)


def scenario(cell, ctx):
    world = world_for(cell, T, needs=("sample_task", "sample_contrast_pair", "probe"), depth=3, exception_rate=0.5)
    policy = cell["regime"]
    teacher = CurriculumTeacher("frontier", DEPTHS, seed=cell["seed"]) if policy == "frontier" else None
    ladder = eval_ladder(world, cell["seed"])
    queue, taught = [], {}

    def task_fn(t, w):
        if policy == "contrast":
            if not queue:
                pair = w.sample_contrast_pair()
                queue.extend(pair if pair else [w.sample_task()])
            return queue.pop(0)
        if policy == "frontier":
            task, d = teacher.next_task(w)
            taught[t] = d
            return task
        return w.sample_task()
    world.task_fn = task_fn

    async def after(t, rec, agent):
        if teacher:
            teacher.record(taught[t], key_of(rec["task"]), rec["reward"] > 0)
        if (t + 1) % EVAL_EVERY == 0:
            ev = await probe_ladder(world, ladder, agent)
            LIVE.metrics(eval_by_depth={k: round(v, 2) for k, v in ev.items()})
            return {"eval": ev}
        return {}

    return Scenario(world, T=T, after_step=after, seed=cell["seed"])


def analyze(run_dir):
    rows = []
    for h, steps in collect_steps(run_dir):
        evals = [s["eval"] for s in steps if "eval" in s]
        if not evals:
            continue
        final = evals[-1]
        rows.append({"agent": h["cell"]["agent"]["name"], "policy": h["cell"]["regime"],
                     "final_d1": final["1"], "final_d2": final["2"], "final_d3": final["3"],
                     "eval_auc": float(np.mean([np.mean(list(e.values())) for e in evals])),
                     "eval_auc_exceptions": float(np.mean([(e["2"] + e["3"]) / 2 for e in evals]))})
    f = ("final_d1", "final_d2", "final_d3", "eval_auc", "eval_auc_exceptions")
    print_table(summarize_by(rows, ("agent", "policy"), f), ("agent", "policy"), f)


if __name__ == "__main__":
    run_experiment("exp10", __doc__, cells, scenario, REFERENCE_AGENTS, analyze, config={"T": T, "policies": POLICIES})
