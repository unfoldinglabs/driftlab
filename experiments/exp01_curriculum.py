"""Experiment 1 — Curriculum targeting: which difficulty signal teaches fastest?

A programmatic teacher picks the depth (rule-nesting level) of each training
task. Policies: random, monotonic, frontier (target ~55% success),
failure_replay, frontier_novelty. Learning is scored on a FIXED held-out ladder
the teacher never sees, probed every EVAL_EVERY steps without feedback.

Hypothesis: frontier targeting acquires held-out skill fastest; pure frontier
targeting narrows the task distribution and the novelty term prevents it.

    python -m experiments.exp01_curriculum            # reference agent
    python -m experiments.exp01_curriculum --analyze
"""

from collections import Counter

import numpy as np

from experiments.common import world_for, NOTES, q, qlist, ref, run_experiment
from driftlab.agents.teacher import CurriculumTeacher
from driftlab.core import Scenario
from driftlab.live import LIVE
from driftlab.metrics import collect_steps, print_table, summarize_by
from driftlab.runner import env_cells
from driftlab.worlds.gridtext import RuleWorld, key_of, task_of

T, EVAL_EVERY, EVAL_PER_DEPTH = q(150), q(25), 4
DEPTHS = [1, 2, 3]
POLICIES = ["random", "monotonic", "frontier", "failure_replay", "frontier_novelty"]
REFERENCE_AGENTS = [ref("learner_notes", NOTES)]


def cells(seeds):
    return env_cells(POLICIES, seeds)


def eval_ladder(world, seed):
    rng = np.random.default_rng(seed + 777)
    ladder = []
    for d in DEPTHS:
        pool = [k for k in world.truth_table() if world.task_depth(task_of(k)) == d]
        ladder += [(task_of(pool[i]), d) for i in rng.permutation(len(pool))[:EVAL_PER_DEPTH]]
    return ladder


async def probe_ladder(world, ladder, agent):
    out = {d: [] for d in DEPTHS}
    for task, d in ladder:
        choice = world.parse(await agent.act(world.probe(task)))
        out[d].append(int(choice == world.correct(task)))
    return {str(d): float(np.mean(v)) for d, v in out.items()}


def scenario(cell, ctx):
    world = world_for(cell, T, needs=("sample_task", "probe"), depth=3, exception_rate=0.5)
    teacher = CurriculumTeacher(cell["regime"], DEPTHS, seed=cell["seed"])
    ladder = eval_ladder(world, cell["seed"])
    taught, keys = {}, Counter()

    def task_fn(t, w):
        task, d = teacher.next_task(w)
        taught[t] = d
        return task
    world.task_fn = task_fn

    async def after(t, rec, agent):
        k = key_of(rec["task"])
        teacher.record(taught[t], k, rec["reward"] > 0)
        keys[k] += 1
        extra = {"depth_taught": taught[t]}
        if (t + 1) % EVAL_EVERY == 0:
            extra["eval"] = await probe_ladder(world, ladder, agent)
            LIVE.metrics(eval_by_depth={k: round(v, 2) for k, v in extra["eval"].items()})
        return extra

    def summary():
        c = np.array(list(keys.values()), dtype=float)
        p = c / c.sum()
        return {**world.summary(), "curriculum_entropy": float(-(p * np.log(p)).sum()), "n_unique_keys": len(keys)}

    return Scenario(world, T=T, after_step=after, summary=summary, seed=cell["seed"])


def analyze(run_dir):
    rows = []
    for h, steps in collect_steps(run_dir):
        evals = [s["eval"] for s in steps if "eval" in s]
        if not evals:
            continue
        final = evals[-1]
        rows.append({"policy": h["cell"]["regime"], "agent": h["cell"]["agent"]["name"],
                     "final_d1": final["1"], "final_d2": final["2"], "final_d3": final["3"],
                     "eval_auc": float(np.mean([np.mean(list(e.values())) for e in evals])),
                     "entropy": h["curriculum_entropy"]})
    f = ("final_d1", "final_d2", "final_d3", "eval_auc", "entropy")
    print_table(summarize_by(rows, ("agent", "policy"), f), ("agent", "policy"), f)


if __name__ == "__main__":
    run_experiment("exp01", __doc__, cells, scenario, REFERENCE_AGENTS, analyze, config={"T": T, "policies": POLICIES})
