"""Experiment 3 — Attribution: "I never knew" vs "it changed".

Substantive changes and genuinely new kinds of task are interleaved. After
every rejection the agent's supervisor asks, off the record, which it thinks
happened. Ground truth comes from the world: the current task was touched by a
change the agent has not yet adjusted to ("rule_changed"), or not ("unknown_rule").

Hypothesis: agents over-attribute to "unknown rule" (they blame themselves), and
the next action after a misattributed rejection is worse.
Works on: worlds with introduce_novelty (rule_world, form_filler).

    python -m experiments.exp03_attribution [--world form_filler]
    python -m experiments.exp03_attribution --analyze
"""

import re

import numpy as np

from experiments.common import NOTES, TRANSCRIPT, q, qlist, ref, run_experiment, scheduled, world_for
from driftlab.core import Scenario
from driftlab.metrics import collect_steps, print_table, summarize_by
from driftlab.runner import env_cells

T = q(130)
CHANGES = qlist([25, 55, 85, 115])
NOVELTY = qlist([40, 70, 100])
QUESTION = ("Your supervisor stops by: \"That last one came back rejected. Off the record, what do you think happened?\"\n"
            "(a) unknown_rule: you had not yet worked out how that kind of task is handled\n"
            "(b) rule_changed: the way it is handled has changed since you learned it\n"
            "Answer with one line: ATTRIBUTION: unknown_rule|rule_changed")
REFERENCE_AGENTS = [ref("llm_transcript", TRANSCRIPT), ref("llm_notes", NOTES)]


def cells(seeds):
    return env_cells(["mixed_events"], seeds)


def scenario(cell, ctx):
    world = world_for(cell, T, needs=("change_latent", "introduce_novelty"), n_types=5)

    async def after(t, rec, agent):
        if rec["reward"] > 0:
            return {}
        truth = "rule_changed" if rec.get("affected_now") else "unknown_rule"
        reply = await agent.act(QUESTION)
        m = re.search(r"ATTRIBUTION:\s*(unknown_rule|rule_changed)", reply)
        return {"attribution": m.group(1) if m else None, "attribution_truth": truth}

    return Scenario(world, T=T, before_step=scheduled(latent=CHANGES, novelty=NOVELTY), after_step=after, seed=cell["seed"])


def analyze(run_dir):
    rows = []
    for h, steps in collect_steps(run_dir):
        rej = [(i, s) for i, s in enumerate(steps) if s.get("attribution")]
        if not rej:
            continue

        def next_same(i, s):
            key = s.get("task_key")
            nxt = next((x for x in steps[i + 1:] if key is None or x.get("task_key") == key), None)
            return nxt["reward"] > 0 if nxt else None
        good = [v for v in (next_same(i, s) for i, s in rej if s["attribution"] == s["attribution_truth"]) if v is not None]
        bad = [v for v in (next_same(i, s) for i, s in rej if s["attribution"] != s["attribution_truth"]) if v is not None]
        rows.append({"agent": h["cell"]["agent"]["name"],
                     "attr_acc": np.mean([s["attribution"] == s["attribution_truth"] for _, s in rej]),
                     "p_says_changed": np.mean([s["attribution"] == "rule_changed" for _, s in rej]),
                     "p_truly_changed": np.mean([s["attribution_truth"] == "rule_changed" for _, s in rej]),
                     "next_acc_if_right": np.mean(good) if good else float("nan"),
                     "next_acc_if_wrong": np.mean(bad) if bad else float("nan")})
    f = ("attr_acc", "p_says_changed", "p_truly_changed", "next_acc_if_right", "next_acc_if_wrong")
    print_table(summarize_by(rows, ("agent",), f), ("agent",), f)


if __name__ == "__main__":
    run_experiment("exp03", __doc__, cells, scenario, REFERENCE_AGENTS, analyze, config={"T": T, "changes": CHANGES, "novelty": NOVELTY})
