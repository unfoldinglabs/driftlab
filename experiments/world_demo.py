"""Run any world with any agent: the smoke test and the template for new experiments.

    python -m experiments.world_demo --world inventory --mock --steps 20
    python -m experiments.world_demo --world form_filler --memory notes --feedback verbose --live
    python -m experiments.world_demo --world codebase --steps 8
    python -m experiments.world_demo --world inventory --baseline oracle       # non-LLM policy
    python -m experiments.world_demo --world rule_world --agent tabular
"""

import re

from experiments.common import NOTES, ref, run_experiment
from driftlab.core import Scenario
from driftlab.worlds.codebase import CodebaseWorld
from driftlab.worlds.formfiller import FormWorld
from driftlab.worlds.gridtext import RuleWorld
from driftlab.worlds.inventory import BaseStockOracle, InventoryWorld, MovingAverageAgent

WORLDS = {"rule_world": RuleWorld, "form_filler": FormWorld, "inventory": InventoryWorld, "codebase": CodebaseWorld}
OPTS: dict = {}


def cells(seeds):
    return [{"regime": OPTS["world"], "seed": s, "world_kwargs": OPTS["world_kwargs"]} for s in range(seeds)]


def scenario(cell, ctx):
    world = WORLDS[cell["regime"]](seed=cell["seed"], **cell.get("world_kwargs", {}))
    return Scenario(world, seed=cell["seed"])


class InventoryBaseline:
    """Adapter: a non-LLM inventory policy as an Agent. It reads the day from the prompt."""

    def __init__(self, spec, cell, scen, ctx):
        self.pol = BaseStockOracle(scen.world) if spec["baseline"] == "oracle" else MovingAverageAgent(scen.world)

    async def act(self, prompt):
        t = int(re.search(r"Day (\d+)", prompt).group(1)) - 1
        return f"ORDER: {self.pol.choose(t)}"

    async def observe(self, feedback, reward): ...


def analyze(run_dir):
    import numpy as np
    from driftlab.metrics import collect_steps, print_table, summarize_by
    rows = [{"agent": h["cell"]["agent"]["name"], "world": h["cell"]["regime"],
             "mean_reward": float(np.mean([s["reward"] for s in steps])), "parse_failures": h.get("parse_failures", 0)}
            for h, steps in collect_steps(run_dir)]
    print_table(summarize_by(rows, ("world", "agent"), ("mean_reward", "parse_failures")), ("world", "agent"), ("mean_reward", "parse_failures"))


def extra_args(ap):
    ap.add_argument("--world", required=True, choices=list(WORLDS))
    ap.add_argument("--memory", default="notes", choices=["none", "transcript", "notes", "skills"])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--feedback", default=None, help="world feedback richness level")
    ap.add_argument("--regime", default=None, help="inventory: seasonal | drifting | stationary | retail")
    ap.add_argument("--baseline", default=None, choices=["oracle", "moving_average"], help="inventory only")


if __name__ == "__main__":
    import sys
    argv = sys.argv[1:]
    # pre-parse world options so cells()/scenario() can see them (the shared CLI parses the rest)
    import argparse
    pre = argparse.ArgumentParser(add_help=False)
    extra_args(pre)
    known, _ = pre.parse_known_args(argv)
    kw = {k: v for k, v in (("T", known.steps), ("feedback", known.feedback)) if v is not None}
    if known.world == "inventory" and known.regime:
        kw["regime"] = known.regime
    OPTS.update(world=known.world, world_kwargs=kw)
    agents = [ref(f"llm_{known.memory}", {"kind": known.memory, **({"every": 10} if known.memory in ("notes", "skills") else {})})]
    if known.baseline:
        agents = [{"name": known.baseline, "type": "custom", "factory": "experiments.world_demo:InventoryBaseline",
                   "baseline": known.baseline}]
    sys.argv = [sys.argv[0]] + argv + ["--out", f"runs/demo_{known.world}"] if "--out" not in argv else sys.argv
    run_experiment("demo", __doc__, cells, scenario, agents, analyze, extra_args=extra_args)
