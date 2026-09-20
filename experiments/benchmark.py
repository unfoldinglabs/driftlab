"""The driftlab reference suite: the same five canonical experiments for every
new agent, each on its canonical world, producing one comparable profile matrix.

    E02 detect + adapt      · intake desk       (discrete rules, nested exceptions)
    E04 surface vs latent   · order forms       (structural cosmetic change vs schema drift)
    E06 calibration         · intake desk       (confidence as a change signal)
    E15 endogenous drift    · inventory         (continuous costs, self-caused strain)
    E18 adversarial chase   · claims desk       (drift aimed by an opponent)

Suite v2 (2026-09-09): one world certified only discrete routing, and the
flagship phenomena (adversarial drift, a continuous world) were missing. The
suite now spans four worlds. exp05 left the suite because its arms ARE the
experiment (a memory ablation collapses when one benchmarked agent replaces
them) — the same reason exp17 is not in it. Change the suite rarely: every
past matrix becomes incomparable with the next.

The suite is the stable yardstick; the research experiments grow independently.
Runs land in the normal per-experiment directories (runs/expNN/<world>/), so a
benchmarked agent is directly comparable with every other agent that ever
played those experiments — including external harnesses via the MCP server,
whose logs the matrix picks up with --analyze. Seeds default to 5, the floor
below which no verdict is conclusive.

    python -m experiments.benchmark --mock --smoke                 # pipeline check
    python -m experiments.benchmark --agent tabular                # baseline (rule_world only)
    python -m experiments.benchmark --agent custom:mypkg.agents:build
    python -m experiments.benchmark --agent-spec agents.json       # your agent profile(s)
    python -m experiments.benchmark --analyze                      # matrix from existing logs only
    python -m experiments.benchmark --world rule_world             # force one world (old behavior)

Output: the agent x experiment matrix of driftlab scores, mean dimensions per
agent, and runs/benchmark/suite.json (or <world>.json when --world forces one)
with the full profile per cell.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.common import NOTES, SMOKE, VERDICT_SEEDS, ref, resolve_agents  # noqa: E402
from experiments.registry import REGISTRY  # noqa: E402

SUITE = (("exp02_detect_adapt_lag", "rule_world"),
         ("exp04_surface_vs_latent", "form_filler"),
         ("exp06_calibration", "rule_world"),
         ("exp15_endogenous_drift", "inventory"),
         ("exp18_adversarial_chase", "claims_desk"))
DEFAULT_AGENTS = [ref("llm_notes", NOTES)]


def _cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", default=None,
                    choices=["rule_world", "form_filler", "inventory", "codebase", "claims_desk", "campaign_desk"],
                    help="force every suite experiment onto one world (default: each on its canonical world)")
    ap.add_argument("--analyze", action="store_true", help="build the matrix from existing logs, run nothing")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, default=None,
                    help=f"default {VERDICT_SEEDS} (the conclusive-verdict floor); 1 in smoke mode")
    ap.add_argument("--model", default=None)
    ap.add_argument("--effort", default=None)
    ap.add_argument("--agent", default=None, help="tabular | custom:pkg.module:factory")
    ap.add_argument("--agent-spec", default=None, help="JSON file with a list of agent specs")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--budget-usd", type=float, default=None)
    ap.add_argument("--resume", action="store_true", help="skip episodes whose run log is already complete")
    args = ap.parse_args()
    if args.budget_usd is not None:
        from driftlab.agents.brain import LEDGER
        LEDGER.budget_usd = args.budget_usd
    return args


def run_suite(args):
    import importlib

    from driftlab.agents.reference import make_agent
    from driftlab.runner import expand, run_cells
    specs = resolve_agents(args, DEFAULT_AGENTS)
    factory = lambda cell, scen, ctx: make_agent(cell["agent"], cell, scen, ctx)  # noqa: E731
    for exp, suite_world in SUITE:
        name = exp.split("_")[0]
        m = importlib.import_module(f"experiments.{exp}")
        env = [{"world": args.world or suite_world, **c} for c in m.cells(args.seeds or (1 if SMOKE else VERDICT_SEEDS))]
        world = env[0]["world"]  # a cell that pins its own world (exp18) wins over --world
        out = ROOT / "runs" / name / world
        print(f"\n== {name} ({REGISTRY[name]['title']}) on {world} -> {os.path.relpath(out)}")
        asyncio.run(run_cells(expand(env, specs), m.scenario, factory, str(out), args.concurrency,
                              config={"suite": True, "quick": SMOKE, "world": world, "registry": REGISTRY[name]},
                              resume=args.resume))


def matrix(world_override: str | None = None) -> dict:
    """Aggregate the suite's profiles into one agent x experiment view. Each experiment
    is read from its canonical world (or from world_override when one was forced)."""
    from driftlab.profile import DIMENSIONS, _fmt, _nanmean, profile_dir
    cells: dict = {}
    worlds: dict = {}
    for exp, suite_world in SUITE:
        name = exp.split("_")[0]
        world = world_override or suite_world
        d = ROOT / "runs" / name / world
        if not list(d.glob("*.jsonl")) and world_override:
            d = ROOT / "runs" / name / suite_world  # a pinned-world experiment ignored the override
            world = suite_world
        worlds[name] = world
        if not list(d.glob("*.jsonl")):
            continue
        prof = profile_dir(str(d))
        (d / "profile.json").write_text(json.dumps(prof, indent=2))
        for agent, p in prof["agents"].items():
            cells.setdefault(agent, {})[name] = p
    exps = [e.split("_")[0] for e, _ in SUITE]
    agents = {a: {"experiments": by,
                  "mean_score": _nanmean([p["driftlab_score"] for p in by.values()]),
                  "mean_dimensions": {d: _nanmean([p["dimensions"][d] for p in by.values()]) for d in DIMENSIONS}}
              for a, by in sorted(cells.items())}

    head = f"{'agent':<20}" + "".join(f"{e:>9}" for e in exps) + f"{'mean':>9}"
    print("\nDRIFTLAB REFERENCE SUITE — driftlab score per agent x experiment")
    print("worlds: " + ", ".join(f"{e}·{worlds[e]}" for e in exps))
    print(head + "\n" + "-" * len(head))
    for a, row in agents.items():
        line = f"{a[:19]:<20}" + "".join(_fmt(row["experiments"].get(e, {}).get("driftlab_score"), "{:.2f}", 9) for e in exps)
        print(line + _fmt(row["mean_score"], "{:.2f}", 9))
    print(f"\n{'mean dimensions':<20}" + "".join(f"{d:>12}" for d in DIMENSIONS))
    for a, row in agents.items():
        print(f"{a[:19]:<20}" + "".join(_fmt(row["mean_dimensions"][d], "{:.2f}", 12) for d in DIMENSIONS))
    print("\nThe score averages the capability dimensions (adaptation, knowledge, epistemics);")
    print("efficiency is reported beside it, never blended in. The matrix is the summary, not the")
    print("result: every cell expands into its profile in runs/<exp>/<world>/profile.json.")

    label = world_override or "suite"
    result = {"schema": 2, "suite": exps, "worlds": worlds, "world": world_override or "mixed",
              "generated": time.time(),
              "versions": {e.split("_")[0]: REGISTRY[e.split("_")[0]]["version"] for e, _ in SUITE}, "agents": agents}
    out = ROOT / "runs" / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{label}.json").write_text(json.dumps(result, indent=2))
    print(f"matrix written to {os.path.relpath(out / f'{label}.json')}")
    return result


if __name__ == "__main__":
    args = _cli()
    if not args.analyze:
        run_suite(args)
    matrix(args.world)
