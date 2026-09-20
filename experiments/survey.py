"""Survey: one agent across every experiment — small, stratified, fast.

    python -m experiments.survey --plan                              # the bill, nothing runs
    python -m experiments.survey --model openai/gpt-5.6-luna --quick --resume
    python -m experiments.survey --agent-spec agents.json --quick --seeds 2 --budget-usd 30
    python -m experiments.survey --only exp04,exp05,exp16 --quick    # just the re-versioned ones

A survey plays every registered experiment's full regime grid with ONE agent,
each on its canonical world (the benchmark suite's pairing where one exists,
the intake desk otherwise; experiments that pin their own worlds keep them).
With --quick and the default 3 seeds it fills the whole benchmark matrix and
every experiment table with indicative values in one pass:

    stratified   every experiment, every regime — no condition is skipped
    small        3 seeds, half-length episodes (--quick), compact task spaces
    honest       quick runs count (they stay above the validity floor), and 3
                 seeds sit right at the verdict floor: bootstrap intervals are
                 wide at n=3, so only large effects turn conclusive — a rough
                 first read on every hypothesis, confirmed later at 5+ seeds

Logs land in the normal runs/<exp>/<world> directories and are resume-
compatible with hypothesis runs and full grids: a later run at more seeds
fills in on top of the survey instead of repeating it. Validation refreshes
at the end. To turn survey coverage into verdicts, extend the interesting
hypotheses with `run_hypothesis HNNN --seeds 5 --resume`.

One honest caveat: experiments whose reference arms ARE the contrast (exp05's
windows, exp12's note schedules, exp17's wipes) reduce to plain coverage when
one surveyed agent replaces the arms — the agent still plays their changing
worlds, but the ablation itself needs the reference arms.
"""

import argparse
import asyncio
import contextlib
import importlib
import io
import pkgutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.common import (MODE, NOTES, VERDICT_SEEDS, out_dir, ref,  # noqa: E402
                                refresh_validation, resolve_agents)
from experiments.registry import REGISTRY  # noqa: E402

SURVEY_SEEDS = 3       # the verdict floor: wide intervals, but every hypothesis gets a first read
DEFAULT_AGENTS = [ref("llm_notes", NOTES)]


def _modules() -> dict:
    import experiments
    return {m.name.split("_")[0]: f"experiments.{m.name}"
            for m in pkgutil.iter_modules(experiments.__path__) if m.name.startswith("exp")}


def _suite_worlds() -> dict:
    from experiments.benchmark import SUITE
    return {e.split("_")[0]: w for e, w in SUITE}


def plan(seeds: int, only: set | None) -> list[dict]:
    mods, suite = _modules(), _suite_worlds()
    steps = []
    for eid in REGISTRY:
        if eid not in mods or (only and eid not in only):
            continue
        mod = importlib.import_module(mods[eid])
        world = suite.get(eid, "rule_world")
        cells = [{"world": world, **c} for c in mod.cells(seeds)]
        steps.append({"eid": eid, "mod": mod, "cells": cells, "world": cells[0].get("world", world),
                      "regimes": sorted({str(c.get("regime")) for c in cells}), "T": getattr(mod, "T", None)})
    return steps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="print what would run, run nothing")
    ap.add_argument("--seeds", type=int, default=SURVEY_SEEDS)
    ap.add_argument("--only", default=None, help="comma-separated expNN filter (default: every experiment)")
    ap.add_argument("--model", default=None, help="survey this model as a notes agent")
    ap.add_argument("--effort", default=None)
    ap.add_argument("--agent", default=None, help="tabular | cli:<harness>[:model] | custom:pkg.mod:factory")
    ap.add_argument("--agent-spec", default=None, help="JSON list of agent specs (surveys each)")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="plumbing check (a tenth of everything; never counts)")
    ap.add_argument("--quick", action="store_true", help="half-length but valid episodes (recommended)")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--budget-usd", type=float, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if not (args.plan or args.model or args.agent or args.agent_spec or args.mock):
        raise SystemExit("a survey is one agent everywhere: give --model, --agent or --agent-spec "
                         "(or --plan to see the bill first)")
    only = {e.strip() for e in args.only.split(",")} if args.only else None

    steps = plan(args.seeds, only)
    n_specs = len(resolve_agents(args, DEFAULT_AGENTS)) if not args.plan else None
    print(f"survey: every experiment, full regime grids, {args.seeds} seed(s) ({MODE} mode)")
    total_eps, total_steps = 0, 0
    for s in steps:
        eps = len(s["cells"]) * (n_specs or 1)
        total_eps += eps
        total_steps += eps * (s["T"] or 0)
        print(f"  {s['eid']}  {REGISTRY[s['eid']]['title']:<46} {s['world']:<14}"
              f" {len(s['regimes'])} regime(s) x {args.seeds} seeds = {eps} episodes of {s['T'] or '?'} steps")
    print(f"  total: {total_eps} episodes, ~{total_steps} steps"
          + ("" if args.plan else f" (first-read verdicts at {args.seeds} seeds; confirm at {VERDICT_SEEDS}+ with run_hypothesis --resume)"))
    if args.plan:
        return

    if args.budget_usd is not None:
        from driftlab.agents.brain import LEDGER
        LEDGER.budget_usd = args.budget_usd
    from driftlab.agents.reference import make_agent
    from driftlab.live import LIVE
    from driftlab.profile import report
    from driftlab.runner import expand, run_cells

    factory = lambda cell, scen, ctx: make_agent(cell["agent"], cell, scen, ctx)  # noqa: E731
    done, incomplete = 0, []
    for s in steps:
        out = out_dir(s["eid"], s["world"], lambda n: s["cells"][:1] or [{}])
        planned = len(s["cells"]) * (n_specs or 1)
        print(f"\n=== {s['eid']} -> {out}")
        try:
            grid = expand(s["cells"], resolve_agents(args, DEFAULT_AGENTS))
            results = asyncio.run(run_cells(grid, s["mod"].scenario, factory, out, args.concurrency,
                                            config={"mode": MODE, "quick": MODE == "smoke", "world": s["world"],
                                                    "survey": True, "registry": REGISTRY[s["eid"]]},
                                            resume=args.resume))
        except SystemExit as e:  # a capability gate, or every episode failing the same way — skip, don't die
            print(f"  {s['eid']} skipped: {e}")
            incomplete.append(f"{s['eid']} (0/{planned})")
            continue
        ok = sum(1 for r in results if isinstance(r, dict) and not r.get("error"))
        done += ok
        if ok < planned:
            incomplete.append(f"{s['eid']} ({ok}/{planned})")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            s["mod"].analyze(out)
            print()
            report(out)
        LIVE.analysis(s["eid"], buf.getvalue())
        print(buf.getvalue())

    refresh_validation()
    print(f"\nsurvey finished: {done} of {total_eps} planned episodes are on disk.")
    if incomplete:
        print(f"INCOMPLETE — re-run the same command with --resume to fill in: {', '.join(incomplete)}")
    else:
        print("Full coverage: the matrix, experiment tables and first-read verdicts now cover this agent "
              f"everywhere; confirm anything interesting at {VERDICT_SEEDS}+ seeds "
              "(run_hypothesis HNNN --resume runs only the extension).")


if __name__ == "__main__":
    main()
