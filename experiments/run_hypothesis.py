"""Run exactly what a hypothesis needs — the preferred way to test one.

    python -m experiments.run_hypothesis H008 --plan            # show what would run
    python -m experiments.run_hypothesis H008                   # run it (5 seeds)
    python -m experiments.run_hypothesis H008 --model openai/gpt-5.6-luna --resume
    python -m experiments.run_hypothesis H001 --seeds 10 --quick

The plan comes from the registry: every experiment that bears on the hypothesis
and registers a prediction, trimmed to the regimes those predictions actually
reference (exp18's stable control, for example, is not needed for a verdict and
is skipped). Seeds default to 5, the confirmatory standard (verdicts are
attempted from 3 pairs, on wide small-n intervals, so only large effects turn
conclusive early); agents default to each experiment's own arms, with --model pinning
one model across them. Logs land in the same runs/<exp>/<world> directories as
full grids and are resume-compatible with them — a later benchmark or full run
just fills in the regimes a hypothesis run skipped. Validation refreshes
automatically at the end and prints this hypothesis's verdict.
"""

import argparse
import asyncio
import contextlib
import importlib
import io
import json
import pkgutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.common import (MIN_PAIRS, MODE, VERDICT_SEEDS, WORLD, out_dir,  # noqa: E402
                                refresh_validation, resolve_agents)
from experiments.registry import REGISTRY, regimes_needed  # noqa: E402


def _modules() -> dict:
    import experiments
    mods = {}
    for m in pkgutil.iter_modules(experiments.__path__):
        if m.name.startswith("exp"):
            mods[m.name.split("_")[0]] = f"experiments.{m.name}"
    return mods


def plan(hid: str, seeds: int, world: str) -> list[dict]:
    mods = _modules()
    steps = []
    for eid, entry in REGISTRY.items():
        if hid not in entry["hypotheses"] or not entry.get("predictions"):
            continue
        mod = importlib.import_module(mods[eid])
        needed = regimes_needed(entry)
        cells_all = [{"world": world, **c} for c in mod.cells(seeds)]
        cells = [c for c in cells_all if needed is None or str(c.get("regime")) in needed]
        dropped = sorted({str(c.get("regime")) for c in cells_all} - {str(c.get("regime")) for c in cells})
        steps.append({"eid": eid, "entry": entry, "mod": mod, "cells": cells, "dropped": dropped,
                      "arms": len(mod.REFERENCE_AGENTS), "T": getattr(mod, "T", None)})
    return steps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hypothesis", help="HNNN, e.g. H008")
    ap.add_argument("--plan", action="store_true", help="print what would run, run nothing")
    ap.add_argument("--seeds", type=int, default=VERDICT_SEEDS)
    ap.add_argument("--world", default=WORLD,
                    choices=["rule_world", "form_filler", "inventory", "codebase", "claims_desk", "campaign_desk"])
    ap.add_argument("--model", default=None, help="pin one model across every experiment's arms")
    ap.add_argument("--effort", default=None)
    ap.add_argument("--agent", default=None, help="tabular | cli:<harness>[:model] | custom:pkg.mod:factory")
    ap.add_argument("--agent-spec", default=None, help="JSON list of agent specs (replaces the arms — beware for ablations)")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="plumbing check (a tenth of everything; never counts)")
    ap.add_argument("--quick", action="store_true", help="half-length but valid episodes")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--budget-usd", type=float, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    hid = args.hypothesis.upper()

    steps = plan(hid, args.seeds, args.world)
    if not steps:
        raise SystemExit(f"{hid}: no experiment in the registry bears on it with a registered prediction")
    if args.seeds < MIN_PAIRS and not (args.plan or args.smoke):
        print(f"note: {args.seeds} seeds cannot produce a verdict (the floor is {MIN_PAIRS}; "
              f"{VERDICT_SEEDS}+ is the confirmatory standard)")

    print(f"{hid} needs, per the registry ({MODE} mode):")
    total = 0
    for s in steps:
        kept = sorted({str(c.get("regime")) for c in s["cells"]})
        episodes = len(s["cells"]) * s["arms"]
        total += episodes
        drop = f" (skipping {', '.join(s['dropped'])}: not referenced by any prediction)" if s["dropped"] else ""
        print(f"  {s['eid']}  {s['entry']['title']}\n"
              f"        regimes {', '.join(kept)}{drop} · {s['arms']} arm(s) x {args.seeds} seeds"
              f" = {episodes} episodes of {s['T'] or '?'} steps")
    print(f"  total: {total} episodes")
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
    for s in steps:
        out = out_dir(s["eid"], args.world, lambda n: s["cells"][:1] or [{}])
        print(f"\n=== {s['eid']} -> {out}")
        try:
            grid = expand(s["cells"], resolve_agents(args, s["mod"].REFERENCE_AGENTS))
            asyncio.run(run_cells(grid, s["mod"].scenario, factory, out, args.concurrency,
                                  config={"mode": MODE, "quick": MODE == "smoke", "world": args.world,
                                          "hypothesis_run": hid, "registry": s["entry"]}, resume=args.resume))
        except SystemExit as e:
            # a capability gate (the world can't host this experiment) is skippable;
            # anything else that killed every episode (bad credentials, a code error)
            # must abort the whole run rather than masquerade as completion
            if "lacks capabilities" in str(e):
                print(f"  {s['eid']} skipped on this world: {e}")
                continue
            raise
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            s["mod"].analyze(out)
            print()
            report(out)
        LIVE.analysis(s["eid"], buf.getvalue())
        print(buf.getvalue())

    refresh_validation()
    val = json.loads((ROOT / "runs" / "validation.json").read_text())
    h = val.get("hypotheses", {}).get(hid)
    if h:
        print(f"\n{hid}: proposed {h['proposed_status']}   ({h['scope']})")
        print("Full evidence: the dashboard's Research tab, or python -m experiments.validate " + hid)


if __name__ == "__main__":
    main()
