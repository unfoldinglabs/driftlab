"""Cost reporting.

    python -m driftlab.costs runs/exp01        # one run directory
    python -m driftlab.costs runs               # every experiment under runs/

Reads the per-run cost blocks the runner writes into each manifest.json.
Per-run blocks written before the per-episode attribution fix overstate cost
under --concurrency (each block absorbed its concurrent siblings' calls);
grid-level totals were always correct.
"""

import json
import sys
from pathlib import Path

from .agents.brain import CostLedger


def summarize_manifest(path: Path) -> dict:
    data = json.loads(path.read_text())
    totals: dict = {}
    for run in data.get("runs", []):
        for model, d in run.get("cost", {}).items():
            t = totals.setdefault(model, {f: 0 for f in CostLedger.FIELDS})
            for f in CostLedger.FIELDS:
                t[f] += d.get(f, 0)
    return {"runs": len(data.get("runs", [])), "by_model": totals}


def print_report(root: Path):
    manifests = sorted(root.rglob("manifest.json"))
    if not manifests:
        print(f"no manifests under {root}")
        return
    grand = 0.0
    for m in manifests:
        s = summarize_manifest(m)
        print(f"\n{m.parent}  ({s['runs']} runs)")
        print(f"  {'model':<14}{'calls':>7}{'cached':>8}{'in_tok':>10}{'out_tok':>10}{'reason':>10}{'cost_usd':>10}{'saved_usd':>11}")
        for model, t in s["by_model"].items():
            print(f"  {model:<14}{t['calls']:>7}{t['cache_hits']:>8}{t['input_tokens']:>10}{t['output_tokens']:>10}"
                  f"{t['reasoning_tokens']:>10}{t['cost_usd']:>10.4f}{t['cache_saved_usd']:>11.4f}")
            grand += t["cost_usd"]
    print(f"\nTotal spent across manifests: ${grand:.4f}")


if __name__ == "__main__":
    print_report(Path(sys.argv[1] if len(sys.argv) > 1 else "runs"))
