"""CI-based hypothesis validation: evaluate the registry's machine-checkable
predictions against the run logs and propose scope-qualified verdicts.

A prediction is a directional claim (registry.py, `predictions`): an `effect`
or `reversal` over paired deltas, or a `sign` claim that a metric's mean sits
on one side of zero. All are evaluated with the same bootstrap machinery the
profile report uses. Per prediction and agent:

    supported      the 95% CI of the paired delta excludes 0 in the predicted
                   direction, with at least --min-pairs matched cells (default 3:
                   at small n the bootstrap interval is wide, so only large
                   effects turn conclusive early; 5+ seeds is the confirmatory
                   standard, and n is printed with every verdict)
    contradicted   the CI excludes 0 in the opposite direction
    inconclusive   the CI includes 0, or there is too little paired data

Per hypothesis, verdicts aggregate into a proposed status, always scoped to
the agents and worlds actually tested: SUPPORTED (everything conclusive
agrees), NOT SUPPORTED (everything conclusive disagrees), PARTIALLY SUPPORTED
(both appear), INCONCLUSIVE (nothing conclusive). Quick-mode runs are
excluded automatically. The proposal never edits the research log by itself:
statistical support is not interpretation. `--apply` updates each hypothesis
file's Status line and appends evidence rows; without it the report prints and
runs/validation.json is written for the dashboard.

    python -m experiments.validate                # every hypothesis, every world found
    python -m experiments.validate H003           # one hypothesis
    python -m experiments.validate exp15          # one experiment's predictions
    python -m experiments.validate --apply        # also update research/hypotheses/*.md
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.registry import HYPOTHESES_DIR, REGISTRY  # noqa: E402
from driftlab.profile import paired_effects, profile_dir  # noqa: E402

MIN_STEPS = 30  # runs shorter than this are treated as quick-mode sense checks


def _run_identity(row: dict) -> tuple:
    """Collapse the historical model-suffixed arm names to one identity.

    Early survey runs used compact names such as dsv41flash_api_low_notes;
    later extensions used names such as llm_notes-deepseek-v4.1-flash.
    They are the same model/substrate arm for scale selection, although the
    original names remain intact for prediction matching and display.
    """
    agent = row["agent"]
    if "deepseek" in agent or agent.startswith("dsv41"):
        model = "deepseek"
    elif "qwen" in agent:
        model = "qwen"
    elif "luna" in agent or agent.startswith("llm_"):
        model = "luna"
    else:
        return ("named", agent)
    if "transcript" in agent:
        substrate = "transcript"
    elif "notes" in agent:
        substrate = "notes"
    else:
        substrate = agent
    return ("reference", model, substrate)


def _conclusive(delta, ci, direction):
    """supported / contradicted / inconclusive for a b-minus-a delta and its CI."""
    if ci is None:
        return "inconclusive"
    lo, hi = ci
    if direction == ">":
        return "supported" if lo > 0 else ("contradicted" if hi < 0 else "inconclusive")
    if direction == "<":
        return "supported" if hi < 0 else ("contradicted" if lo > 0 else "inconclusive")
    return "supported" if (lo > 0 or hi < 0) else "inconclusive"  # "differs"


def _oriented(effects, a, b):
    """Find the (a, b) pair in paired_effects output, orienting delta and CI as b - a."""
    for e in effects:
        if e["a"] == a and e["b"] == b:
            return e["delta"], e["ci"], e["n"]
        if e["a"] == b and e["b"] == a:
            ci = [-e["ci"][1], -e["ci"][0]] if e["ci"] else None
            return -e["delta"], ci, e["n"]
    return None, None, 0


def _full_length_rows(run_dir: Path) -> list[dict]:
    """Return valid rows at each agent's longest completed episode length.

    A protocol directory can contain an earlier half-length read and a later
    full-length extension.  Treating both as one pool lets the earlier read
    determine a hypothesis status after the extension has failed to replicate.
    Keep the validity floor, then select the longest episode length separately
    for each agent; agents with only the shorter setting remain reportable, but
    cannot be mistaken for a matched full-length extension.
    """
    prof = profile_dir(str(run_dir))
    rows = [r for r in prof["runs"] if r["n_steps"] >= MIN_STEPS]
    longest = {}
    for r in rows:
        key = _run_identity(r)
        longest[key] = max(longest.get(key, 0), r.get("episode_steps") or 0)
    return [r for r in rows
            if (r.get("episode_steps") or 0) == longest[_run_identity(r)]]


def evaluate_prediction(pred: dict, rows: list[dict], min_pairs: int) -> list[dict]:
    """One prediction against one directory's rows -> verdict rows (one per agent, or one
    pooled), each with a `reason` stating the basis for the verdict in plain language."""
    out = []
    field, kind = pred.get("field"), pred.get("kind")
    if kind == "effect" and pred.get("within") == "agent":
        for agent in sorted({r["agent"] for r in rows}):
            fx = paired_effects([r for r in rows if r["agent"] == agent], field, pred["vary"])
            delta, ci, n = _oriented(fx, pred["a"], pred["b"])
            if n == 0:
                verdict, reason = "inconclusive", f"this agent has no runs matched across '{pred['a']}' and '{pred['b']}' on the same seeds"
            elif n < min_pairs:
                verdict, reason = "inconclusive", f"only {n} matched pairs; {min_pairs} needed before a verdict is allowed"
            else:
                verdict = _conclusive(delta, ci, pred["direction"])
                want = "above" if pred["direction"] == ">" else "below"
                reason = (f"the 95% CI of {field}({pred['b']}) − {field}({pred['a']}) is entirely {want} zero across {n} seed-matched pairs" if verdict == "supported" else
                          f"the 95% CI is entirely on the other side of zero: the data points the opposite way" if verdict == "contradicted" else
                          f"the 95% CI includes zero: {n} pairs cannot separate '{pred['b']}' from '{pred['a']}' on {field}")
            out.append({"agent": agent, "delta": delta, "ci": ci, "n": n, "verdict": verdict, "reason": reason})
    elif kind == "effect" and pred.get("direction") in (">", "<"):  # a specific, oriented agent pair
        fx = paired_effects(rows, field, "agent")
        delta, ci, n = _oriented(fx, pred["a"], pred["b"])
        label = f"{pred['b']} vs {pred['a']}"
        if n == 0:
            verdict, reason = "inconclusive", f"no runs matched across '{pred['a']}' and '{pred['b']}' on the same seeds"
        elif n < min_pairs:
            verdict, reason = "inconclusive", f"only {n} matched pairs; {min_pairs} needed before a verdict is allowed"
        else:
            verdict = _conclusive(delta, ci, pred["direction"])
            want = "above" if pred["direction"] == ">" else "below"
            reason = (f"the 95% CI of {field}({pred['b']}) − {field}({pred['a']}) is entirely {want} zero across {n} seed-matched pairs" if verdict == "supported" else
                      "the 95% CI is entirely on the other side of zero: the data points the opposite way" if verdict == "contradicted" else
                      f"the 95% CI includes zero: {n} pairs cannot separate '{pred['b']}' from '{pred['a']}' on {field}")
        out.append({"agent": label, "delta": delta, "ci": ci, "n": n, "verdict": verdict, "reason": reason})
    elif kind == "effect":  # vary == "agent", direction "differs": any agent pair separates
        fx = [e for e in paired_effects(rows, field, "agent") if e["n"] >= min_pairs]
        hits = [e for e in fx if e["ci"] and (e["ci"][0] > 0 or e["ci"][1] < 0)]
        if hits:
            e = max(hits, key=lambda e: abs(e["delta"]))
            out.append({"agent": f"{e['b']} vs {e['a']}", "delta": e["delta"], "ci": e["ci"], "n": e["n"], "verdict": "supported",
                        "reason": f"at least one agent pair separates on {field}: the 95% CI of the paired delta excludes zero"})
        elif fx:
            out.append({"agent": "all pairs", "delta": None, "ci": None, "n": max(e["n"] for e in fx), "verdict": "contradicted",
                        "reason": f"every agent pair's 95% CI on {field} includes zero: no architecture separates from another on these seeds"})
        else:
            out.append({"agent": "all pairs", "delta": None, "ci": None, "n": 0, "verdict": "inconclusive",
                        "reason": f"no two agents share {min_pairs}+ seed-matched runs here, so no pair can be compared"})
    elif kind == "sign":  # a metric's mean sits on one side of zero, per agent (no pairing)
        import numpy as np
        for agent in sorted({r["agent"] for r in rows}):
            vals = [r[field] for r in rows
                    if r["agent"] == agent and (pred.get("regime") is None or r["regime"] == pred["regime"])
                    and r.get(field) is not None and not math.isnan(r[field])]
            n = len(vals)
            if n < min_pairs:
                out.append({"agent": agent, "delta": float(np.mean(vals)) if vals else None, "ci": None, "n": n,
                            "verdict": "inconclusive",
                            "reason": f"only {n} runs measure {field} here; {min_pairs} needed before a verdict is allowed"})
                continue
            arr = np.array(vals)
            rng = np.random.default_rng(0)
            boots = arr[rng.integers(0, n, (2000, n))].mean(axis=1)
            ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
            verdict = _conclusive(float(arr.mean()), ci, pred["direction"])
            want = "above" if pred["direction"] == ">" else "below"
            reason = (f"the 95% CI of mean {field} is entirely {want} zero across {n} runs" if verdict == "supported" else
                      "the 95% CI is entirely on the other side of zero: the data points the opposite way" if verdict == "contradicted" else
                      f"the 95% CI of mean {field} includes zero across {n} runs")
            out.append({"agent": agent, "delta": float(arr.mean()), "ci": ci, "n": n, "verdict": verdict, "reason": reason})
    elif kind == "reversal":  # an agent-pair ranking that flips between two regimes
        fa = {(e["a"], e["b"]): e for e in paired_effects([r for r in rows if r["regime"] == pred["a_regime"]], field, "agent") if e["n"] >= min_pairs}
        fb = {(e["a"], e["b"]): e for e in paired_effects([r for r in rows if r["regime"] == pred["b_regime"]], field, "agent") if e["n"] >= min_pairs}
        found = None
        for k in set(fa) & set(fb):
            ea, eb = fa[k], fb[k]
            if ea["ci"] and eb["ci"] and ((ea["ci"][0] > 0 and eb["ci"][1] < 0) or (ea["ci"][1] < 0 and eb["ci"][0] > 0)):
                found = (k, ea, eb)
                break
        if found:
            (a, b), ea, eb = found
            out.append({"agent": f"{b} vs {a}", "delta": eb["delta"] - ea["delta"], "ci": None,
                        "n": min(ea["n"], eb["n"]), "verdict": "supported",
                        "reason": f"this pair's {field} effect is significantly positive in '{pred['a_regime']}' and significantly negative in '{pred['b_regime']}' (or vice versa): the ranking flips"})
        else:
            shared = set(fa) & set(fb)
            out.append({"agent": "all pairs", "delta": None, "ci": None,
                        "n": min((min(fa[k]["n"], fb[k]["n"]) for k in shared), default=0),
                        "verdict": "contradicted" if shared else "inconclusive",
                        "reason": (f"agent pairs are measurable in both '{pred['a_regime']}' and '{pred['b_regime']}', and none reverses sign" if shared else
                                   f"no agent pair has {min_pairs}+ matched runs in both '{pred['a_regime']}' and '{pred['b_regime']}'")})
    return out


def run_validation(targets: set | None, min_pairs: int) -> dict:
    result = {"generated": time.time(), "min_pairs": min_pairs, "min_steps": MIN_STEPS,
              "hypotheses": {}, "experiments": {}}
    for exp, entry in REGISTRY.items():
        preds = entry.get("predictions", [])
        if targets and exp not in targets and not (set(entry["hypotheses"]) & targets):
            continue
        rows_out = []
        for wdir in sorted((ROOT / "runs" / exp).glob("*")) if (ROOT / "runs" / exp).is_dir() else []:
            if not wdir.is_dir() or not list(wdir.glob("*.jsonl")):
                continue
            rows = _full_length_rows(wdir)
            for pred in preds:
                verdicts = evaluate_prediction(pred, rows, min_pairs) if rows else \
                    [{"agent": "—", "delta": None, "ci": None, "n": 0, "verdict": "inconclusive",
                      "reason": f"no valid runs in runs/{exp}/{wdir.name} (runs under {MIN_STEPS} scored steps never count)"}]
                for v in verdicts:
                    rows_out.append({"exp": exp, "version": entry["version"], "world": wdir.name,
                                     "claim": pred["claim"], **v})
        result["experiments"][exp] = {"predictions": len(preds), "rows": rows_out}
        for h in entry["hypotheses"]:
            result["hypotheses"].setdefault(h, {"rows": []})["rows"].extend(rows_out)
    for h, data in result["hypotheses"].items():
        data["predicted_by"] = [e for e, ent in REGISTRY.items() if h in ent["hypotheses"] and ent.get("predictions")]
        vs = [r["verdict"] for r in data["rows"]]
        sup, con = vs.count("supported"), vs.count("contradicted")
        data["proposed_status"] = ("SUPPORTED" if sup and not con else
                                   "NOT SUPPORTED" if con and not sup else
                                   "PARTIALLY SUPPORTED" if sup and con else "INCONCLUSIVE")
        agents = sorted({r["agent"] for r in data["rows"] if r["verdict"] != "inconclusive"})
        worlds = sorted({r["world"] for r in data["rows"] if r["verdict"] != "inconclusive"})
        data["scope"] = (f"{sup} supported, {con} contradicted, {vs.count('inconclusive')} inconclusive"
                         + (f" · agents: {', '.join(agents)}" if agents else "")
                         + (f" · worlds: {', '.join(worlds)}" if worlds else ""))
    return result


def print_report(result: dict):
    print(f"CI-based validation (min {result['min_pairs']} matched pairs; runs under "
          f"{result['min_steps']} steps excluded as quick mode)\n")
    for h in sorted(result["hypotheses"]):
        data = result["hypotheses"][h]
        print(f"{h}: proposed {data['proposed_status']}   [{data['scope']}]")
        for r in data["rows"]:
            ci = f" CI [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]" if r.get("ci") else ""
            delta = f" delta {r['delta']:+.3f}" if r.get("delta") is not None else ""
            print(f"  {r['verdict']:<13} {r['exp']} v{r['version']} · {r['world']} · {r['agent']}"
                  f"{delta}{ci} · n={r['n']}\n                claim: {r['claim']}\n                basis: {r['reason']}")
        if not data["rows"]:
            print(f"  predictions registered ({', '.join(data['predicted_by'])}) but no full-length runs yet"
                  if data["predicted_by"] else "  no machine-checkable predictions registered for its experiments yet")
        print()
    print("A proposal, not a finding: statuses in research/hypotheses/ change only with --apply,")
    print("and interpretation stays yours.")


def apply_to_files(result: dict):
    for h, data in result["hypotheses"].items():
        files = list(HYPOTHESES_DIR.glob(f"{h}-*.md"))
        if not files or data["proposed_status"] == "INCONCLUSIVE":
            continue
        text = files[0].read_text()
        lines = text.splitlines()
        stamp = time.strftime("%Y-%m-%d")
        for i, line in enumerate(lines):
            if line.lower().startswith("status:"):
                lines[i] = f"Status: {data['proposed_status']} (auto-evaluated {stamp}; confirm interpretation)"
        new_rows = [f"| {r['exp']} v{r['version']} | {r['world']} | {r['agent']}, n={r['n']} | {r['verdict']} | "
                    f"{'delta %+.3f' % r['delta'] if r['delta'] is not None else '—'}"
                    f"{' CI [%+.3f, %+.3f]' % tuple(r['ci']) if r.get('ci') else ''} (auto {stamp}) |"
                    for r in data["rows"] if r["verdict"] != "inconclusive"]
        for row in new_rows:
            if row not in text:
                idx = max(i for i, ln in enumerate(lines) if ln.startswith("|"))
                lines.insert(idx + 1, row)
                text = "\n".join(lines)
        files[0].write_text("\n".join(lines) + "\n")
        print(f"updated {files[0].relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", nargs="?", help="HNNN or expNN (default: everything)")
    from experiments.common import MIN_PAIRS
    ap.add_argument("--min-pairs", type=int, default=MIN_PAIRS,
                    help="matched cells required before a verdict is attempted (small n = wide CI: "
                         "only large effects turn conclusive early)")
    ap.add_argument("--apply", action="store_true", help="update Status + evidence in research/hypotheses/")
    args = ap.parse_args()
    targets = {args.target} if args.target else None
    result = run_validation(targets, args.min_pairs)
    print_report(result)
    out = ROOT / "runs" / "validation.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=lambda o: None))
    print(f"\nwritten to {out.relative_to(ROOT)} (the dashboard's Research tab reads it)")
    if args.apply:
        apply_to_files(result)
