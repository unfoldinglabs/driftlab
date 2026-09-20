"""Standardized experiment result and scoring.

Every run directory reduces to the same two reports, whatever the experiment,
world, or agent that produced the logs:

    ADAPTATION PROFILE   raw metrics per agent (latencies, accuracy, retention,
                         stale-memory rate, interference, calibration, cost)
    DRIFT PROFILE        the same metrics normalized to 0-1 and grouped into four
                         dimensions: adaptation, knowledge, epistemics, efficiency
    PAIRED EFFECTS       agent-vs-agent and regime-vs-regime deltas paired on the
                         shared seeded cells, with bootstrap confidence intervals

The vector is the result; the single `driftlab_score` (mean of the measured
capability dimensions — adaptation, knowledge, epistemics; efficiency is
reported beside it, never blended in) is a secondary convenience for ranking,
never a substitute for the profile. Metrics that a world or experiment does not measure (no confidence
elicited, no task keys, cost untracked for harness runs) stay NaN and are
shown as "—"; they are skipped, not zeroed, when scoring.

Like everything in metrics.py, profiles are computed from the JSONL trajectory
logs after the fact, never inside the run loop, so they apply to old runs and
are identical for in-process agents and external harnesses.

    python -m driftlab.profile runs/exp02/rule_world   # one run directory
    python -m driftlab.profile runs                    # every run directory under runs/

`report(run_dir)` also writes profile.json next to the logs: the standardized,
machine-readable result ({schema, world, agents: {metrics, dimensions, score}, runs}).
"""

import json
import math
import os
import time
from pathlib import Path

import numpy as np

from .metrics import change_events, collapse_tasks, collect_steps, lags_generic
from .paths import portable_path

SCHEMA_VERSION = 2  # v2: success prefers privileged truth (action == correct) over the reported
                    # reward, so lying feedback and pooled feedback are measured against reality;
                    # adds confidence_lag / confidence_lead and prediction_wrong_rate; the score
                    # averages the capability dimensions only (efficiency reported, not blended);
                    # a change with too few post-change encounters is unmeasured, not a zero

# Normalization constants (documented, deliberate, and arbitrary: they set the
# scale of a score, not the ordering of agents on any one metric).
LAG_HALF = 5.0        # a detection/recovery lag of 5 encounters scores 0.5
COST_HALF = 0.01      # $0.01 per successful step scores 0.5
FINAL_FRAC = 0.2      # "final" accuracy = last 20% of the episode
RETENTION_SKIP = 8    # retention: affected-task encounters after the first 8 post-change
RETENTION_SPAN = 30   #            ... up to 30 encounters/steps later
INTERFERENCE_W = 15   # interference: unaffected-task accuracy 15 steps before vs after
STALE_GRACE = 5       # stale-memory grace when the agent never gets the new rule right

METRICS = (  # (field, label, format)
    ("detection_lag", "detection latency", "{:.1f}"),
    ("recovery_lag", "recovery latency", "{:.1f}"),
    ("final_accuracy", "final accuracy", "{:.3f}"),
    ("retention", "retention", "{:.3f}"),
    ("stale_rate", "stale-memory rate", "{:.3f}"),
    ("overupdate_rate", "over-update rate", "{:.3f}"),
    ("interference", "interference", "{:+.3f}"),
    ("calibration_error", "calibration error", "{:.3f}"),
    ("overconfidence", "overconfidence", "{:+.3f}"),
    ("confidence_lag", "confidence lag", "{:.1f}"),
    ("confidence_lead", "confidence lead", "{:+.1f}"),
    ("prediction_wrong_rate", "wrong predictions", "{:.3f}"),
    ("parse_failure_rate", "parse failure rate", "{:.3f}"),
    ("cost_per_success", "cost per success ($)", "{:.4f}"),
    ("final_reward", "final mean reward", "{:.3f}"),
)
DIMENSIONS = ("adaptation", "knowledge", "epistemics", "efficiency")
CAPABILITY_DIMS = ("adaptation", "knowledge", "epistemics")  # what the driftlab score averages;
# efficiency (instruction-following and dollars) is reported beside the score, never blended in —
# token pricing must not move a capability ranking
FAIR_ENCOUNTERS = 4  # a change with fewer post-change encounters than this, and no recovery,
                     # is unmeasured rather than a zero: the agent never got a fair chance


def _nanmean(vals) -> float:
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _key(s: dict):
    k = s.get("task_key") or s.get("key")
    return tuple(k) if k else None


def _successes(steps: list[dict]) -> list[float] | None:
    """Per-step success. Worlds that log the privileged truth (`correct`) are scored
    against it — action == correct — so a world that lies in its feedback (noise) or
    withholds it (pooled reports) is still measured against reality. Worlds without
    a `correct` fall back to reward > 0; pure-cost worlds (never positive) get None."""
    if any(s.get("correct") is not None for s in steps):
        return [1.0 if s.get("action") == s["correct"] else 0.0 if s.get("correct") is not None else float("nan")
                for s in steps]
    if not any(s["reward"] > 0 for s in steps):
        return None
    return [1.0 if s["reward"] > 0 else 0.0 for s in steps]


def _retention(steps, succ, m) -> float:
    """Accuracy on the change's affected tasks well after the change, before those tasks change again."""
    stop = min((c["t"] for c in change_events(steps) if c["t"] > m["t"] and c["affected"] & m["affected"]),
               default=steps[-1]["t"] + 1)
    if m["affected"] and any(_key(s) for s in steps[:3]):
        idx = [i for i, s in enumerate(steps) if m["t"] < s["t"] < stop and _key(s) in m["affected"]]
        idx = idx[RETENTION_SKIP:RETENTION_SKIP + RETENTION_SPAN]
    else:
        idx = [i for i, s in enumerate(steps) if m["t"] + RETENTION_SKIP < s["t"] <= min(stop, m["t"] + RETENTION_SKIP + RETENTION_SPAN)]
    return _nanmean([succ[i] for i in idx]) if idx else float("nan")


def _stale_rate(steps, succ, m) -> float:
    """How often the agent still acts on the obsolete rule once it should know better:
    on affected tasks after the change, the share of encounters choosing the OLD action
    after the first post-change success (or after a grace period if it never succeeds)."""
    old = m.get("old")
    if old is None or not m["affected"]:
        return float("nan")
    enc = [i for i, s in enumerate(steps) if s["t"] > m["t"] and _key(s) in m["affected"]]
    first_ok = next((j for j, i in enumerate(enc) if succ[i] > 0), None)
    tail = enc[first_ok + 1:] if first_ok is not None else enc[STALE_GRACE:]
    return _nanmean([1.0 if steps[i]["action"] == old else 0.0 for i in tail]) if tail else float("nan")


def _overupdate_rate(steps) -> float:
    """The mirror image of stale_rate: after feedback wrongly punishes a correct action
    (possible only in worlds with misleading feedback), how often does the agent abandon
    the still-correct behavior at its next encounter with the same task? NaN when no
    such misleading events occurred — deterministic worlds never measure it."""
    by_key: dict = {}
    for i, s in enumerate(steps):
        k = _key(s)
        if k and s.get("correct") is not None:
            by_key.setdefault(k, []).append(i)
    events = []
    for idxs in by_key.values():
        for a, b in zip(idxs, idxs[1:]):
            sa, sb = steps[a], steps[b]
            if sa.get("action") == sa["correct"] and sa["reward"] <= 0 and sb.get("correct") == sa["correct"]:
                events.append(1.0 if sb.get("action") != sb["correct"] else 0.0)
    return _nanmean(events) if events else float("nan")


CONF_DROP = 0.15  # a confidence "sag" = falling this far below the pre-change baseline


def _confidence_lags(steps, changes, lags) -> tuple[float, float]:
    """confidence_lag: post-change encounters on affected tasks until stated confidence
    sags below its pre-change baseline. confidence_lead: recovery_lag minus confidence_lag
    per change — positive means confidence sagged before behavior recovered, i.e.
    confidence works as a change detector. Both NaN unless confidence was elicited."""
    per = []
    for m, lg in zip(changes, lags):
        after = [s for s in steps if s["t"] > m["t"] and s.get("confidence") is not None
                 and (not m["affected"] or _key(s) in m["affected"])]
        before = [s["confidence"] for s in steps if s["t"] <= m["t"] and s.get("confidence") is not None][-10:]
        if len(before) >= 2 and len(after) >= 2:
            thr = float(np.mean(before)) - CONF_DROP
            clag = next((float(i) for i, s in enumerate(after) if s["confidence"] < thr), float("nan"))
            per.append((clag, lg["recovery_lag"]))
    lag = _nanmean([c for c, _ in per])
    lead = _nanmean([r - c for c, r in per
                     if not math.isnan(c) and r is not None and not math.isnan(r)])
    return lag, lead


def _prediction_wrong_rate(steps) -> float:
    """When the agent states an outcome prediction (an EXPECT: success|failure line, e.g.
    the worldmodel substrate), the share of predictions the world then contradicted.
    NaN when the agent never predicts."""
    import re
    events = []
    for s in steps:
        m = re.search(r"EXPECT:\s*(success|failure)", s.get("reply") or "", re.IGNORECASE)
        if m:
            events.append(1.0 if (m.group(1).lower() == "success") != (s["reward"] > 0) else 0.0)
    return _nanmean(events) if len(events) >= 5 else float("nan")


def _interference(steps, succ, m) -> float:
    """Accuracy delta on UNaffected tasks around the change (negative = collateral damage)."""
    if not m["affected"] or not any(_key(s) for s in steps[:3]):
        return float("nan")
    before = [succ[i] for i, s in enumerate(steps) if m["t"] - INTERFERENCE_W <= s["t"] <= m["t"] and _key(s) not in m["affected"]]
    after = [succ[i] for i, s in enumerate(steps) if m["t"] < s["t"] <= m["t"] + INTERFERENCE_W and _key(s) not in m["affected"]]
    return _nanmean(after) - _nanmean(before) if before and after else float("nan")


def run_profile(header: dict, steps: list[dict]) -> dict:
    """The standardized per-run result: every metric NaN when not measured.
    Multi-step tasks (task_id in the log) are collapsed to one row per task first,
    so accuracy and lags count tasks, not the exchanges inside them."""
    raw_steps = len(steps)
    steps = collapse_tasks(steps)
    cell = header["cell"]
    succ = _successes(steps)
    changes = change_events(steps)
    lags = [lags_generic(steps, m) for m in changes]
    # a keyed change the agent barely met (few encounters, no recovery) is unmeasured, not a
    # failure; so is a change the agent was not competent before (pre_competent=False): an agent
    # that never learned the old rule has nothing to unlearn, and must not score as instantly adaptive
    keyed = any(_key(s) for s in steps[:3])
    fair = [l for l in lags
            if l.get("pre_competent", True)
            and (not keyed or not math.isnan(l["recovery_lag"]) or not math.isnan(l["detection_lag"])
                 or l.get("n_encounters", 0) >= FAIR_ENCOUNTERS)]
    conf = [(s["confidence"], 1.0 if s["reward"] > 0 else 0.0) for s in steps if s.get("confidence") is not None]
    c = np.array([x for x, _ in conf]); y = np.array([x for _, x in conf])
    conf_lag, conf_lead = _confidence_lags(steps, changes, lags) if conf and changes else (float("nan"), float("nan"))
    tail = steps[-max(10, int(len(steps) * FINAL_FRAC)):]
    cost = sum(d.get("cost_usd", 0.0) for d in header.get("cost", {}).values())
    n_success = float(np.nansum(succ)) if succ else 0
    return {
        "agent": cell["agent"]["name"], "regime": cell.get("regime"), "seed": cell.get("seed"),
        "agent_spec": {k: cell["agent"][k] for k in ("type", "model", "harness", "substrate")
                       if cell["agent"].get(k) is not None},
        "world": cell.get("world"), "n_steps": len(steps), "episode_steps": raw_steps, "n_changes": len(changes),
        "n_changes_scored": len(fair),
        "detection_lag": _nanmean([l["detection_lag"] for l in fair]),
        "recovery_lag": _nanmean([l["recovery_lag"] for l in fair]),
        "final_accuracy": _nanmean(succ[-len(tail):]) if succ else float("nan"),
        "final_reward": _nanmean([s["reward"] for s in tail]),
        "retention": _nanmean([_retention(steps, succ, m) for m in changes]) if succ and changes else float("nan"),
        "stale_rate": _nanmean([_stale_rate(steps, succ, m) for m in changes]) if succ and changes else float("nan"),
        "overupdate_rate": _overupdate_rate(steps),
        "interference": _nanmean([_interference(steps, succ, m) for m in changes]) if succ and changes else float("nan"),
        "calibration_error": float(np.mean((c - y) ** 2)) if len(conf) >= 5 else float("nan"),
        "overconfidence": float(c.mean() - y.mean()) if len(conf) >= 5 else float("nan"),
        "confidence_lag": conf_lag,
        "confidence_lead": conf_lead,
        "prediction_wrong_rate": _prediction_wrong_rate(steps),
        "parse_failure_rate": header.get("parse_failures", 0) / max(1, len(steps)),
        "cost_usd": cost,
        "cost_per_success": cost / n_success if cost > 0 and n_success else float("nan"),
    }


def _lag_score(lag: float, n_changes: float) -> float:
    """0-1; a NaN lag when changes did happen means 'never', which is worst, not unmeasured."""
    if not n_changes:
        return float("nan")
    return 0.0 if math.isnan(lag) else 1.0 / (1.0 + lag / LAG_HALF)


def dimension_scores(p: dict) -> dict:
    """Normalize a profile into the four 0-1 dimensions. NaN = not measured by these runs."""
    interference_score = float("nan") if math.isnan(p["interference"]) else float(np.clip(1.0 + 2.0 * min(0.0, p["interference"]), 0.0, 1.0))
    cost_score = float("nan") if math.isnan(p["cost_per_success"]) else 1.0 / (1.0 + p["cost_per_success"] / COST_HALF)
    calibration_score = float("nan") if math.isnan(p["calibration_error"]) else 1.0 - min(1.0, p["calibration_error"])
    stale_score = float("nan") if math.isnan(p["stale_rate"]) else 1.0 - p["stale_rate"]
    overupdate_score = float("nan") if math.isnan(p["overupdate_rate"]) else 1.0 - p["overupdate_rate"]
    n_scored = p.get("n_changes_scored", p["n_changes"])
    return {
        "adaptation": _nanmean([_lag_score(p["detection_lag"], n_scored), _lag_score(p["recovery_lag"], n_scored)]),
        "knowledge": _nanmean([p["final_accuracy"], p["retention"], interference_score]),
        "epistemics": _nanmean([calibration_score, stale_score, overupdate_score]),
        "efficiency": _nanmean([1.0 - p["parse_failure_rate"], cost_score]),
    }


def driftlab_score(dims: dict) -> float:
    """Secondary aggregate: mean of the measured CAPABILITY dimensions (adaptation,
    knowledge, epistemics). Efficiency — instruction-following and dollars — is reported
    beside the score, never blended in: token pricing must not move a capability
    ranking. Compare scores only between agents measuring the same dimensions; the
    vector is primary."""
    return _nanmean([dims[d] for d in CAPABILITY_DIMS if d in dims])


MIN_FULL_STEPS = 30  # runs shorter than this are quick-mode sense checks


def profile_dir(run_dir: str) -> dict:
    """Aggregate run profiles per agent: mean of each metric across runs, then scores.
    When the directory holds any full-length runs, quick-mode runs (under
    MIN_FULL_STEPS steps) are excluded from the aggregates and effects — their
    numbers are noise and would sit beside real measurements as if comparable.
    A directory of only quick runs still profiles (a plumbing check stays useful)."""
    all_runs = [run_profile(h, steps) for h, steps in collect_steps(run_dir)
                if h and steps and "reward" in steps[0]]  # skips non-trajectory logs (e.g. the dashboard's events.jsonl)
    full = [r for r in all_runs if r["n_steps"] >= MIN_FULL_STEPS]
    runs = full or all_runs
    agents = {}
    for name in sorted({r["agent"] for r in runs}):
        rs = [r for r in runs if r["agent"] == name]
        p = {f: _nanmean([r[f] for r in rs]) for f, _, _ in METRICS}
        p.update(n_changes=_nanmean([r["n_changes"] for r in rs]),
                 n_changes_scored=_nanmean([r.get("n_changes_scored", r["n_changes"]) for r in rs]),
                 cost_usd=sum(r["cost_usd"] for r in rs))
        dims = dimension_scores(p)
        agents[name] = {"n_runs": len(rs), "spec": next((r["agent_spec"] for r in rs if r.get("agent_spec")), {}),
                        "metrics": p, "dimensions": dims, "driftlab_score": driftlab_score(dims)}
    return {"schema": SCHEMA_VERSION, "run_dir": portable_path(run_dir), "generated": time.time(),
            "world": runs[0]["world"] if runs else None, "agents": agents,
            "excluded_quick": len(all_runs) - len(runs),
            "effects": all_effects(runs) if runs else None, "runs": all_runs}


# ---- paired effects ---------------------------------------------------------
# The runner crosses the same seeded cells over every agent, so paired
# comparisons exist in the logs already; these helpers stop throwing the
# pairing away. Report "B - A = +0.031 [CI], n pairs", not "A=81%, B=84%".

BOOT_N, BOOT_SEED = 2000, 0


def effect_field(runs: list[dict]) -> str:
    """The per-run outcome effects are computed on: accuracy where defined, else raw reward."""
    return "final_accuracy" if any(not math.isnan(r["final_accuracy"]) for r in runs) else "final_reward"


def paired_effects(runs: list[dict], field: str, vary: str, within: str | None = None) -> list[dict]:
    """Paired deltas of `field` between every pair of values of `vary` ('agent' or
    'regime'), matched on the remaining cell coordinates (seed, world, ...). Pass
    within='agent' to estimate a regime effect separately per agent. Each result:
    {vary, a, b, delta (mean of b-a), ci (95% bootstrap, n>=3), n}."""
    if within:
        return [{within: w, **e} for w in sorted({r[within] for r in runs})
                for e in paired_effects([r for r in runs if r[within] == w], field, vary)]
    coords = tuple(k for k in ("agent", "regime", "seed", "world") if k != vary)
    table: dict = {}
    for r in runs:
        v = r.get(field)
        if v is not None and not math.isnan(v):
            # episode_steps joins the key: a paired delta requires byte-identical episodes,
            # and a half-length run is a different episode than a full-length one
            key = tuple(r.get(k) for k in coords) + (r.get("episode_steps"),)
            table.setdefault(r[vary], {})[key] = v
    out, vals = [], sorted(table)
    for i, a in enumerate(vals):
        for b in vals[i + 1:]:
            shared = sorted(set(table[a]) & set(table[b]), key=str)
            deltas = np.array([table[b][k] - table[a][k] for k in shared])
            if len(deltas) < 2:
                continue
            ci = None
            if len(deltas) >= 3:
                rng = np.random.default_rng(BOOT_SEED)
                boots = deltas[rng.integers(0, len(deltas), (BOOT_N, len(deltas)))].mean(axis=1)
                ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
            out.append({"vary": vary, "a": a, "b": b, "delta": float(deltas.mean()), "ci": ci, "n": len(deltas)})
    return out


def all_effects(runs: list[dict]) -> dict:
    field = effect_field(runs)
    return {"field": field, "agent": paired_effects(runs, field, "agent"),
            "regime": paired_effects(runs, field, "regime", within="agent")}


def print_effects(effects: dict):
    rows = effects["agent"] + effects["regime"]
    if not rows:
        return
    print(f"\nPAIRED EFFECTS on {effects['field']}  (delta = b - a on matched cells; 95% bootstrap CI)")
    for e in rows:
        label = f"{e['b']} - {e['a']}" + (f"  ({e['agent']})" if e["vary"] == "regime" else "")
        ci = f"[{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}]" if e["ci"] else "[n<3]"
        print(f"  {label:<44} {e['delta']:+.3f}  {ci:<20} n={e['n']}")


def _fmt(v, spec="{:.3f}", width=12) -> str:
    s = "—" if v is None or (isinstance(v, float) and math.isnan(v)) else spec.format(v)
    return f"{s:>{width}}"


def print_profile(prof: dict):
    agents = prof["agents"]
    if not agents:
        print("no runs to profile")
        return
    names = list(agents)
    head = f"{'':<24}" + "".join(f"{n[:15]:>16}" for n in names)
    if prof.get("excluded_quick"):
        print(f"({prof['excluded_quick']} smoke run(s) under {MIN_FULL_STEPS} steps excluded from the aggregates)")
    print("ADAPTATION PROFILE  (mean per agent across runs; — = not measured by this world/experiment)")
    print(head + "\n" + "-" * len(head))
    for f, label, spec in METRICS:
        print(f"{label:<24}" + "".join(_fmt(agents[n]["metrics"][f], spec, 16) for n in names))
    print("\nDRIFT PROFILE  (0-1 per dimension; the score averages the capability dimensions —")
    print("adaptation, knowledge, epistemics — with efficiency reported beside it, never blended in)")
    print(head + "\n" + "-" * len(head))
    for d in DIMENSIONS:
        print(f"{d:<24}" + "".join(_fmt(agents[n]["dimensions"][d], "{:.2f}", 16) for n in names))
    print(f"{'driftlab score':<24}" + "".join(_fmt(agents[n]["driftlab_score"], "{:.2f}", 16) for n in names))
    if prof.get("effects"):
        print_effects(prof["effects"])


def report(run_dir: str, save: bool = True) -> dict:
    """Print the standardized reports for a run directory and write profile.json."""
    prof = profile_dir(run_dir)
    print_profile(prof)
    if save and prof["agents"]:
        (Path(run_dir) / "profile.json").write_text(json.dumps(prof, indent=2))
        print(f"\nprofile.json written to {os.path.relpath(run_dir)}")
    return prof


if __name__ == "__main__":
    import sys
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
    for d in sorted({p.parent for p in root.rglob("*.jsonl") if p.parent.name != "llm_cache"}):
        prof = profile_dir(str(d))
        if prof["agents"]:
            print(f"\n=== {d} ===")
            print_profile(prof)
            (d / "profile.json").write_text(json.dumps(prof, indent=2))
