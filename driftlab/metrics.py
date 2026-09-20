"""Post-hoc metric helpers. Metrics are computed from JSONL trajectory logs,
never inside the run loop, so new metrics can be applied to old runs, and they
are identical whether the agent was in-process or an external harness."""

import json
from pathlib import Path

import numpy as np


def load_run(path: str) -> tuple[dict, list[dict]]:
    header, steps = None, []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["kind"] == "header":
                header = rec
            else:
                steps.append(rec)
    return header, steps


def collect_steps(run_dir: str):
    for p in sorted(Path(run_dir).glob("*.jsonl")):
        yield load_run(str(p))


def rolling_mean(x, w: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return np.array([x.mean()] * len(x)) if len(x) else x
    c = np.cumsum(np.insert(x, 0, 0))
    return np.concatenate([np.full(w - 1, np.nan), (c[w:] - c[:-w]) / w])


def summarize_by(rows: list[dict], keys: tuple, fields: tuple) -> dict:
    groups: dict = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = {}
    for g, rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        out[g] = {"n": len(rs)}
        for f in fields:
            vals = [r[f] for r in rs if r.get(f) is not None and np.isfinite(r[f])]
            out[g][f] = float(np.mean(vals)) if vals else float("nan")
    return out


def print_table(summary: dict, keys: tuple, fields: tuple, width: int = 16):
    w = {f: max(12, len(f) + 2) for f in fields}  # a name longer than the column must widen it
    head = "".join(f"{k:<{width}}" for k in keys) + f"{'n':>4}" + "".join(f"{f:>{w[f]}}" for f in fields)
    print(head)
    print("-" * len(head))
    for g, s in summary.items():
        row = "".join(f"{str(v):<{width}}" for v in g) + f"{s['n']:>4}"
        print(row + "".join(f"{s[f]:>{w[f]}.3f}" for f in fields))


def collapse_tasks(steps: list[dict]) -> list[dict]:
    """For logs where one task spans several steps (worlds that report task_id): one row per
    task, taking the resolving step's outcome and merging the changes seen along the way.
    Logs without task_id pass through unchanged, so all step-level metrics stay valid."""
    if not any("task_id" in s for s in steps):
        return steps
    out: list[dict] = []
    cur: dict | None = None
    for s in steps:
        if cur is not None and s.get("task_id") == cur.get("task_id"):
            cur = {**s, "changes": cur.get("changes", []) + s.get("changes", [])}
        else:
            if cur is not None:
                out.append(cur)
            cur = dict(s)
    if cur is not None:
        out.append(cur)
    return out


# ---- RuleWorld change metrics ------------------------------------------------------

def change_events(steps: list[dict], kinds=("latent", "endogenous")) -> list[dict]:
    """Flatten logged world changes of the given kinds: [{t, kind, desc, affected:set, ...}]."""
    out = []
    for s in steps:
        for c in s.get("changes", []):
            if c.get("kind") in kinds:
                out.append({**c, "affected": {tuple(a) for a in c.get("affected", [])}})
    return out


def mutation_events(steps: list[dict]) -> list[dict]:
    """Substantive changes (latent + endogenous); legacy name."""
    return change_events(steps)


# Lags are conditional on prior competence: an agent that never learned the old rule has
# nothing to unlearn and would score as instantly adaptive. A change is only measured if the
# agent was correct on at least MIN_PRE_COMPETENCE of its last PRE_WINDOW pre-change
# encounters with the affected keys (at least MIN_PRE_ENCOUNTERS of them); otherwise its
# lags are unmeasured, never zero.
MIN_PRE_COMPETENCE = 0.5
MIN_PRE_ENCOUNTERS = 2
PRE_WINDOW = 8


def lags_after_mutation(steps: list[dict], m: dict, horizon: int = 40) -> dict:
    """Encounter-based lags on the keys a flip invalidated:
        detection_lag  encounters until the agent stops choosing the OLD option
        recovery_lag   encounters until the first correct choice
        stable_lag     encounters until two correct choices in a row
    All NaN (pre_competent=False) when the agent was not competent before the change."""
    enc = [s for s in steps if m["t"] < s["t"] <= m["t"] + horizon and tuple(s.get("task_key") or s.get("key") or []) in m["affected"]]
    pre = [s for s in steps if s["t"] <= m["t"] and tuple(s.get("task_key") or s.get("key") or []) in m["affected"]][-PRE_WINDOW:]
    ok = [(s["action"] == s["correct"]) if s.get("correct") is not None else s["reward"] > 0 for s in pre]
    if len(ok) < MIN_PRE_ENCOUNTERS or sum(ok) / len(ok) < MIN_PRE_COMPETENCE:
        return {"detection_lag": float("nan"), "recovery_lag": float("nan"), "stable_lag": float("nan"),
                "n_encounters": len(enc), "pre_competent": False}
    det = rec = stab = float("nan")
    streak = 0
    old = m.get("old")
    for i, s in enumerate(enc):
        if det != det and (old is None or s["action"] != old):
            det = i
        if s["reward"] > 0:
            streak += 1
            if rec != rec:
                rec = i
            if streak >= 2 and stab != stab:
                stab = i
        else:
            streak = 0
    return {"detection_lag": det, "recovery_lag": rec, "stable_lag": stab, "n_encounters": len(enc), "pre_competent": True}


def dip_after(rewards, t: int, window: int = 10) -> float:
    r = np.asarray(rewards, dtype=float)
    return float(r[max(0, t - window):t].mean() - r[t:t + window].mean())


def steps_to_streak(rewards, streak: int = 10) -> float:
    """First step index at which the next `streak` rewards are all positive (learning speed). NaN if never."""
    r = [1 if x > 0 else 0 for x in rewards]
    for i in range(len(r) - streak + 1):
        if sum(r[i:i + streak]) == streak:
            return float(i)
    return float("nan")


def recovery_time(steps: list[dict], change_t: int, pre: int = 10, post: int = 30, w: int = 5) -> float:
    """World-agnostic recovery: steps after a change until rolling reward (window w) regains the
    pre-change rolling level minus half the post-change dip. NaN if never within `post` steps."""
    r = np.asarray([s["reward"] for s in steps], dtype=float)
    if change_t < w or change_t + w >= len(r):
        return float("nan")
    before = r[max(0, change_t - pre):change_t].mean()
    seg = r[change_t:change_t + post]
    roll = rolling_mean(seg, w)
    trough = np.nanmin(roll) if np.any(~np.isnan(roll)) else seg.mean()
    target = trough + 0.5 * (before - trough)
    for i, v in enumerate(roll):
        if not np.isnan(v) and v >= target:
            return float(i)
    return float("nan")


def lags_generic(steps: list[dict], m: dict, horizon: int = 40) -> dict:
    """Encounter-based lags when the world reports task keys, else time-based recovery."""
    if m.get("affected") and any(s.get("task_key") or s.get("key") for s in steps[:3]):
        return lags_after_mutation(steps, m, horizon)
    return {"detection_lag": float("nan"), "recovery_lag": recovery_time(steps, m["t"]), "stable_lag": float("nan"), "n_encounters": 0}
