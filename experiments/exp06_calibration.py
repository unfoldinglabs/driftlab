"""Experiment 6 — Calibration as a continual-learning signal.

The world asks for a confidence (0-100) with every action. Six substantive
changes. Is confidence calibrated (Brier, overconfidence gap)? Does it fall on
affected tasks after a change, and how quickly (confidence lag) compared with
behavioral recovery?
Works on: any world.

    python -m experiments.exp06_calibration [--world inventory]
    python -m experiments.exp06_calibration --analyze
"""

import numpy as np

from experiments.common import NOTES, TRANSCRIPT, q, ref, run_experiment, scheduled, spaced, world_for
from driftlab.core import Scenario
from driftlab.metrics import change_events, collect_steps, lags_generic, print_table, summarize_by
from driftlab.runner import env_cells

T = q(120)
REFERENCE_AGENTS = [ref("llm_transcript", TRANSCRIPT), ref("llm_notes", NOTES)]


def cells(seeds):
    return env_cells(["changing"], seeds)


def scenario(cell, ctx):
    world = world_for(cell, T, needs=("change_latent", "ask_confidence"), ask_confidence=True)
    return Scenario(world, T=T, before_step=scheduled(latent=spaced(6, T, q(20))), seed=cell["seed"])


def analyze(run_dir):
    rows = []
    for h, steps in collect_steps(run_dir):
        st = [s for s in steps if s.get("confidence") is not None]
        if len(st) < q(20):
            continue
        c = np.array([s["confidence"] for s in st])
        y = np.array([1.0 if s["reward"] > 0 else 0.0 for s in st])
        conf_lags, rec_lags = [], []
        for m in change_events(steps):
            after = [s for s in steps if s["t"] > m["t"] and s.get("confidence") is not None and (not m["affected"] or tuple(s.get("task_key") or []) in m["affected"])]
            before = [s["confidence"] for s in steps if s["t"] <= m["t"] and s.get("confidence") is not None][-10:]
            if len(before) >= 2 and len(after) >= 2:
                thr = np.mean(before) - 0.15
                conf_lags.append(next((i for i, s in enumerate(after) if s["confidence"] < thr), float("nan")))
                rec_lags.append(lags_generic(steps, m)["recovery_lag"])
        rows.append({"agent": h["cell"]["agent"]["name"], "brier": float(np.mean((c - y) ** 2)),
                     "overconfidence": float(c.mean() - y.mean()),
                     "conf_acc_corr": float(np.corrcoef(c, y)[0, 1]) if c.std() > 0 and y.std() > 0 else float("nan"),
                     "confidence_lag": float(np.nanmean(conf_lags)) if conf_lags else float("nan"),
                     "recovery_lag": float(np.nanmean(rec_lags)) if rec_lags else float("nan")})
    f = ("brier", "overconfidence", "conf_acc_corr", "confidence_lag", "recovery_lag")
    print_table(summarize_by(rows, ("agent",), f), ("agent",), f)


if __name__ == "__main__":
    run_experiment("exp06", __doc__, cells, scenario, REFERENCE_AGENTS, analyze, config={"T": T})
