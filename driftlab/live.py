"""Live event bus: every step, world event and running metric is appended as a
JSON line to an events file (always), and optionally echoed to the terminal
(`--live`). The visual dashboard (`python -m driftlab.viz.server`) tails the
same file, so it works identically for in-process runs, for external harnesses
playing through the MCP server (a separate process appending to the same
file), and for replaying finished runs.

The current run's label is carried in a contextvar so worlds, agents and
substrates can report without threading a run id through every signature.
"""

import contextvars
import json
import os
import time
from pathlib import Path

import numpy as np

from .agents.brain import LEDGER

current_run = contextvars.ContextVar("driftlab_run", default="run")
ROOT = Path(__file__).resolve().parents[1]
EVENTS_PATH = Path(os.environ.get("DRIFTLAB_EVENTS", ROOT / "runs" / "events.jsonl"))


class LiveReporter:
    def __init__(self):
        self.enabled = False       # terminal echo
        self.every = 10
        self.write_events = True   # events file for the dashboard
        self.runs: dict = {}

    # ---- plumbing -----------------------------------------------------------------
    def _state(self):
        label = current_run.get()
        return label, self.runs.setdefault(label, {"rewards": [], "parse_fail": 0, "t0": time.time()})

    def _emit(self, kind: str, **data):
        if not self.write_events:
            return
        rec = {"ts": time.time(), "run": current_run.get(), "kind": kind, **data}
        EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with EVENTS_PATH.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def _say(self, text: str):
        if self.enabled:
            print(text, flush=True)

    # ---- lifecycle ----------------------------------------------------------------
    def run_start(self, cell: dict, T: int, instructions: str, experiment: str | None = None, world: str | None = None):
        label, st = self._state()
        st.update(rewards=[], parse_fail=0, t0=time.time())
        self._emit("run_start", cell=cell, T=T, instructions=instructions, experiment=experiment, world=world)
        self._say(f"[{label}] ** start  agent={cell.get('agent', {}).get('name')} regime={cell.get('regime')} seed={cell.get('seed')}")

    def run_end(self, summary: dict):
        label, _ = self._state()
        self._emit("run_end", **summary)
        self._say(f"[{label}] ** done   " + "  ".join(f"{k}={v}" for k, v in summary.items() if k in ("steps", "mean_reward", "wall_s", "cost_usd")))

    def analysis(self, experiment: str, text: str):
        self._emit("analysis", experiment=experiment, text=text)

    # ---- per step -----------------------------------------------------------------
    def step(self, t: int, T: int, rec: dict):
        label, st = self._state()
        st["rewards"].append(rec["reward"])
        st["parse_fail"] += int(rec.get("parse_failed", False))
        payload = {k: rec.get(k) for k in ("observation", "action", "reward", "feedback", "parse_failed", "notice",
                                           "key", "correct", "depth", "confidence", "act_s", "changes", "mutations", "reply")}
        # world-specific change markers the dashboard can use
        for k in ("version", "season", "session", "true_rate", "stock_end", "lost", "task"):
            if k in rec:
                payload[k] = rec[k]
        self._emit("step", t=t, T=T, **payload)
        if self.enabled:
            obs = " ".join(str(rec["observation"]).split())
            fb = " ".join(str(rec["feedback"]).split())
            print(f"[{label}] t={t + 1:>3}/{T}  {obs[:48]}{'…' if len(obs) > 48 else ''} | did: {str(rec['action'])[:28]}"
                  f"  -> {rec['reward']:+.2f}  {fb[:60]}{'…' if len(fb) > 60 else ''}", flush=True)
            if (t + 1) % self.every == 0:
                self.metrics()

    def metrics(self, **extra):
        label, st = self._state()
        r = st["rewards"]
        m = {"t": len(r), "roll10": float(np.mean(r[-10:])) if r else None, "cum": float(np.mean(r)) if r else None,
             "parse_fail": st["parse_fail"], "spent_usd": LEDGER.total_cost(), "elapsed_s": time.time() - st["t0"], **extra}
        self._emit("metrics", **m)
        if self.enabled:
            parts = [f"t={m['t']}"] + ([f"roll10 {m['roll10']:.2f}", f"cum {m['cum']:.2f}"] if r else [])
            parts += [f"parse_fail {m['parse_fail']}", f"spent ${m['spent_usd']:.4f}", f"{m['elapsed_s']:.0f}s"]
            parts += [f"{k} {v}" for k, v in extra.items()]
            print(f"[{label}] -- " + "  ".join(parts), flush=True)

    def event(self, msg: str, kind: str = "event", **data):
        label, _ = self._state()
        self._emit("event", msg=msg, event_kind=kind, **data)
        self._say(f"[{label}] ** {msg}")


LIVE = LiveReporter()
