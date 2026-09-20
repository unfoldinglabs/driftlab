"""driftlab dashboard: watch agents work, live or replayed, in a browser.

    python -m driftlab.viz.server              # http://localhost:8765
    python -m driftlab.viz.server --port 9000 --events runs/events.jsonl

The page tails the events file that every run appends to (in-process runs,
harness runs through the MCP server, all of them) and renders: the current
observation, the agent's reply and the outcome; world changes, notices and
memory rewrites as they happen; rolling and cumulative accuracy with event
markers; a per-request-key grid for RuleWorld; running metrics and the final
analysis table. Finished runs can be replayed from their JSONL logs.

No dependencies beyond the standard library.
"""

import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EVENTS = Path(os.environ.get("DRIFTLAB_EVENTS", ROOT / "runs" / "events.jsonl"))


def _clean_json(path: Path) -> str:
    """Python's json writes bare NaN, which browsers' JSON.parse rejects; serve null instead."""
    return json.dumps(json.loads(path.read_text(), parse_constant=lambda c: None))


_RUN_CACHE: dict = {}  # str(path) -> ((mtime_ns, size), entry) — /runs is polled, logs are immutable once written


def _run_entry(p: Path) -> dict:
    """One /runs row: who played (the header's agent spec), where it stands (steps,
    mean reward) and whether the log is a complete episode (same test as resume uses:
    parseable header + parseable last line)."""
    st = p.stat()
    key = (st.st_mtime_ns, st.st_size)
    hit = _RUN_CACHE.get(str(p))
    if hit and hit[0] == key:
        return hit[1]
    entry = {"path": str(p.relative_to(ROOT)), "label": p.stem}
    try:
        lines = p.read_text().splitlines()
        header = json.loads(lines[0]) if lines else {}
        if header.get("kind") == "header":
            cell = header.get("cell") or {}
            rewards = []
            last_ok = False
            for ln in lines[1:]:
                if not ln.strip():
                    continue
                try:
                    rec = json.loads(ln)
                    last_ok = True
                except json.JSONDecodeError:
                    last_ok = False
                    continue
                if "reward" in rec:
                    rewards.append(rec["reward"])
            entry.update(agent=cell.get("agent") or {}, regime=cell.get("regime"), seed=cell.get("seed"),
                         world=cell.get("world"), steps=len(rewards),
                         mean_reward=(sum(rewards) / len(rewards)) if rewards else None,
                         cost_usd=sum((c or {}).get("cost_usd", 0.0) for c in (header.get("cost") or {}).values()),
                         complete=bool(len(lines) >= 2 and last_ok))
    except (OSError, json.JSONDecodeError):
        pass
    _RUN_CACHE[str(p)] = (key, entry)
    return entry


def _benchmarks() -> dict:
    """The agent x experiment score matrix per world, computed from every profile.json
    under runs/exp*/ — any agent with analyzed runs appears, not just agents launched
    through the benchmark suite runner."""
    suite_by_world: dict = {}  # a suite column belongs on its canonical world's tab only
    try:
        sys.path.insert(0, str(ROOT))
        from experiments.benchmark import SUITE
        for entry in SUITE:
            exp, world = entry if isinstance(entry, (list, tuple)) else (entry, None)
            suite_by_world.setdefault(world, set()).add(exp.split("_")[0])
    except Exception:  # noqa: BLE001  (the matrix still renders from whatever profiles exist)
        pass
    dims = ("adaptation", "knowledge", "epistemics", "efficiency")
    num = lambda v: isinstance(v, (int, float))  # noqa: E731  (NaN was parsed to None)
    out: dict = {}
    for prof_path in sorted((ROOT / "runs").glob("exp*/*/profile.json")):
        exp, world = prof_path.parent.parent.name, prof_path.parent.name
        try:
            prof = json.loads(prof_path.read_text(), parse_constant=lambda c: None)
        except (OSError, ValueError):
            continue
        w = out.setdefault(world, {"suite": set(), "agents": {}, "generated": 0, "stale": False})
        w["suite"].add(exp)
        w["generated"] = max(w["generated"], prof.get("generated") or 0)
        newest_log = max((f.stat().st_mtime for f in prof_path.parent.glob("*.jsonl")), default=0)
        if newest_log > prof_path.stat().st_mtime + 1:
            w["stale"] = True  # runs landed after this profile was computed
        # a profile that had to fall back to smoke runs (no full-length data) is a sense
        # check, not a measurement: carry the flag so the matrix can say so
        runs_full = [r for r in prof.get("runs", []) if (r.get("n_steps") or 0) >= 30]
        quick_only = bool(prof.get("runs")) and not runs_full
        all_regimes = {str(r.get("regime")) for r in runs_full}
        for name, a in (prof.get("agents") or {}).items():
            row = w["agents"].setdefault(name, {"experiments": {}, "spec": {}})
            mine = {str(r.get("regime")) for r in runs_full if r.get("agent") == name}
            row["experiments"][exp] = {"driftlab_score": a.get("driftlab_score"),
                                       "dimensions": a.get("dimensions") or {}, "quick": quick_only,
                                       # a hypothesis run plays only the regimes its predictions need;
                                       # the score then covers fewer conditions than the full grid
                                       "partial": bool(mine) and mine < all_regimes}
            if not row["spec"] and a.get("spec"):
                row["spec"] = a["spec"]
    for world, w in out.items():
        w["suite"] = sorted(suite_by_world.get(world, set()) | w["suite"])
        for row in w["agents"].values():
            full = [e["driftlab_score"] for e in row["experiments"].values() if num(e["driftlab_score"]) and not e["quick"]]
            scores = full or [e["driftlab_score"] for e in row["experiments"].values() if num(e["driftlab_score"])]
            row["mean_score"] = sum(scores) / len(scores) if scores else None
            src = [e for e in row["experiments"].values() if not e["quick"]] or list(row["experiments"].values())
            row["mean_dimensions"] = {
                d: (lambda vs: sum(vs) / len(vs) if vs else None)(
                    [e["dimensions"].get(d) for e in src if num(e["dimensions"].get(d))])
                for d in dims}
            row["quick_only"] = all(e["quick"] for e in row["experiments"].values())
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            return self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        if u.path == "/events":
            return self._sse()
        if u.path == "/registry":
            try:
                sys.path.insert(0, str(ROOT))
                from experiments.registry import REGISTRY
                return self._send(200, json.dumps(REGISTRY))
            except Exception:  # noqa: BLE001  (the page renders fine without it)
                return self._send(200, "{}")
        if u.path == "/profile":
            rel = parse_qs(u.query).get("dir", [""])[0]
            path = (ROOT / rel / "profile.json").resolve()
            if not str(path).startswith(str(ROOT / "runs")) or not path.exists():
                return self._send(200, "null")
            return self._send(200, _clean_json(path))
        if u.path == "/benchmarks":
            return self._send(200, json.dumps(_benchmarks()))
        if u.path == "/worlds":
            try:
                sys.path.insert(0, str(ROOT))
                from driftlab.worlds.base import capabilities
                from driftlab.worlds.registry import WORLDS, make_world
                out = []
                for name in WORLDS:
                    w = make_world(name, seed=0, T=20)
                    out.append({"id": name, "system_prompt": getattr(w, "system_prompt", ""),
                                "capabilities": sorted(capabilities(w))})
                return self._send(200, json.dumps(out))
            except Exception:  # noqa: BLE001  (e.g. numpy missing in this interpreter)
                snap = HERE / "worlds.json"  # written by driftlab.doctor, which builds every world
                return self._send(200, snap.read_text() if snap.exists() else "[]")
        if u.path == "/agents":
            p = ROOT / "agents.json"
            try:
                return self._send(200, _clean_json(p) if p.exists() else "[]")
            except ValueError:
                return self._send(200, "[]")
        if u.path == "/validation":
            p = ROOT / "runs" / "validation.json"
            if not p.exists():
                return self._send(200, "null")
            data = json.loads(p.read_text(), parse_constant=lambda c: None)
            newest = max((f.stat().st_mtime for f in (ROOT / "runs").glob("exp*/*/*.jsonl")), default=0)
            data["stale"] = bool(newest > (data.get("generated") or 0) + 1)  # runs landed after the last validation
            return self._send(200, json.dumps(data))
        if u.path == "/hypotheses":
            items = []
            for p in sorted((ROOT / "research" / "hypotheses").glob("H*.md")):
                lines = p.read_text().splitlines()
                title = lines[0].lstrip("# ").split("—", 1)[-1].strip() if lines else p.stem
                status = next((ln.split(":", 1)[1].strip() for ln in lines if ln.lower().startswith("status:")), "")
                claim, in_claim = [], False  # the "in plain words" section, shown as the summary
                for ln in lines:
                    if ln.startswith("## "):
                        if in_claim:
                            break
                        in_claim = "plain words" in ln.lower()
                    elif in_claim and ln.strip():
                        claim.append(ln.strip())
                items.append({"id": p.name.split("-")[0], "file": p.name, "title": title,
                              "status": status, "claim": " ".join(claim), "body": p.read_text()})
            return self._send(200, json.dumps(items))
        if u.path == "/runs":
            logs = sorted(p for p in ROOT.joinpath("runs").rglob("*.jsonl")
                          if p.name != "events.jsonl" and p.parent.name != "llm_cache")
            return self._send(200, json.dumps([_run_entry(p) for p in logs]))
        if u.path == "/log":
            rel = parse_qs(u.query).get("path", [""])[0]
            path = (ROOT / rel).resolve()
            if not str(path).startswith(str(ROOT / "runs")) or not path.exists():
                return self._send(404, json.dumps({"error": "not found"}))
            lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
            return self._send(200, json.dumps({"header": lines[0], "steps": lines[1:]}, default=str))
        self._send(404, json.dumps({"error": "not found"}))

    def _sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        pos, last_beat = 0, time.time()
        try:
            while True:
                if EVENTS.exists():
                    with EVENTS.open("rb") as f:
                        f.seek(pos)
                        chunk = f.read()
                        pos = f.tell()
                    if chunk:
                        for line in chunk.splitlines():
                            if line.strip():
                                self.wfile.write(b"data: " + line + b"\n\n")
                        self.wfile.flush()
                if time.time() - last_beat > 15:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    last_beat = time.time()
                time.sleep(0.25)
        except (BrokenPipeError, ConnectionResetError):
            return


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--events", default=None, help="events file to tail (default runs/events.jsonl)")
    ap.add_argument("--open", action="store_true", help="open the browser")
    args = ap.parse_args()
    global EVENTS
    if args.events:
        EVENTS = Path(args.events).resolve()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"driftlab dashboard at {url}  (tailing {EVENTS})", flush=True)
    if args.open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
