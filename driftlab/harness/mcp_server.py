"""driftlab MCP server: lets an external agent harness be the agent.

The harness (Claude Code, Codex, Cursor, a script) connects over stdio, picks
an experiment and a cell, and then alternates `act` calls: each call returns
the outcome of the previous action and the next observation. The scenario,
drift, privileged logging, and metrics are identical to in-process runs; only
the agent is outside. How the harness remembers, reflects, or takes notes is
entirely its own business.

    python3 driftlab/harness/mcp_server.py                      # stdio (use the absolute path when registering)
    claude mcp add driftlab -- python3 /abs/path/driftlab/driftlab/harness/mcp_server.py
    codex mcp add driftlab -- python3 /abs/path/driftlab/driftlab/harness/mcp_server.py

Tools: run_experiment (the one to use), act, status, abort; list_experiments, describe_experiment, start_run.
Logs go to runs/<experiment>/ like any other run; analyze with
`python -m experiments.<experiment> --analyze`.
"""

import asyncio
import hashlib
import importlib
import json
import os
import pkgutil
import sys
import time
from pathlib import Path

from mcp.server.mcpserver import MCPServer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from driftlab.core import run  # noqa: E402
from driftlab.live import LIVE, current_run  # noqa: E402
from driftlab.harness.channel import ChannelAgent  # noqa: E402
from driftlab.runner import run_complete, write_run  # noqa: E402

server = MCPServer("driftlab", instructions=(
    "driftlab runs an agent through a job at a fictional company and measures how it copes. To run an experiment end to end, "
    "call run_experiment(name) once, then keep calling act(text) with your reply to each observation until "
    "the result says finished. Every result tells you the outcome of your previous action, the next "
    "observation, and the exact reply format required. The server walks through all of the experiment's "
    "environment configurations for you and returns the computed metrics at the end. How you remember and "
    "learn between steps is entirely up to you; that is what is being measured. list_experiments shows what "
    "is available; start_run/describe_experiment exist for single-configuration control."))

STATE: dict = {}
FORMAT_HINT = {
    "rule_world": "End your reply with a line exactly like: CHOICE: <A|B|C|D>",
    "form_filler": "End your reply with a line exactly like: FORM: {\"field\": \"value\", ...}  (a single JSON object)",
    "inventory": "End your reply with a line exactly like: ORDER: <integer>",
    "codebase": "Reply with the complete function source inside one ```python code block",
}


def _hint(scenario) -> str:
    return FORMAT_HINT.get(getattr(scenario.world, "name", ""), "Follow the reply format given in the instructions.")


TRANSITION = {"first": "Reply to each message with act(text).",
              "next": ("You have been reassigned to a different site. Same job, new colleagues, and their way of "
                       "doing things may differ from where you were before.")}


def _experiments_quick(quick: bool):
    """Toggle quick mode and reload experiment modules so their schedules pick it up."""
    from experiments import common
    if common.QUICK != quick:
        common.QUICK = quick
        for mod in _experiments().values():
            importlib.reload(mod)
    return _experiments()


NOTEBOOK_HINT = ("\n\nYou have a personal notebook: call write_notebook(text) whenever you want to save what you've "
                 "worked out (procedures, things that went wrong, hunches) — it overwrites what was there, so include "
                 "everything you still want to keep. Call read_notebook() to see what you saved. It persists between "
                 "requests and follows you if you are reassigned; nobody else reads it. It starts empty, and driftlab "
                 "will not remind you again — decide for yourself when it's worth writing something down.")


def _notebook(agent_label: str) -> Path:
    p = ROOT / "runs" / "notebooks" / agent_label / "notebook.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text("")
    return p


def _bind():
    """MCP tool calls run in their own task, so the run label contextvar set in _launch is not
    visible there. Re-bind it from STATE before anything logs an event, or memory/analysis events
    get attributed to a run called "run" and never show up on the dashboard."""
    if STATE.get("rid"):
        current_run.set(STATE["rid"])


def _snapshot_notebook(t: int, skip_empty: bool = False):
    """If the harness edited its notebook since the last step, log a versioned memory event."""
    nb = STATE.get("notebook")
    if not nb:
        return
    _bind()
    text = nb.read_text() if nb.exists() else ""
    if not text and (skip_empty or not STATE.get("memory_versions")):
        return   # an empty notebook that was never written is "nothing yet", not a version; a later clear is
    digest = hashlib.sha1(text.encode()).hexdigest()
    if digest != STATE.get("notebook_digest"):
        STATE["notebook_digest"] = digest
        STATE.setdefault("memory_versions", []).append({"episode": STATE.get("idx", 0) + 1, "t": t, "chars": len(text), "text": text})
        LIVE.event(f"notebook updated ({len(text)} chars)", kind="memory", memory=text, memory_kind="notebook",
                   version=len(STATE["memory_versions"]))


def _launch(m, cell: dict, agent_label: str, experiment: str):
    """Start one episode in the background; return (scenario, agent, task, out)."""
    out = ROOT / "runs" / experiment.split("_")[0] / cell.get("world", "rule_world")
    ctx = {"out_dir": out, "cache_dir": ROOT / "runs" / "llm_cache", "agent_factory": None}
    scenario = m.scenario(cell, ctx)
    notebook = _notebook(agent_label)
    scenario.system_prompt = scenario.instructions + NOTEBOOK_HINT.format(path=notebook)
    agent = ChannelAgent()
    t0 = time.time()
    rid = f"{agent_label}__{cell.get('regime', 'na')}__seed{cell['seed']}"
    STATE["rid"] = rid
    current_run.set(rid)
    STATE["notebook"] = notebook
    STATE.setdefault("memory_versions", [])   # versions accumulate across episodes; each header carries all so far
    STATE["notebook_digest"] = None           # so the notebook carried into this episode is re-shown for its run
    LIVE.run_start(cell, scenario.steps, scenario.instructions, experiment=out.parent.name, world=getattr(scenario.world, "name", None))
    _snapshot_notebook(0, skip_empty=True)

    async def go():
        header, records = await run(scenario, agent)
        header = {**header, "memory_versions": STATE.get("memory_versions", []), "notebook": str(notebook)}
        entry = write_run(out, rid, cell, header, records, cost={}, wall_s=time.time() - t0)
        result = {"run": entry["run"], "log": entry["path"], "steps": len(records),
                  "mean_reward": round(sum(r["reward"] for r in records) / max(len(records), 1), 3),
                  "parse_failures": header.get("parse_failures", 0)}
        LIVE.run_end({**{k: result[k] for k in ("steps", "mean_reward", "parse_failures")}, "wall_s": round(time.time() - t0, 1)})
        return result

    return scenario, agent, asyncio.create_task(go()), out


def _experiments():
    import experiments
    mods = {}
    for m in pkgutil.iter_modules(experiments.__path__):
        if m.name.startswith("exp"):
            mods[m.name] = importlib.import_module(f"experiments.{m.name}")
    return mods


@server.tool()
def list_experiments() -> list[dict]:
    """List available experiments with a one-line summary."""
    return [{"name": n, "summary": (m.__doc__ or "").strip().splitlines()[0]} for n, m in _experiments().items()]


@server.tool()
def describe_experiment(name: str, seeds: int = 3) -> dict:
    """Full description of an experiment and its environment cells (regime x seed)."""
    m = _experiments()[name]
    cells = m.cells(seeds)
    # for the experimenter, not the agent: the doc states the hypothesis and cells name regimes
    return {"doc": m.__doc__, "cells": [{"index": i, **{k: v for k, v in c.items() if k != "agent"}} for i, c in enumerate(cells)]}


def _resolve(name: str) -> str:
    """Accept 'exp04', 'exp04_surface_vs_latent', '4', or 'experiment 4'."""
    mods = _experiments()
    if name in mods:
        return name
    digits = "".join(ch for ch in name if ch.isdigit())
    for n in mods:
        if digits and n.startswith(f"exp{int(digits):02d}"):
            return n
    raise KeyError(f"unknown experiment {name!r}; available: {sorted(mods)}")


# Agent identity recorded in every run header. The launcher (path A: driftlab.harness.launch)
# enforces a profile through the environment; a harness may also self-declare one via
# run_experiment(agent_profile=...). Enforced keys win over self-declared ones.
ENFORCED_PROFILE = json.loads(os.environ.get("DRIFTLAB_AGENT_PROFILE") or "null")


async def _begin_episode(idx: int) -> dict:
    m = _experiments()[STATE["experiment"]]
    profile = {**(STATE.get("agent_profile") or {}), **(ENFORCED_PROFILE or {})}
    if ENFORCED_PROFILE:
        profile["profile_enforced"] = True
    cell = {**STATE["cells"][idx], "agent": {"name": STATE["agent_label"], "type": "harness", **profile}}
    scenario, agent, task, out = _launch(m, cell, STATE["agent_label"], STATE["experiment"])
    STATE.update(task=task, agent=agent, cell=cell, t=0, T=scenario.steps, idx=idx, scenario=scenario, out=out)
    first = await agent.to_harness.get()
    total = len(STATE["cells"])
    return {"episode": f"{idx + 1} of {total}", "steps": scenario.steps, "instructions": scenario.instructions,
            "reply_format": _hint(scenario), "observation": first["prompt"], "t": 1,
            "note": TRANSITION["next" if idx else "first"]}


@server.tool()
async def run_experiment(experiment: str, agent_label: str = "harness", seeds: int = 1,
                         independent_worlds: bool = True, quick: bool = False, world: str = "rule_world",
                         agent_profile: dict | None = None, resume: bool = False) -> dict:
    """Run a whole experiment: every environment configuration in turn, then the metrics.
    Call once, then keep calling act(text) until the result says finished. `experiment` may be
    'exp04', 'exp04_surface_vs_latent' or just '4'. `seeds` = independent repeats per configuration.
    `independent_worlds` (default true) gives every episode different hidden rules, so knowledge carried
    in your context from one episode cannot be reused in the next; set false to get the exact same worlds
    an in-process agent would see (paired comparison), accepting that carry-over then inflates results.
    `quick` runs tenth-length episodes with scaled schedules: a sense-check, not a measurement.
    `world`: rule_world | form_filler | inventory | codebase (experiments refuse worlds lacking what they need).
    `agent_profile`: optional facts about yourself (model, version, memory strategy, ...) recorded
    verbatim in every run header so results stay attributable.
    `resume`: skip episodes whose run log is already complete for this agent_label — episodes are
    seeded, so an experiment interrupted by a usage limit continues exactly where it stopped, on
    this machine or another one sharing the runs directory.
    You get a personal notebook file; whatever you write there is versioned and shown on the dashboard."""
    if STATE.get("task") and not STATE["task"].done():
        return {"error": "a run is in progress; call abort() or finish it first"}
    name = _resolve(experiment)
    m = _experiments_quick(quick)[name]
    cells = [{"world": world, **c} for c in m.cells(seeds)]
    if independent_worlds:
        cells = [{**c, "seed": c["seed"] + 1000 * (i + 1)} for i, c in enumerate(cells)]
    skipped = 0
    if resume:
        exp_dir = ROOT / "runs" / name.split("_")[0] / world
        remaining = [c for c in cells
                     if not run_complete(exp_dir / f"{agent_label}__{c.get('regime', 'na')}__seed{c['seed']}.jsonl")]
        skipped = len(cells) - len(remaining)
        cells = remaining
        if not cells:
            STATE.clear()
            STATE.update(experiment=name, out=exp_dir)
            analysis = _analysis_text()
            STATE.clear()
            return {"finished": True, "experiment": name, "episodes_skipped": skipped,
                    "note": "every episode for this agent_label is already complete; nothing to play",
                    "metrics": analysis}
    STATE.clear()
    STATE.update(experiment=name, agent_label=agent_label, cells=cells, auto=True, quick=quick,
                 agent_profile=agent_profile, memory_versions=[], notebook_digest=None)
    return {"experiment": name, "episodes": len(STATE["cells"]),
            **({"episodes_skipped": skipped} if skipped else {}), "quick": quick, **(await _begin_episode(0))}


@server.tool()
async def start_run(experiment: str, cell_index: int = 0, agent_label: str = "harness", seeds: int = 3,
                    world: str = "rule_world") -> dict:
    """Advanced: run a single environment configuration (cell). Prefer run_experiment."""
    if STATE.get("task") and not STATE["task"].done():
        return {"error": "a run is in progress; call abort() or finish it first"}
    name = _resolve(experiment)
    m = _experiments()[name]
    STATE.clear()
    STATE.update(experiment=name, agent_label=agent_label, cells=[{"world": world, **m.cells(seeds)[cell_index]}], auto=False)
    return await _begin_episode(0)


def _analysis_text() -> str:
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            _experiments()[STATE["experiment"]].analyze(str(STATE["out"]))
            print()
            from driftlab.profile import report
            report(str(STATE["out"]))
        except Exception as e:  # noqa: BLE001
            print(f"(analysis failed: {e!r})")
    return buf.getvalue()


@server.tool()
async def act(text: str) -> dict:
    """Send your reply to the current observation. Returns the outcome of that action, the next observation, and done."""
    if not STATE.get("task"):
        return {"error": "no run in progress; call start_run first"}
    agent, task = STATE["agent"], STATE["task"]
    _bind()
    _snapshot_notebook(STATE["t"] + 1)
    await agent.from_harness.put(text)
    getter = asyncio.ensure_future(agent.to_harness.get())
    done, _ = await asyncio.wait({getter, task}, return_when=asyncio.FIRST_COMPLETED)
    if task in done and not getter.done():
        getter.cancel()
        result = task.result()
        finished = {"last_feedback": agent.pending_feedback, "episode_result": result}
        STATE.setdefault("results", []).append(result)
        nxt = STATE["idx"] + 1
        if STATE.get("auto") and nxt < len(STATE["cells"]):
            _snapshot_notebook(STATE["T"])
            return {"finished": False, "episode_finished": finished, **(await _begin_episode(nxt))}
        analysis = _analysis_text() if STATE.get("auto") else None
        if analysis:
            _bind()
            LIVE.analysis(STATE["experiment"], analysis)
        _snapshot_notebook(STATE["T"])
        summary = {"finished": True, "experiment": STATE["experiment"], **finished,
                   "notebook": str(STATE.get("notebook")), "notebook_versions": len(STATE.get("memory_versions", [])),
                   "episodes": STATE["results"], "logs_dir": str(STATE["out"]), "metrics": analysis,
                   "next": f"to compare with other agents later: python -m experiments.{STATE['experiment']} --analyze"}
        STATE.clear()
        return summary
    msg = getter.result()
    STATE["t"] += 1
    out = {"finished": False, "t": STATE["t"] + 1, "steps": STATE["T"], "feedback": msg["feedback"],
           "observation": msg["prompt"], "reply_format": _hint(STATE["scenario"])}
    if msg["feedback"] is None and STATE["t"] > 0:
        out["warning"] = "No feedback for the previous step (it was an evaluation probe)."
    return out


@server.tool()
def write_notebook(text: str) -> dict:
    """Save your notebook, replacing whatever was there before. Call this whenever you want to
    keep something for later — a procedure you worked out, a mistake to avoid, a hunch. Include
    everything you still want to keep; this overwrites rather than appends."""
    nb = STATE.get("notebook")
    if not nb:
        return {"error": "no run in progress"}
    nb.write_text(text)
    _snapshot_notebook(STATE.get("t", 0))
    return {"saved": True, "chars": len(text)}


@server.tool()
def read_notebook() -> dict:
    """Read back what you last saved with write_notebook."""
    nb = STATE.get("notebook")
    if not nb:
        return {"error": "no run in progress"}
    return {"text": nb.read_text() if nb.exists() else "", "note": "empty — nothing saved yet" if not (nb.exists() and nb.read_text()) else None}


@server.tool()
def status() -> dict:
    """Which experiment/cell is running and how far along it is."""
    if not STATE.get("task"):
        return {"running": False}
    return {"running": not STATE["task"].done(), "experiment": STATE["experiment"],
            "episode": f"{STATE['idx'] + 1} of {len(STATE['cells'])}",
            "configuration": {k: v for k, v in STATE["cell"].items() if k != "agent"}, "t": STATE["t"], "steps": STATE["T"]}


@server.tool()
def abort() -> dict:
    """Abort the current run without writing a log."""
    if STATE.get("task"):
        STATE["task"].cancel()
    STATE.clear()
    return {"aborted": True}


if __name__ == "__main__":
    server.run("stdio")
