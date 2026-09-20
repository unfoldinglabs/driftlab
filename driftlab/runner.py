"""Grid runner: cells x agents -> one JSONL trajectory per run, plus a manifest
with cost. The runner knows nothing about how agents work; it gets an agent
from `agent_factory(cell, scenario, ctx)` and drives it through the scenario.
Harness runs (external agents via MCP) use `write_run` directly."""

import asyncio
import json
import time
from pathlib import Path

from .agents.brain import LEDGER, CostLedger
from .core import run
from .live import LIVE, current_run
from .paths import portable_path


def expand(env_cells: list[dict], agents: list[dict]) -> list[dict]:
    """Cross environment-side cells (regime, seed, ...) with agent specs."""
    return [{**c, "agent": a} for c in env_cells for a in agents]


def env_cells(regimes, seeds: int, **extra) -> list[dict]:
    return [{"regime": r, "seed": s, **extra} for r in regimes for s in range(seeds)]


def default_run_id(cell: dict) -> str:
    return f"{cell['agent']['name']}__{cell.get('regime', 'na')}__seed{cell['seed']}"


def run_complete(path: Path, expected_steps: int | None = None) -> bool:
    """True when `path` holds a fully written run of the CURRENT grid's length. write_run
    only writes after an episode finishes, so a parseable header and last line mean the
    episode completed — but a shorter mode's finished episode (a smoke run before a quick
    grid, a quick run before a full one) is not this grid's episode: when expected_steps
    is given, the file must hold at least that many steps or it is re-run and overwritten."""
    try:
        lines = path.read_text().splitlines()
        whole = (len(lines) >= 2 and json.loads(lines[0]).get("kind") == "header"
                 and bool(json.loads(lines[-1])))
        return whole and (expected_steps is None or len(lines) - 1 >= expected_steps)
    except (OSError, json.JSONDecodeError):
        return False


def write_run(out_dir: Path, run_id: str, cell: dict, header_extra: dict, records: list[dict],
              cost: dict, wall_s: float, config: dict | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id}.jsonl"
    tmp = path.with_name(path.name + ".tmp")  # write-then-replace: a kill mid-write never truncates a log
    with tmp.open("w") as f:
        f.write(json.dumps({"kind": "header", "cell": cell, "ts": time.time(), "wall_s": wall_s,
                            "cost": cost, **header_extra}) + "\n")
        for r in records:
            f.write(json.dumps({"kind": "step", **r}) + "\n")
    tmp.replace(path)
    entry = {"run": run_id, "path": portable_path(path), "cost": cost, "wall_s": wall_s}
    manifest = out_dir / "manifest.json"
    prior = json.loads(manifest.read_text()) if manifest.exists() else {"runs": [], "grid_costs": []}
    prior["runs"] = [r for r in prior["runs"] if r["run"] != run_id] + [entry]  # a rerun replaces its entry
    if config is not None:
        prior["config"] = config
    prior["unpriced_models"] = sorted(LEDGER.unpriced_models)
    mtmp = manifest.with_name("manifest.json.tmp")
    mtmp.write_text(json.dumps(prior, indent=2))
    mtmp.replace(manifest)
    return entry


async def run_cells(cells: list[dict], scenario_factory, agent_factory, out_dir: str, concurrency: int = 6,
                    run_id=default_run_id, config: dict | None = None, resume: bool = False) -> list[dict]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ctx = {"out_dir": out, "cache_dir": out.parent / "llm_cache", "agent_factory": agent_factory}
    sem = asyncio.Semaphore(concurrency)

    async def guarded(cell):
        async with sem:
            rid = run_id(cell)
            current_run.set(rid)
            try:
                scenario = scenario_factory(cell, ctx)  # cheap: worlds init without any LLM call
                if resume and run_complete(out / f"{rid}.jsonl", scenario.steps):
                    print(f"resume: {rid} already complete, skipped")
                    return {"run": rid, "path": portable_path(out / f"{rid}.jsonl"), "skipped": True}
                agent = agent_factory(cell, scenario, ctx)
                # out is runs/<exp>/<world>; the parent is the experiment (out.name is the world dir)
                exp = out.parent.name if out.parent.name.startswith("exp") else out.name
                LIVE.run_start(cell, scenario.steps, scenario.instructions, experiment=exp,
                               world=getattr(scenario.world, "name", None))
                LEDGER.start_run(rid)
                t0 = time.time()
                header_extra, records = await run(scenario, agent)
            except SystemExit as e:
                # one episode's hard failure (a capability gate, a harness error envelope that
                # survived its retry) must not discard its siblings' finished work: record it,
                # keep going, and let --resume replay it. If EVERY episode failed the same way,
                # the grid-level SystemExit below preserves the old fail-loudly behavior.
                print(f"episode {rid} failed: {e}")
                return {"run": rid, "error": str(e)}
            versions = getattr(agent, "memory_versions", None)
            if versions:  # persisted so a replayed run can show what the agent wrote itself
                header_extra["memory_versions"] = versions
            # the episode's own calls only: attributed by run id, so concurrent episodes
            # never absorb each other's spend (the old wall-clock ledger diff did)
            cost = LEDGER.take_run(rid)
            spent = sum(d["cost_usd"] for d in cost.values())
            rewards = [r["reward"] for r in records]
            LIVE.run_end({"steps": len(records), "mean_reward": round(sum(rewards) / len(rewards), 3),
                          "wall_s": round(time.time() - t0, 1), "cost_usd": round(spent, 4),
                          "parse_failures": header_extra.get("parse_failures", 0)})
            return write_run(out, rid, cell, header_extra, records, cost, time.time() - t0, config)

    start = LEDGER.snapshot()
    results = await asyncio.gather(*(guarded(c) for c in cells))
    errors = [r for r in results if isinstance(r, dict) and r.get("error")]
    if errors and len(errors) == len(results):
        raise SystemExit(errors[0]["error"])  # nothing survived: surface it like before
    if errors:
        print(f"{len(errors)} of {len(results)} episodes failed and were kept out of the logs; "
              "re-run with --resume to replay them")
    grid = CostLedger.diff(LEDGER.snapshot(), start)
    spent = sum(d["cost_usd"] for d in grid.values())
    saved = sum(d["cache_saved_usd"] for d in grid.values())
    print(f"grid cost: ${spent:.4f} spent, ${saved:.4f} served from cache")
    from .agents.brain import aclose_clients
    await aclose_clients()   # otherwise their transports GC after the loop closes, noisily
    return results
