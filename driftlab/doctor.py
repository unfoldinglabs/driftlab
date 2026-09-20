"""The doctor: checks that every entity fits its contract and its connections.

    python -m driftlab.doctor          # exit 0 when healthy, 1 when something is broken

Run it after adding or changing a world, experiment, hypothesis, agent spec or
the benchmark suite (CONTRIBUTING.md has the per-entity checklists). It verifies:

    worlds        build from the registry, survive one observe/parse/act/privileged
                  cycle, and report the privileged keys the metrics rely on
    experiments   every experiments/expNN module exposes the four-part contract
                  (cells, scenario, REFERENCE_AGENTS, analyze) and has a registry
                  entry with the required fields — and vice versa
    predictions   are well-formed, name profile metrics that exist, and reference
                  regimes the experiment's cells actually produce
    hypotheses    every id an experiment bears on has a file with a Status line,
                  and every hypothesis file is carried by at least one experiment
    benchmark     the reference suite names real, registered experiments
    agents        agents.json (when present) parses and uses known spec types
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROBLEMS: list[str] = []
WARNINGS: list[str] = []


def problem(msg: str):
    PROBLEMS.append(msg)
    print(f"  FAIL  {msg}")


def warn(msg: str):
    WARNINGS.append(msg)
    print(f"  warn  {msg}")


def ok(msg: str):
    print(f"  ok    {msg}")


def check_worlds():
    print("\nworlds")
    from driftlab.worlds.base import capabilities
    from driftlab.worlds.registry import WORLDS, make_world
    rng = np.random.default_rng(0)
    snapshot = []
    for name in WORLDS:
        try:
            w = make_world(name, seed=0, T=20)
            for attr in ("name", "system_prompt", "T"):
                assert getattr(w, attr, None) is not None, f"missing attribute {attr}"
            obs = w.observe(0)
            assert isinstance(obs, str) and obs, "observe(0) must return text"
            action = w.default_action(rng)
            reward, fb = w.act(0, action)
            float(reward)
            assert isinstance(fb, str), "act must return (reward, feedback text)"
            priv = w.privileged(0)
            assert isinstance(priv.get("changes"), list), "privileged() must include changes: []"
            if "task_id" in priv:
                assert "task_done" in priv, "a world reporting task_id must also report task_done"
            assert isinstance(w.summary(), dict), "summary() must return a dict"
            caps = capabilities(w)
            snapshot.append({"id": name, "system_prompt": w.system_prompt, "capabilities": sorted(caps)})
            ok(f"{name}: protocol + one live step; capabilities: {', '.join(sorted(caps))}")
        except Exception as e:  # noqa: BLE001
            problem(f"world {name}: {e!r}")
    if snapshot and len(snapshot) == len(WORLDS):
        # the dashboard's fallback for /worlds when its interpreter can't build worlds live
        (ROOT / "driftlab" / "viz" / "worlds.json").write_text(json.dumps(snapshot, indent=1))
        ok("driftlab/viz/worlds.json refreshed (the dashboard's Worlds fallback)")


def experiment_modules() -> dict:
    import experiments
    import importlib
    import pkgutil
    mods = {}
    for m in pkgutil.iter_modules(experiments.__path__):
        if m.name.startswith("exp"):
            mods[m.name.split("_")[0]] = importlib.import_module(f"experiments.{m.name}")
    return mods


def check_experiments():
    print("\nexperiments + registry")
    from experiments.registry import REGISTRY
    mods = experiment_modules()
    required = ("version", "title", "hypothesis", "independent", "dependent", "drift", "worlds", "hypotheses")
    for eid, mod in sorted(mods.items()):
        missing = [a for a in ("cells", "scenario", "REFERENCE_AGENTS", "analyze") if not hasattr(mod, a)]
        if missing:
            problem(f"{eid}: module lacks {missing} (the four-part experiment contract)")
        if eid not in REGISTRY:
            problem(f"{eid}: no registry entry — every experiment declares hypothesis, variables and version")
            continue
        entry = REGISTRY[eid]
        gaps = [f for f in required if not entry.get(f)]
        if gaps:
            problem(f"{eid}: registry entry missing {gaps}")
        else:
            ok(f"{eid} v{entry['version']}: contract + registry entry")
    for eid in REGISTRY:
        if eid not in mods:
            problem(f"registry entry {eid} has no experiments/{eid}_*.py module")


def check_predictions():
    print("\npredictions")
    from driftlab.profile import METRICS
    from experiments.registry import REGISTRY
    fields = {f for f, _, _ in METRICS} | {"n_changes"}
    mods = experiment_modules()
    for eid, entry in REGISTRY.items():
        for p in entry.get("predictions", []):
            where = f"{eid} prediction {p.get('claim', '?')[:40]!r}"
            if p.get("kind") not in ("effect", "reversal", "sign"):
                problem(f"{where}: unknown kind {p.get('kind')!r}"); continue
            if p.get("field") not in fields:
                problem(f"{where}: field {p.get('field')!r} is not a profile metric {sorted(fields)}"); continue
            need = {"effect": ("vary", "direction"), "reversal": ("a_regime", "b_regime"),
                    "sign": ("direction",)}[p["kind"]]
            gaps = [k for k in need if k not in p]
            if gaps or "claim" not in p:
                problem(f"{where}: missing keys {gaps + (['claim'] if 'claim' not in p else [])}"); continue
            regs = None
            if eid in mods:
                try:
                    regs = {str(c.get("regime")) for c in mods[eid].cells(1)}
                except Exception as e:  # noqa: BLE001
                    warn(f"{eid}: cells(1) failed ({e!r}); cannot verify prediction regimes")
            named = [p[k] for k in ("a", "b", "a_regime", "b_regime") if k in p and p.get("vary", "regime") != "agent"]
            if regs is not None and p["kind"] == "reversal":
                named = [p["a_regime"], p["b_regime"]]
            if regs is not None and p["kind"] == "sign" and p.get("regime"):
                named = [p["regime"]]
            if regs is not None and (p.get("vary") == "regime" or p["kind"] == "reversal"
                                     or (p["kind"] == "sign" and p.get("regime"))):
                bad = [r for r in named if r not in regs]
                if bad:
                    problem(f"{where}: regimes {bad} not produced by cells(); actual: {sorted(regs)}"); continue
            ok(f"{where}: well-formed")


def check_hypotheses():
    print("\nhypotheses")
    from experiments.registry import HYPOTHESES_DIR, REGISTRY
    referenced = {h for e in REGISTRY.values() for h in e.get("hypotheses", [])}
    files = {p.name.split("-")[0]: p for p in sorted(HYPOTHESES_DIR.glob("H*.md"))} if HYPOTHESES_DIR.is_dir() else {}
    for h in sorted(referenced):
        if h not in files:
            problem(f"{h}: referenced by the registry but research/hypotheses/ has no {h}-*.md file")
        elif not any(ln.lower().startswith("status:") for ln in files[h].read_text().splitlines()):
            problem(f"{h}: {files[h].name} has no 'Status:' line")
        else:
            ok(f"{h}: file + status ({files[h].name})")
    for h, p in files.items():
        if h not in referenced:
            warn(f"{h} ({p.name}): no experiment bears on it — register it or it will never gather evidence")


def check_benchmark_and_agents():
    print("\nbenchmark suite + agent specs")
    from experiments.benchmark import SUITE
    from experiments.registry import REGISTRY
    from driftlab.worlds.registry import WORLDS
    mods = experiment_modules()
    for exp, world in SUITE:
        eid = exp.split("_")[0]
        if eid not in mods or eid not in REGISTRY:
            problem(f"benchmark suite names {exp}, which is not a registered experiment module")
        elif world not in WORLDS:
            problem(f"benchmark suite pins {eid} to {world!r}, which is not a registered world")
        else:
            ok(f"suite: {eid} on {world} ({REGISTRY[eid]['title']})")
    spec_path = ROOT / "agents.json"
    if spec_path.exists():
        known = {"reference", "tabular", "harness_cli", "custom", "harness"}
        try:
            specs = json.loads(spec_path.read_text())
            assert isinstance(specs, list) and specs, "agents.json must be a non-empty JSON list"
            for s in specs:
                if not s.get("name"):
                    problem("agents.json: every spec needs a name (it is the identity runs are grouped and resumed by)")
                elif s.get("type", "reference") not in known:
                    problem(f"agents.json: {s['name']}: unknown type {s.get('type')!r} (known: {sorted(known)})")
                elif s.get("type") == "custom" and not s.get("factory"):
                    problem(f"agents.json: {s['name']}: custom agents need factory: pkg.module:function")
                elif s.get("type") == "harness_cli" and not s.get("model"):
                    warn(f"agents.json: {s['name']}: no model pinned — runs will not record which model played")
                else:
                    ok(f"agents.json: {s['name']} ({s.get('type', 'reference')})")
        except (ValueError, AssertionError) as e:
            problem(f"agents.json: {e}")
    else:
        ok("agents.json: none present (optional)")


if __name__ == "__main__":
    check_worlds()
    check_experiments()
    check_predictions()
    check_hypotheses()
    check_benchmark_and_agents()
    print(f"\n{'HEALTHY' if not PROBLEMS else 'BROKEN'}: {len(PROBLEMS)} problem(s), {len(WARNINGS)} warning(s)")
    sys.exit(1 if PROBLEMS else 0)
