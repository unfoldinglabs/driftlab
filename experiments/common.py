"""Shared CLI for experiment entry points.

An experiment defines environment cells, a scenario factory, and an analysis.
The CLI decides which world hosts it and which agent plays:

    --world rule_world|form_filler|inventory|codebase   (default rule_world; an experiment refuses a
                                                         world that lacks a capability it needs)
    (default agent)               the experiment's REFERENCE_AGENTS (in-process LLM + memory plugins)
    --agent tabular               the RuleWorld memorization baseline
    --agent cli:claude[:model]    a harness CLI driven per step, pinned config (path B; also cli:codex)
    --agent custom:pkg.mod:fn     your own factory(spec, cell, scenario, ctx) -> Agent
    --agent-spec agents.json      a JSON list of agent specs
    (harness)                     run the MCP server and let an external harness play; --analyze reads its logs

Logs go to runs/<exp>/<world>/ so the same experiment on different worlds stays separate.
"""

import argparse
import asyncio
import json
import os as _os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_SEEDS = 3   # exploratory single-experiment runs
MIN_PAIRS = 3       # the verdict floor: below this, no CI is even attempted. At 3-4 pairs the
                    # bootstrap interval is wide, so only large effects turn conclusive — small-n
                    # verdicts are self-limiting, and the n is printed with every one
VERDICT_SEEDS = 5   # the confirmatory default for hypothesis runs and the benchmark

# Read at import time because experiments define schedules as module constants.
# Two reduced modes, with different contracts:
#   --smoke  a tenth of everything, 1 seed: a plumbing check. Under 30 steps, so these
#            runs are excluded from profiles and can never produce a verdict.
#   --quick  half of everything, and a smaller task space where the world supports it:
#            shorter but still valid. Runs stay above the 30-step floor and count.
SMOKE = ("--smoke" in sys.argv) or bool(_os.environ.get("DRIFTLAB_SMOKE"))
QUICK = (not SMOKE) and (("--quick" in sys.argv) or bool(_os.environ.get("DRIFTLAB_QUICK")))
QUICK_FACTOR = float(_os.environ.get("DRIFTLAB_QUICK_FACTOR", "0.1" if SMOKE else ("0.5" if QUICK else "1")))
MODE = "smoke" if SMOKE else ("quick" if QUICK else "full")
WORLD = _os.environ.get("DRIFTLAB_WORLD", "rule_world")
if "--world" in sys.argv:
    WORLD = sys.argv[sys.argv.index("--world") + 1]

# reduced-mode task spaces: denser encounters per task, so shorter episodes keep their
# measurement power (each key is revisited about as often as in a full-length episode)
COMPACT_WORLDS = {"rule_world": {"n_types": 4}, "claims_desk": {"n_types": 3}}


def q(n: int, minimum: int = 2) -> int:
    """Scale a step count for the reduced modes (identity in full mode)."""
    return max(minimum, round(n * QUICK_FACTOR)) if MODE != "full" else n


def qlist(xs: list) -> list:
    return [q(x) for x in xs]


# ---- worlds -------------------------------------------------------------------------

def world_for(cell: dict, T: int, needs: tuple = (), **kw):
    """Build the cell's world (cell['world'] or the CLI's --world) and check capabilities.
    Reduced modes shrink the task space where the world supports it (COMPACT_WORLDS)."""
    from driftlab.worlds.base import require
    from driftlab.worlds.registry import make_world
    name = cell.get("world", WORLD)
    if MODE != "full":
        kw = {**COMPACT_WORLDS.get(name, {}), **kw}
    w = make_world(name, cell["seed"], T=T, **kw)
    if needs:
        require(w, *needs)
    return w


def organization(**kw):
    """A before_step hook that drives caused, correlated drift: one OrganizationProcess
    per world (seeded from the world's own seed), stepping every step. Changes it emits
    carry a shared cause id. Keyword arguments go to OrganizationProcess (rate, coherence,
    surface_share, warmup)."""
    from driftlab.drift import OrganizationProcess
    procs: dict = {}

    def before(t, w):
        p = procs.get(id(w))
        if p is None:
            p = procs[id(w)] = OrganizationProcess(seed=getattr(w, "seed", 0), **kw)
        p.step(t, w)
    return before


def scheduled(latent=(), surface=(), novelty=()):
    """A before_step hook that applies world changes at the given steps."""
    latent, surface, novelty = set(latent), set(surface), set(novelty)

    def before(t, w):
        if t in latent:
            w.change_latent(t)
        if t in surface:
            w.change_surface(t)
        if t in novelty:
            w.introduce_novelty(t)
    return before


def spaced(n: int, T: int, warmup: int) -> list[int]:
    """n change steps evenly spaced between warmup and T-5 (what RuleWorld's own schedule did)."""
    import numpy as np
    if n <= 0:
        return []
    return [int(x) for x in np.linspace(warmup, max(T - 5, warmup + 1), n + 1)[:-1]]


def recurring_caseload(seed: int):
    """A dense task stream for worlds with a truth table (the intake desk): draw from a
    fixed pool of one case per request type instead of the full task space, so even a
    5-step memory meets the same case again (a full 36-key space starves short windows:
    a case repeats within 5 steps only ~13% of the time). Every base-rule flip touches
    at least one pool case. Deterministic in (seed, t); the pool follows novelty.

        world = world_for(cell, T, needs=("change_latent", "task_key"))
        if hasattr(world, "truth_table"):
            world.task_fn = recurring_caseload(cell["seed"])
    """
    import numpy as np

    def fn(t, world):
        pool, seen = [], set()
        for k in sorted(world.truth_table()):
            if k[0] not in seen:
                seen.add(k[0])
                pool.append(k)
        rng = np.random.default_rng(seed * 99_991 + t)
        return dict(zip(("type", "region", "size"), pool[rng.integers(len(pool))]))
    return fn


# ---- CLI ----------------------------------------------------------------------------

def cli(name: str, description: str, extra=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", default=WORLD,
                    choices=["rule_world", "form_filler", "inventory", "codebase", "claims_desk", "campaign_desk"])
    ap.add_argument("--analyze", action="store_true", help="analyze existing logs instead of running")
    ap.add_argument("--mock", action="store_true", help="reference agents use a credential-free mock brain")
    ap.add_argument("--smoke", action="store_true", help="plumbing check: everything scaled to a tenth, one seed; never counts")
    ap.add_argument("--quick", action="store_true", help="half-length but valid: episodes stay above the 30-step floor and count")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--model", default=None, help="override the reference agents' model id (default gpt-5-mini)")
    ap.add_argument("--effort", default=None, help="reasoning effort for reasoning models")
    ap.add_argument("--agent", default=None, help="tabular | cli:claude[:model] | cli:codex[:model] | custom:pkg.module:factory")
    ap.add_argument("--agent-spec", default=None, help="JSON file with a list of agent specs")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--out", default=None, help="log directory (default runs/<exp>/<world>)")
    ap.add_argument("--budget-usd", type=float, default=None, help="abort once cumulative LLM spend exceeds this")
    ap.add_argument("--resume", action="store_true",
                    help="skip cells whose run log is already complete; episodes are seeded, so a "
                         "grid interrupted anywhere (or continued from another machine sharing runs/) "
                         "picks up exactly where it left off")
    ap.add_argument("--live", action="store_true", help="stream steps, world events and running metrics")
    ap.add_argument("--live-every", type=int, default=10)
    if extra:
        extra(ap)
    args = ap.parse_args()
    if args.budget_usd is not None:
        from driftlab.agents.brain import LEDGER
        LEDGER.budget_usd = args.budget_usd
    if args.live:
        from driftlab.live import LIVE
        LIVE.enabled, LIVE.every = True, args.live_every
        if args.concurrency == ap.get_default("concurrency"):
            args.concurrency = 1
    return args


def resolve_agents(args, reference_agents: list[dict]) -> list[dict]:
    if args.agent_spec:
        specs = json.loads(Path(args.agent_spec).read_text())
    elif args.agent == "tabular":
        specs = [{"name": "tabular", "type": "tabular"}]
    elif args.agent and args.agent.startswith("cli:"):
        _, harness, *model = args.agent.split(":", 2)
        if not model:
            print(f"warning: no model pinned for cli:{harness} — the harness's account default will play "
                  f"and the run headers will not record which model it was. Prefer cli:{harness}:<model>.")
        specs = [{"name": f"{harness}_cli", "type": "harness_cli", "harness": harness,
                  **({"model": model[0]} if model else {})}]
    elif args.agent and args.agent.startswith("custom:"):
        specs = [{"name": "custom", "type": "custom", "factory": args.agent[len("custom:"):]}]
    else:
        specs = [dict(a) for a in reference_agents]
    for s in specs:
        s.setdefault("type", "reference")
        if s["type"] == "reference":
            s.setdefault("model", "gpt-5-mini")
            s.setdefault("reasoning_effort", "low")
            if args.mock:
                s["mock"] = True
            if args.model:
                s["model"] = args.model
                # run identity is (agent name, regime, seed): the same arms on a second
                # model must be new runs, not resume-hits on the first model's episodes
                bare = args.model.split("/")[-1]
                if bare not in s["name"]:
                    s["name"] = f"{s['name']}-{bare}"
            if args.effort:
                s["reasoning_effort"] = args.effort
    return specs


def refresh_validation():
    """Recompute runs/validation.json so verdicts never lag the data (runs after every
    experiment completes or is re-analyzed — never mid-collection, to avoid peeking)."""
    try:
        from experiments.validate import run_validation
        result = run_validation(None, MIN_PAIRS)
        out = ROOT / "runs" / "validation.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(result, indent=2, default=lambda o: None))
        print("validation refreshed -> runs/validation.json")
    except Exception as e:  # noqa: BLE001  (a broken validator should never block a run)
        print(f"(validation refresh skipped: {e})")


def out_dir(name: str, world: str, cells) -> str:
    """Logs live under runs/<exp>/<world>. Cells may pin their own world (claims_desk,
    campaign_desk experiments); when they all agree, that world names the directory."""
    try:
        worlds = {c.get("world", world) for c in cells(1)}
        if len(worlds) == 1:
            world = worlds.pop()
    except Exception:  # noqa: BLE001  (fall back to the CLI world)
        pass
    return _os.path.relpath(ROOT / "runs" / name / world)  # relative: keeps printed paths portable


def run_experiment(name, doc, cells, scenario, reference_agents, analyze, config=None, extra_args=None):
    from driftlab.agents.reference import make_agent
    from driftlab.runner import expand, run_cells

    from driftlab.profile import report

    args = cli(name, doc, extra_args)
    args.out = args.out or out_dir(name, args.world, cells)
    if args.analyze:
        import contextlib
        import io
        from driftlab.live import LIVE
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            analyze(args.out)
            print()
            report(args.out)
        LIVE.analysis(name, buf.getvalue())  # so the dashboard picks up refreshed tables too
        print(buf.getvalue())
        refresh_validation()
        return
    env = [{"world": args.world, **c} for c in cells(args.seeds or (1 if SMOKE else DEFAULT_SEEDS))]  # a cell may pin its own world
    grid = expand(env, resolve_agents(args, reference_agents))
    from experiments.registry import REGISTRY
    factory = lambda cell, scen, ctx: make_agent(cell["agent"], cell, scen, ctx)  # noqa: E731
    asyncio.run(run_cells(grid, scenario, factory, args.out, args.concurrency,
                          config={**(config or {}), "mode": MODE, "quick": SMOKE, "world": args.world,
                                  "registry": REGISTRY.get(name)}, resume=args.resume))
    import contextlib
    import io
    from driftlab.live import LIVE
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        analyze(args.out)
        print()
        report(args.out)
    LIVE.analysis(name, buf.getvalue())
    print(buf.getvalue())
    refresh_validation()
    print(f"done -> {args.out}   (dashboard: python -m driftlab.viz.server)")


def ref(name: str, substrate: dict, **extra) -> dict:
    return {"name": name, "type": "reference", "substrate": substrate, **extra}


NOTES = {"kind": "notes", "every": 10}
TRANSCRIPT = {"kind": "transcript", "window": 40}
SKILLS = {"kind": "skills", "every": 10}
NONE = {"kind": "none"}
