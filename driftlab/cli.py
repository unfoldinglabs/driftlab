"""Interactive mode: every driftlab action behind one menu.

    python -m driftlab

Walks you through configuring an agent profile, running experiments or the
benchmark suite, analyzing and validating results, launching a harness, and
starting the dashboard — without remembering each command. Every choice is
turned into the real command line and printed before it runs, so the menu
doubles as a reference for scripting the same action later.
"""

import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WORLDS = ["rule_world", "form_filler", "inventory", "codebase"]
PY = [sys.executable, "-m"]
DASH = {"proc": None}


# ---- small prompt helpers ------------------------------------------------------------

def ask(prompt: str, default: str = "") -> str:
    tail = f" [{default}]" if default else ""
    try:
        v = input(f"  {prompt}{tail}: ").strip()
    except EOFError:
        raise SystemExit(0)
    return v or default


def yes(prompt: str, default: bool = False) -> bool:
    v = ask(prompt + (" (y/n)"), "y" if default else "n").lower()
    return v.startswith("y")


def pick(title: str, options: list[tuple[str, str]], default: int = 1):
    """options: [(key, label)]; returns the chosen key, or None for empty input on default 0."""
    print(f"\n{title}")
    for i, (_, label) in enumerate(options, 1):
        print(f"  {i}. {label}")
    while True:
        v = ask("choose", str(default))
        if v.isdigit() and 1 <= int(v) <= len(options):
            return options[int(v) - 1][0]
        print("  pick a number from the list")


def run_cmd(args: list[str], confirm: bool = True, background: bool = False):
    line = shlex.join(args)
    print(f"\n  command: {line}")
    if confirm and not yes("run it now?", default=True):
        print("  not run — copy the command above to use it directly")
        return None
    try:
        if background:
            return subprocess.Popen(args, cwd=ROOT)
        subprocess.run(args, cwd=ROOT)
    except KeyboardInterrupt:
        print("\n  interrupted — back to the menu")
    return None


# ---- experiment / agent choices ------------------------------------------------------

def experiment_modules() -> dict:
    from experiments.registry import REGISTRY
    mods = {p.stem.split("_")[0]: p.stem for p in sorted((ROOT / "experiments").glob("exp*_*.py"))}
    return {eid: (mods[eid], REGISTRY.get(eid, {}).get("title", "")) for eid in sorted(mods)}


def agent_args() -> list[str]:
    opts = [("reference", "reference LLM agents (the experiment's default set; needs OPENAI_API_KEY)"),
            ("mock", "mock reference agents (no credentials, pipeline check only)"),
            ("tabular", "tabular baseline (RuleWorld memorization, no LLM)")]
    if (ROOT / "agents.json").exists():
        opts.append(("spec", "agents.json (the saved agent profile in the repo root)"))
    opts += [("cli", "a harness CLI, path B (claude / antigravity / codex / gemini, pinned config)"),
             ("custom", "your own factory (pkg.module:function)")]
    choice = pick("Which agent plays?", opts)
    if choice == "mock":
        return ["--mock"]
    if choice == "tabular":
        return ["--agent", "tabular"]
    if choice == "spec":
        return ["--agent-spec", "agents.json"]
    if choice == "cli":
        harness = pick("Which harness?", [(h, h) for h in ("claude", "antigravity", "codex", "gemini")])
        model = ask("model (blank = the harness's default)")
        return ["--agent", f"cli:{harness}:{model}" if model else f"cli:{harness}"]
    if choice == "custom":
        return ["--agent", "custom:" + ask("factory, as pkg.module:function")]
    return []


def common_flags() -> list[str]:
    args = []
    seeds = ask("seeds per configuration (blank = experiment default)")
    if seeds:
        args += ["--seeds", seeds]
    if yes("quick mode? (tenth-length sense check, numbers are noise)"):
        args.append("--quick")
    if yes("resume? (skip episodes whose log is already complete)"):
        args.append("--resume")
    return args


# ---- menu actions --------------------------------------------------------------------

def run_experiment_flow():
    mods = experiment_modules()
    eid = pick("Which experiment?", [(e, f"{e}  {title}") for e, (_, title) in mods.items()])
    world = pick("Which world?", [(w, w) for w in WORLDS])
    args = agent_args() + ["--world", world] + common_flags()
    if yes("stream steps live in the terminal?"):
        args.append("--live")
    run_cmd(PY + [f"experiments.{mods[eid][0]}"] + args)


def benchmark_flow():
    if yes("only rebuild the matrix from existing logs (no new runs)?"):
        run_cmd(PY + ["experiments.benchmark", "--analyze"])
        return
    world = pick("Which world?", [(w, w) for w in WORLDS])
    run_cmd(PY + ["experiments.benchmark"] + agent_args() + ["--world", world] + common_flags())


def analyze_flow():
    choice = pick("What to analyze?", [
        ("exp", "one experiment's tables + standardized profile (over every log in its directory)"),
        ("profiles", "adaptation + drift profiles for every run directory"),
        ("costs", "LLM spend by model and experiment"),
        ("validate", "evaluate registered predictions and propose hypothesis verdicts")])
    if choice == "exp":
        mods = experiment_modules()
        have = [e for e in mods if (ROOT / "runs" / e).is_dir()]
        if not have:
            print("  no run directories yet — run an experiment first")
            return
        eid = pick("Which experiment?", [(e, f"{e}  {mods[e][1]}") for e in have])
        world = pick("Which world?", [(w, w) for w in WORLDS])
        run_cmd(PY + [f"experiments.{mods[eid][0]}", "--analyze", "--world", world])
    elif choice == "profiles":
        run_cmd(PY + ["driftlab.profile", "runs"])
    elif choice == "costs":
        run_cmd(PY + ["driftlab.costs", "runs"])
    else:
        target = ask("hypothesis or experiment to validate (blank = all)")
        args = PY + ["experiments.validate"] + ([target] if target else [])
        if yes("write verdicts into the hypothesis files? (--apply)"):
            args.append("--apply")
        run_cmd(args)


def agent_profile_flow():
    print("\nBuild an agent profile (a spec other commands can run via agents.json)")
    name = ask("name for this agent", "my_agent")
    kind = pick("What kind of agent?", [
        ("reference", "reference LLM + memory substrate (in-process)"),
        ("harness_cli", "a harness CLI driven per step, path B (claude / antigravity / codex / gemini)"),
        ("tabular", "tabular baseline"),
        ("custom", "your own factory")])
    spec: dict = {"name": name, "type": kind}
    if kind == "reference":
        spec["model"] = ask("model", "gpt-5-mini")
        spec["reasoning_effort"] = ask("reasoning effort", "low")
        sub = pick("Memory substrate?", [("none", "none"), ("transcript", "transcript window"),
                                         ("notes", "LLM-written notes"), ("skills", "extracted skills")])
        spec["substrate"] = {"kind": sub}
        if sub == "transcript":
            spec["substrate"]["window"] = int(ask("window (steps kept verbatim)", "40"))
        if sub in ("notes", "skills"):
            spec["substrate"]["every"] = int(ask("rewrite every N steps", "10"))
    elif kind == "harness_cli":
        spec["harness"] = pick("Which harness?", [(h, h) for h in ("claude", "antigravity", "codex", "gemini")])
        model = ask("model (blank = the harness's default)")
        if model:
            spec["model"] = model
        spec["timeout_s"] = int(ask("per-step timeout, seconds", "240"))
    elif kind == "custom":
        spec["factory"] = ask("factory, as pkg.module:function")
    path = Path(ask("write to", "agents.json"))
    path.write_text(json.dumps([spec], indent=2) + "\n")
    print(f"\n  wrote {path}:\n{json.dumps([spec], indent=2)}")
    print(f"  use it with: --agent-spec {path}  (any experiment, or the benchmark)")


def harness_flow():
    print("\nLaunch a harness autonomously, path A: it plays through the MCP server with a pinned profile")
    harness = pick("Which harness?", [(h, h) for h in ("claude", "antigravity", "codex", "gemini")])
    args = ["--harness", harness, "--experiment", ask("experiment (e.g. 4 or exp04)", "2")]
    model = ask("model to pin (blank = harness default)")
    if model:
        args += ["--model", model]
    label = ask("agent label (one per configuration; resumed episodes match on it)")
    if label:
        args += ["--label", label]
    if yes("quick mode?"):
        args.append("--quick")
    if yes("resume episodes already completed under this label?"):
        args.append("--resume")
    if yes("dry run first? (print what would happen, run nothing)", default=True):
        args.append("--dry-run")
    run_cmd(PY + ["driftlab.harness.launch"] + args)


def dashboard_flow():
    if DASH["proc"] and DASH["proc"].poll() is None:
        if yes("the dashboard is running — stop it?"):
            DASH["proc"].terminate()
            DASH["proc"] = None
            print("  stopped")
        return
    DASH["proc"] = run_cmd(PY + ["driftlab.viz.server", "--open"], background=True)
    if DASH["proc"]:
        print("  dashboard running in the background at http://localhost:8765 — this menu can keep working")


def registry_flow():
    target = ask("experiment for the full entry (blank = one line per experiment)")
    run_cmd(PY + ["experiments.registry"] + ([target] if target else []), confirm=False)


def doctor_flow():
    run_cmd(PY + ["driftlab.doctor"], confirm=False)


MENU = [
    ("run", "Run an experiment", run_experiment_flow),
    ("bench", "Run the benchmark suite (or rebuild its matrix)", benchmark_flow),
    ("agent", "Configure an agent profile (writes agents.json)", agent_profile_flow),
    ("harness", "Launch a harness autonomously (path A)", harness_flow),
    ("analyze", "Analyze: tables, profiles, costs, hypothesis validation", analyze_flow),
    ("registry", "Browse the experiment registry", registry_flow),
    ("viz", "Start or stop the dashboard", dashboard_flow),
    ("doctor", "Check the system (contracts and cross-references)", doctor_flow),
]


def main():
    print("driftlab — a testbed for agents learning while the world changes")
    print("(every action prints its command line first, so you can script it next time)")
    while True:
        try:
            choice = pick("What do you want to do?", [(k, label) for k, label, _ in MENU] + [("quit", "quit")])
        except KeyboardInterrupt:
            print()
            break
        if choice == "quit":
            break
        for k, _, fn in MENU:
            if k == choice:
                try:
                    fn()
                except KeyboardInterrupt:
                    print("\n  interrupted — back to the menu")
                break
    if DASH["proc"] and DASH["proc"].poll() is None:
        print(f"note: the dashboard is still running (pid {DASH['proc'].pid}); stop it with: kill {DASH['proc'].pid}")


if __name__ == "__main__":
    main()
