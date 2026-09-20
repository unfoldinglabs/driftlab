"""Path A: launch an autonomous harness against the MCP server with an enforced profile.

The harness runs the experiment itself — it paces its own act() calls, manages
its own context across episodes, and decides when to use the notebook. That
autonomy is part of what is measured. What this launcher adds is enforcement:
the model and any declared configuration are pinned by the process that spawns
the harness and stamped into every run header (via DRIFTLAB_AGENT_PROFILE,
marked profile_enforced), instead of trusting the harness to describe itself.

    python -m driftlab.harness.launch --harness claude --model claude-opus-5 \\
        --experiment 4 --world rule_world --label claude_opus5
    python -m driftlab.harness.launch --harness antigravity --model gemini-3.5-flash-medium \\
        --experiment 2 --label agy_g35flash --effort medium
    python -m driftlab.harness.launch --harness codex --model gpt-5.1 --experiment 2 --label codex_gpt51
    python -m driftlab.harness.launch ... --profile memory=notebook --profile scaffold_version=2.1
    python -m driftlab.harness.launch ... --dry-run          # print commands/config, run nothing

MCP wiring per harness: claude takes an ad-hoc --mcp-config file; antigravity
(`agy`) reads the workspace's .agents/mcp_config.json, which the launcher
writes (merging with whatever is there); gemini gets a project-scoped
`gemini mcp add --trust` first. In every case the enforced profile rides in
the server's environment, so the harness cannot misdescribe itself.

For the controlled per-step measurement (driftlab owns the loop, one session
per episode, cost tracked), use path B instead: `--agent cli:claude[:model]`
(also cli:codex, cli:antigravity, cli:gemini) on any experiment, or a
harness_cli spec in --agent-spec.

Logs land in runs/<exp>/<world>/ as usual; the matrix and profiles pick them
up like any other agent's runs. Cost is whatever the harness's own account
bills; it is not metered here.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / "driftlab" / "harness" / "mcp_server.py"

DRIFTLAB_TOOLS = ",".join(f"mcp__driftlab__{t}" for t in (
    "run_experiment", "act", "status", "abort", "list_experiments", "describe_experiment",
    "start_run", "write_notebook", "read_notebook"))

PROMPT = ("Use the driftlab MCP tools to run experiment {exp} end to end: call "
          "run_experiment(\"{exp}\", world=\"{world}\", seeds={seeds}, quick={quick}, agent_label=\"{label}\"{resume_arg}) "
          "once, then keep calling act(text) with your reply to each observation until the result says "
          "finished. Follow the reply format each result gives you. When it finishes, report the metrics.")


def _cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--harness", default="claude", choices=["claude", "codex", "antigravity", "gemini"])
    ap.add_argument("--model", default=None, help="pinned model, recorded in every run header")
    ap.add_argument("--effort", default=None, help="antigravity only: reasoning intensity low|medium|high")
    ap.add_argument("--experiment", required=True, help="e.g. 4 or exp04")
    ap.add_argument("--world", default="rule_world")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="tell the harness to skip episodes already complete for this label — pick up "
                         "a suite another account or machine started (share the runs directory)")
    ap.add_argument("--label", default=None, help="agent_label; default <harness>[_<model>]")
    ap.add_argument("--profile", action="append", default=[], metavar="KEY=VALUE",
                    help="extra enforced profile fields, repeatable")
    ap.add_argument("--extra-arg", action="append", default=[], help="passed to the harness CLI verbatim, repeatable")
    ap.add_argument("--dry-run", action="store_true", help="print the commands and config, run nothing")
    return ap.parse_args()


def build(args) -> dict:
    """Everything the launch needs: {profile, files, pre, cmd} — files are written
    and pre-commands run before cmd (a dry run only prints them)."""
    label = args.label or (args.harness + (f"_{args.model.replace('.', '').replace('-', '')}" if args.model else ""))
    profile = {"harness": args.harness, **({"model": args.model} if args.model else {}),
               **({"effort": args.effort} if args.effort else {}),
               **dict(kv.split("=", 1) for kv in args.profile)}
    prompt = PROMPT.format(exp=args.experiment, world=args.world, seeds=args.seeds,
                           quick=str(args.quick), label=label,
                           resume_arg=", resume=True" if args.resume else "")
    env_json = json.dumps(profile)
    server_cfg = {"command": sys.executable, "args": [str(SERVER)], "env": {"DRIFTLAB_AGENT_PROFILE": env_json}}
    plan: dict = {"profile": profile, "files": {}, "pre": []}

    if args.harness == "claude":
        cfg = Path(tempfile.mkdtemp(prefix="driftlab_")) / "mcp.json"
        plan["files"][cfg] = json.dumps({"mcpServers": {"driftlab": server_cfg}})
        cmd = ["claude", "-p", "--mcp-config", str(cfg), "--allowedTools", DRIFTLAB_TOOLS,
               "--output-format", "text"]
        if args.model:
            cmd += ["--model", args.model]
        plan["cmd"] = cmd + args.extra_arg + [prompt]
    elif args.harness == "antigravity":
        cfg = ROOT / ".agents" / "mcp_config.json"  # agy's workspace-local MCP config
        existing = json.loads(cfg.read_text()) if cfg.exists() else {}
        existing.setdefault("mcpServers", {})["driftlab"] = server_cfg
        plan["files"][cfg] = json.dumps(existing, indent=2)
        cmd = ["agy", "-p", prompt, "--dangerously-skip-permissions"]
        if args.model:
            cmd += ["--model", args.model]
        if args.effort:
            cmd += ["--effort", args.effort]
        plan["cmd"] = cmd + args.extra_arg
    elif args.harness == "gemini":
        plan["pre"].append(["gemini", "mcp", "add", "-s", "project", "--trust",
                            "-e", f"DRIFTLAB_AGENT_PROFILE={env_json}",
                            "driftlab", sys.executable, str(SERVER)])
        cmd = ["gemini", "-p", prompt, "--approval-mode", "yolo"]
        if args.model:
            cmd += ["-m", args.model]
        plan["cmd"] = cmd + args.extra_arg
    else:  # codex: ad-hoc MCP server via -c config overrides (inline TOML tables)
        cmd = ["codex", "exec",
               "-c", f"mcp_servers.driftlab.command={json.dumps(sys.executable)}",
               "-c", f"mcp_servers.driftlab.args=[{json.dumps(str(SERVER))}]",
               "-c", f"mcp_servers.driftlab.env={{DRIFTLAB_AGENT_PROFILE={json.dumps(env_json)}}}"]
        if args.model:
            cmd += ["-m", args.model]
        plan["cmd"] = cmd + args.extra_arg + [prompt]
    return plan


def _show(cmd: list[str]) -> str:
    return " ".join(json.dumps(c) if " " in c else c for c in cmd)


if __name__ == "__main__":
    args = _cli()
    plan = build(args)
    print(f"enforced profile: {json.dumps(plan['profile'])}")
    for path, content in plan["files"].items():
        print(f"config file: {path}\n  {content}")
    for pre in plan["pre"]:
        print("pre-command: " + _show(pre))
    print("command: " + _show(plan["cmd"]))
    if args.dry_run:
        sys.exit(0)
    if shutil.which(plan["cmd"][0]) is None:
        sys.exit(f"{plan['cmd'][0]!r} not found on PATH")
    for path, content in plan["files"].items():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content)
    # The profile also rides on the harness process itself, so the MCP server child
    # inherits it even when the harness connects through its own (global) driftlab
    # registration instead of the config this launcher wires up.
    env = {**os.environ, "DRIFTLAB_AGENT_PROFILE": json.dumps(plan["profile"])}
    for pre in plan["pre"]:
        subprocess.run(pre, cwd=ROOT, check=True, env=env)
    sys.exit(subprocess.run(plan["cmd"], cwd=ROOT, env=env).returncode)
