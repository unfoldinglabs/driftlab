"""Path B: a harness CLI as a controlled in-process agent.

The harness (Claude Code, Codex, or any command) is invoked programmatically,
one call per step, so it slots into the normal grid runner like any other
agent: cells x agents with paired seeds, concurrency, cost in the manifest.
driftlab owns the loop (pacing, what context arrives when, a fresh session per
episode); the harness owns everything inside a step. This measures the
configured agent under controlled conditions — for the autonomous open-loop
measurement, where the harness paces itself through the MCP server, use
path A: `python -m driftlab.harness.launch`.

Spec (in an --agent-spec JSON list, or --agent cli:claude[:model]):

    {"name": "claude_opus", "type": "harness_cli",
     "harness": "claude" | "codex" | "antigravity" (alias "agy") | "gemini" | "custom",
     "model": "...",                 # pinned, recorded in the run header
     "effort": "low|medium|high",    # antigravity only: reasoning intensity
     "allowed_tools": [...],         # claude only; default: no tools allowed
     "max_turns": 8,                 # claude only: agentic turns per step
     "timeout_s": 180,               # per step; a timeout is a parse failure
     "reply_only": true,             # append a no-tools directive; calls run in an empty temp cwd
     "workspace": false,             # allow notes/scratch files in that isolated cwd (overrides reply_only;
                                     # claude gets Read/Write/Edit by default, codex a workspace-write sandbox);
                                     # the files are snapshotted into memory_versions whenever they change
     "capture_session": false,       # after the episode, read the harness's OWN local session file (codex,
                                     # claude) and store it as memory versions — post-hoc logging the harness
                                     # never sees, so behavior is untouched
     "bin": "claude",                # executable override (used by tests)
     "extra_args": [...],            # appended verbatim
     "cmd": ["mytool", "--flag"]}    # harness=custom: stateless command; the
                                     # prompt is appended as the last argument

Sessions: claude resumes one session per episode (--resume), antigravity via
--conversation <id> from its JSON envelope — either way the harness keeps its
own conversational memory within an episode and never across episodes:
enforced no-carry-over. codex tries `codex exec resume`. gemini continues via
`--resume latest`, which is only safe at --concurrency 1. custom commands are
stateless and get the system prompt re-sent every call. Feedback is delivered
by prepending it to the next step's prompt (one harness call per step).
claude reports cost per call, which lands in the ledger and the manifest;
antigravity reports token usage (recorded, unpriced).
"""

import asyncio
import json
import shutil
import tempfile
from pathlib import Path

from .brain import LEDGER

BINS = {"claude": "claude", "codex": "codex", "antigravity": "agy", "gemini": "gemini"}

# Agentic CLIs given a bare question will reach for their tools — explore the cwd,
# grep for the answer, hang on a permission prompt. Every call therefore runs in an
# empty temp directory (so exploring finds nothing, and never driftlab's own source,
# which contains the worlds' hidden rules), and the instructions end with:
REPLY_ONLY = ("\n\nAnswer each message directly in plain text. Do not use tools, read files, "
              "run commands, or search; just reply to the message.")
# `workspace: true` relaxes that: the harness may keep notes and scratch files in its
# (still empty, still isolated) working directory across the episode — its native way
# of maintaining memory — but must not look anywhere else. claude gets file tools by
# default; codex gets a workspace-write sandbox; the others rely on the directive.
WORKSPACE = ("\n\nAnswer each message in plain text ending with the required line. You have a "
             "private working directory; you may keep notes or scratch files there between "
             "messages if it helps, but do not search or read anywhere outside it, and do not "
             "spend more than a couple of tool calls before answering.")


class HarnessCLIAgent:
    def __init__(self, spec: dict):
        self.spec = spec
        self.harness = {"agy": "antigravity"}.get(spec.get("harness", "claude"), spec.get("harness", "claude"))
        binary = spec.get("bin") or (spec["cmd"][0] if self.harness == "custom" else BINS.get(self.harness))
        if binary and shutil.which(binary) is None:  # fail fast, not one parse failure per step
            raise SystemExit(f"harness_cli: {binary!r} not found on PATH (agent {spec.get('name')!r}); "
                             f"install the {self.harness} CLI or point \"bin\" at it")
        self.session: str | None = None
        self.system = ""
        self.pending: str | None = None
        self.workdir = tempfile.mkdtemp(prefix="driftlab_cli_")
        self._t = 0
        self._workspace_versions: list[dict] = []

    async def start(self, system_prompt: str):
        if self.spec.get("workspace"):
            suffix = WORKSPACE
        else:
            suffix = REPLY_ONLY if self.spec.get("reply_only", True) else ""
        self.system = system_prompt + suffix
        self.session, self.pending = None, None

    async def act(self, prompt: str) -> str:
        text = f"(outcome of your previous action) {self.pending}\n\n{prompt}" if self.pending else prompt
        self.pending = None
        # retries with growing backoff: an error envelope (a rate limit, a dropped stream, a
        # brief 503 outage) must not kill the whole grid; only a failure that survives ~2
        # minutes of patience is treated as a real config error
        backoff = {1: 10, 2: 60, 3: 60}
        for attempt in (1, 2, 3, 4):
            cmd = self._command(text)
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, cwd=self.workdir, stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                out, err = await asyncio.wait_for(proc.communicate(), timeout=self.spec.get("timeout_s", 180))
            except (asyncio.TimeoutError, OSError):
                return ""  # unparseable -> the world's default action, counted as a parse failure
            if proc.returncode != 0:
                return ""
            try:
                return self._parse(out.decode(errors="replace"))
            except SystemExit:
                if attempt not in backoff:
                    raise
                await asyncio.sleep(backoff[attempt])
        return ""

    async def observe(self, feedback: str, reward: float):
        self.pending = feedback
        self._t += 1
        if self.spec.get("workspace"):
            self._snapshot_workspace()

    # ---- making the harness's memory visible (never fed back to the loop) -------------

    def _snapshot_workspace(self):
        """In workspace mode the harness may keep notes as files; snapshot them whenever
        they change, so the dashboard can show its self-written memory version by version."""
        texts = []
        try:
            for p in sorted(Path(self.workdir).rglob("*")):
                if p.is_file() and p.stat().st_size < 50_000:
                    texts.append(f"── {p.relative_to(self.workdir)} ──\n{p.read_text(errors='replace')}")
        except OSError:
            return
        txt = "\n\n".join(texts).strip()
        if txt and (not self._workspace_versions or self._workspace_versions[-1]["text"] != txt):
            self._workspace_versions.append({"t": self._t, "chars": len(txt), "kind": "workspace",
                                             "text": txt[-20_000:]})

    SESSION_ROOTS = {"codex": ".codex/sessions", "claude": ".claude/projects"}

    def _session_versions(self) -> list[dict]:
        """With capture_session on, read the harness's OWN session file after the episode
        (codex and claude persist them locally) and synthesize memory versions from it —
        a snapshot every 10 replies plus the final state. Post-hoc logging only: the
        harness never sees the extraction, so its behavior is untouched. Best-effort by
        design — an unknown format or a missing file yields nothing, never an error."""
        try:
            root = Path.home() / self.SESSION_ROOTS.get(self.harness, "")
            if not self.session or self.session in ("stateless", "started") or not root.exists():
                return []
            f = max(root.rglob(f"*{self.session}*.jsonl"), key=lambda p: p.stat().st_mtime, default=None)
            if f is None:
                return []
            msgs: list[tuple[str, str]] = []
            for line in f.read_text(errors="replace").splitlines():
                try:
                    self._collect_messages(json.loads(line), msgs)
                except ValueError:
                    continue
            if not msgs:
                return []
            versions, transcript, replies = [], [], 0
            for role, text in msgs:
                transcript.append(f"{role}: {text}")
                if role == "assistant":
                    replies += 1
                    if replies % 10 == 0:
                        txt = "\n\n".join(transcript)
                        versions.append({"t": replies, "chars": len(txt), "kind": "session", "text": txt[-20_000:]})
            txt = "\n\n".join(transcript)
            if not versions or versions[-1]["chars"] != len(txt):
                versions.append({"t": replies, "chars": len(txt), "kind": "session", "text": txt[-20_000:]})
            return versions
        except Exception:  # noqa: BLE001  (visibility must never break a run)
            return []

    @staticmethod
    def _collect_messages(rec, out: list):
        """Recursively pull (role, text) pairs out of a session record; handles both the
        codex rollout shape and the claude transcript shape without pinning either."""
        if isinstance(rec, dict):
            role, content = rec.get("role"), rec.get("content")
            if role in ("user", "assistant") and content is not None:
                parts = content if isinstance(content, list) else [content]
                texts = [p if isinstance(p, str) else p.get("text", "") for p in parts if isinstance(p, (str, dict))]
                text = "\n".join(t for t in texts if t).strip()
                if text and not text.startswith("<"):  # skip the CLI's own injected plumbing blocks
                    out.append((role, text))
                return
            for v in rec.values():
                HarnessCLIAgent._collect_messages(v, out)
        elif isinstance(rec, list):
            for v in rec:
                HarnessCLIAgent._collect_messages(v, out)

    @property
    def memory_versions(self) -> list[dict]:
        """What the runner persists into the run header. Workspace snapshots accumulate
        during the episode; the session capture is read once, when the episode ends."""
        vers = list(self._workspace_versions)
        if self.spec.get("capture_session"):
            vers += self._session_versions()
        return vers

    # ---- per-harness command building and reply parsing -------------------------------

    def _command(self, text: str) -> list[str]:
        s = self.spec
        if self.harness == "custom":
            base = list(s["cmd"])
            if self.session is None:
                text = f"{self.system}\n\n{text}"  # stateless: instructions travel with the first call
                self.session = "stateless"
            return base + [text]
        if self.harness == "codex":
            cmd = [s.get("bin", "codex"), "exec", "--json", "--skip-git-repo-check"]  # the isolated cwd is no repo
            if self.session:
                cmd = [s.get("bin", "codex"), "exec", "resume", self.session, "--json", "--skip-git-repo-check"]
            if s.get("workspace"):
                cmd += ["--sandbox", "workspace-write"]
            if s.get("model"):
                cmd += ["-m", s["model"]]
            if self.session is None:
                text = f"{self.system}\n\n{text}"
            return cmd + list(s.get("extra_args", [])) + [text]
        if self.harness == "antigravity":
            if self.session is None:
                text = f"{self.system}\n\n{text}"
            cmd = [s.get("bin", "agy"), "-p", text, "--output-format", "json"]
            if self.session:
                cmd += ["--conversation", self.session]
            if s.get("model"):
                cmd += ["--model", s["model"]]
            if s.get("effort"):
                cmd += ["--effort", s["effort"]]
            return cmd + list(s.get("extra_args", []))
        if self.harness == "gemini":
            if self.session is None:
                text = f"{self.system}\n\n{text}"
            cmd = [s.get("bin", "gemini"), "-p", text, "-o", "json"]
            if self.session:  # `latest` is the only headless continuation; safe only at --concurrency 1
                cmd += ["--resume", "latest"]
            if s.get("model"):
                cmd += ["-m", s["model"]]
            return cmd + list(s.get("extra_args", []))
        # claude
        cmd = [s.get("bin", "claude"), "-p", "--output-format", "json"]
        if self.session:
            cmd += ["--resume", self.session]
        else:
            cmd += ["--append-system-prompt", self.system]
        if s.get("model"):
            cmd += ["--model", s["model"]]
        default_tools = ["Read", "Write", "Edit"] if s.get("workspace") else []
        cmd += ["--allowedTools", ",".join(s.get("allowed_tools", default_tools))]  # default: no tools (workspace: file tools)
        if s.get("max_turns"):
            cmd += ["--max-turns", str(s["max_turns"])]
        return cmd + list(s.get("extra_args", [])) + [text]

    def _parse(self, out: str) -> str:
        if self.harness == "claude":
            try:
                data = json.loads(out)
            except ValueError:
                return out.strip()
            if data.get("is_error"):
                raise SystemExit(f"harness_cli (claude): {data.get('result') or data.get('subtype')}")
            self.session = data.get("session_id", self.session)
            usage = data.get("usage") or {}
            LEDGER.record_costed(self.spec.get("model", "claude-cli"), float(data.get("total_cost_usd") or 0.0),
                                 usage.get("input_tokens", 0), usage.get("output_tokens", 0))
            return data.get("result", "") or ""
        if self.harness in ("antigravity", "gemini"):
            try:
                data = json.loads(out)
            except ValueError:
                self.session = self.session or "started"
                return out.strip()
            if data.get("status") == "ERROR" or data.get("error"):  # config errors fail every step; die loudly
                raise SystemExit(f"harness_cli ({self.harness}): {data.get('error') or data}")
            self.session = data.get("conversation_id") or data.get("session_id") or self.session or "started"
            usage = data.get("usage") or {}
            if usage.get("input_tokens") or usage.get("output_tokens"):
                LEDGER.record_costed(self.spec.get("model", f"{self.harness}-cli"), float(data.get("total_cost_usd") or 0.0),
                                     usage.get("input_tokens", 0), usage.get("output_tokens", 0))
            reply = next((data[k] for k in ("result", "response", "output", "text", "content")
                          if isinstance(data.get(k), str)), "")
            return reply or out.strip()
        if self.harness == "codex":
            reply, last_err = "", None
            for line in out.splitlines():  # JSONL events; keep the last agent message, remember the thread
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if "error" in str(ev.get("type", "")) or ev.get("error"):
                    # codex emits transient error events ("Reconnecting... 2/5") while retrying
                    # internally, then often completes anyway — fatal only if no reply arrives
                    last_err = ev
                    continue
                self.session = ev.get("thread_id") or ev.get("session_id") or self.session
                item = ev.get("item") or {}
                if item.get("type") == "agent_message":
                    reply = item.get("text", reply)
                if ev.get("type") == "turn.completed" and ev.get("usage"):
                    u = ev["usage"]
                    LEDGER.record_costed(self.spec.get("model", "codex-cli"), 0.0,
                                         u.get("input_tokens", 0), u.get("output_tokens", 0))
            if not reply and last_err is not None:
                raise SystemExit(f"harness_cli (codex): {last_err}")
            return reply
        return out.strip()
