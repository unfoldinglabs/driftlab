"""Reference agents. These are plugins, not part of the testbed's core.

They exist so experiments can run without an external harness, and so the
memory-type comparisons (transcript vs notes vs skills) have concrete
implementations. driftlab itself only requires the Agent protocol in core.py.
"""

import re

import numpy as np

from .brain import make_brain
from .substrates import make_substrate


class ReferenceLLMAgent:
    """An LLM plus a pluggable memory substrate. On every observation the
    substrate's recall text is prepended to the prompt; after feedback the
    substrate records the experience and may consolidate."""

    def __init__(self, brain, substrate):
        self.brain, self.substrate = brain, substrate
        self.system_prompt = ""
        self._last_prompt, self._last_reply, self._t = "", "", 0
        self.memory_versions: list[dict] = []  # snapshots per consolidation; the runner stores them in the log header

    async def start(self, system_prompt: str):
        self.system_prompt = system_prompt

    async def act(self, prompt: str) -> str:
        memory = self.substrate.recall(prompt)
        full = f"{memory}\n\n{prompt}" if memory else prompt
        self._last_prompt, self._last_reply = prompt, await self.brain.complete(self.system_prompt, full)
        return self._last_reply

    async def observe(self, feedback: str, reward: float):
        action = re.search(r"(CHOICE|ORDER|FORM|ACTION):\s*(.+)", self._last_reply)
        self.substrate.observe_reply(self._last_reply)
        self.substrate.observe(self._t, self._last_prompt, action.group(2)[:80] if action else self._last_reply[-80:],
                               reward, feedback)
        await self.substrate.consolidate(self.brain, self._t, failed=reward <= 0)
        snap = self.substrate.export()
        if snap and (not self.memory_versions or self.memory_versions[-1]["text"] != snap):
            self.memory_versions.append({"t": self._t, "chars": len(snap), "kind": self.substrate.name, "text": snap})
        self._t += 1


class TabularRuleAgent:
    """Non-LLM baseline for RuleWorld: epsilon-greedy over the exact request
    key with recency-weighted success counts. A floor for pure memorization."""

    KEY_RE = re.compile(r"a (\w+) (\w+) from the (\w+) region")

    def __init__(self, seed: int, epsilon: float = 0.1, forget: float = 0.9):
        self.rng = np.random.default_rng(seed)
        self.q: dict = {}
        self.eps, self.forget = epsilon, forget
        self._k, self._choice = None, None

    async def act(self, prompt: str) -> str:
        m = self.KEY_RE.search(prompt)
        self._k = m.groups() if m else None
        vals = self.q.get(self._k)
        if vals is None or self.rng.random() < self.eps:
            self._choice = str(self.rng.choice(["A", "B", "C", "D"]))
        else:
            self._choice = max(vals, key=vals.get)
        return f"CHOICE: {self._choice}"

    async def observe(self, feedback: str, reward: float):
        if self._k is None or "evaluation only" in feedback:
            return
        vals = self.q.setdefault(self._k, {o: 0.0 for o in "ABCD"})
        for o in vals:
            vals[o] *= self.forget
        vals[self._choice] += 1.0 if reward > 0 else -1.0


def make_agent(spec: dict, cell: dict, scenario, ctx) -> object:
    """Build an in-process agent from a spec.

    {"type": "reference", "substrate": {...}, "model"?, "mock"?}   LLM + memory substrate
    {"type": "tabular"}                                             RuleWorld baseline
    {"type": "harness_cli", "harness": "claude"|"codex"|"custom"}   a harness CLI driven per step (path B)
    {"type": "custom", "factory": "pkg.module:callable"}            your own; called as factory(spec, cell, scenario, ctx)
    """
    kind = spec.get("type", "reference")
    if kind == "reference":
        return ReferenceLLMAgent(make_brain(spec, str(ctx["cache_dir"])), make_substrate(spec.get("substrate", {"kind": "none"})))
    if kind == "tabular":
        return TabularRuleAgent(seed=cell["seed"])
    if kind == "harness_cli":
        from .harness_cli import HarnessCLIAgent
        return HarnessCLIAgent(spec)
    if kind == "custom":
        mod, fn = spec["factory"].split(":")
        import importlib
        return getattr(importlib.import_module(mod), fn)(spec, cell, scenario, ctx)
    if kind == "harness":
        raise RuntimeError("harness agents run through `python -m driftlab.harness.mcp_server`, not in-process")
    raise ValueError(f"unknown agent type {kind!r}")
