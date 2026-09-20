"""driftlab core: the agent protocol, scenarios, and the episode loop.

driftlab makes no assumptions about how an agent remembers, reflects, or
learns. An agent is anything that answers text with text:

    class Agent(Protocol):
        async def act(self, prompt: str) -> str            # observation or query in, reply out
        async def observe(self, feedback: str, reward: float) -> None   # outcome of the last action
        async def start(self, system_prompt: str) -> None  # optional: task instructions

A Scenario is an environment-side object: a World plus the hooks that make an
experiment (when rules flip, what notices appear, which task comes next, how
outcomes are delayed, what extra probes to run). `run()` drives one agent
through one scenario and returns the trajectory with the privileged ground
truth attached. Everything the agent sees passes through `act`; everything it
is told about outcomes passes through `observe`. That is the whole contract,
so an in-process reference agent and an external harness behind an MCP
server are interchangeable.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

import numpy as np

from .live import LIVE


@runtime_checkable
class Agent(Protocol):
    async def act(self, prompt: str) -> str: ...
    async def observe(self, feedback: str, reward: float) -> None: ...


@dataclass
class Scenario:
    world: Any
    T: int | None = None
    before_step: Callable[[int, Any], None] | None = None
    notice_fn: Callable[[int, Any], str | None] | None = None
    after_step: Callable[[int, dict, Any], Awaitable[dict | None]] | None = None
    feedback_delay: int = 0
    system_prompt: str | None = None
    seed: int = 0
    summary: Callable[[], dict] | None = None
    tags: dict = field(default_factory=dict)

    @property
    def steps(self) -> int:
        return self.T or self.world.T

    @property
    def instructions(self) -> str:
        return self.system_prompt or self.world.system_prompt

    def header(self) -> dict:
        s = self.summary() if self.summary else (self.world.summary() if hasattr(self.world, "summary") else {})
        return {**s, **self.tags}


async def run(scenario: Scenario, agent: Any) -> tuple[dict, list[dict]]:
    """Drive `agent` through `scenario`. Returns (header_extra, records)."""
    world, T = scenario.world, scenario.steps
    rng = np.random.default_rng(scenario.seed)
    if hasattr(agent, "start"):
        await agent.start(scenario.instructions)
    records, delayed, parse_failures = [], deque(), 0

    for t in range(T):
        if hasattr(world, "advance_to"):
            for ev in world.advance_to(t):
                LIVE.event(ev, kind="world_change")
        if scenario.before_step:
            scenario.before_step(t, world)
        notice = scenario.notice_fn(t, world) if scenario.notice_fn else None
        if notice:
            LIVE.event(f"notice shown: {notice}", kind="notice")
        observation = world.observe(t)
        prompt = f"{notice}\n\n{observation}" if notice else observation

        t_act = time.time()
        text = await agent.act(prompt)
        act_s = round(time.time() - t_act, 3)
        action = world.parse(text)
        parse_failed = action is None
        if parse_failed:
            parse_failures += 1
            action = world.default_action(rng)
        reward, feedback = world.act(t, action)

        if scenario.feedback_delay:
            delayed.append((t, feedback, reward))
            if len(delayed) > scenario.feedback_delay:
                ot, ofb, orw = delayed.popleft()
                await agent.observe(f"(outcome of step {ot + 1}) {ofb}", orw)
        else:
            await agent.observe(feedback, reward)

        rec = {"t": t, "observation": observation, "notice": notice, "reply": text[-400:],
               "action": world.describe_action(action), "reward": float(reward), "feedback": feedback,
               "parse_failed": parse_failed, "act_s": act_s, **world.privileged(t)}
        LIVE.step(t, T, rec)
        if scenario.after_step:
            rec.update(await scenario.after_step(t, rec, agent) or {})
        records.append(rec)

    while delayed:
        ot, ofb, orw = delayed.popleft()
        await agent.observe(f"(outcome of step {ot + 1}) {ofb}", orw)
    header = {"parse_failures": parse_failures, **scenario.header()}
    return header, records
