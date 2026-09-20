"""ChannelAgent: an Agent whose brain is on the other side of a message channel.

The scenario loop calls `act(prompt)`; the prompt is queued for the external
harness and the call blocks until the harness replies. `observe()` stores the
outcome so it can be delivered together with the next prompt. This is what
lets an MCP client (Claude Code, Codex, a script) be the agent without the
scenario code knowing anything about it.
"""

import asyncio


class ChannelAgent:
    def __init__(self):
        self.to_harness: asyncio.Queue = asyncio.Queue()
        self.from_harness: asyncio.Queue = asyncio.Queue()
        self.pending_feedback: dict | None = None
        self.system_prompt = ""

    async def start(self, system_prompt: str):
        self.system_prompt = system_prompt

    async def act(self, prompt: str) -> str:
        await self.to_harness.put({"prompt": prompt, "feedback": self.pending_feedback})
        self.pending_feedback = None
        return await self.from_harness.get()

    async def observe(self, feedback: str, reward: float):
        self.pending_feedback = {"feedback": feedback, "reward": reward}
