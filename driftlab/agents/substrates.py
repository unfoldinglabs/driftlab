"""Pluggable memory substrates: what an agent keeps between steps, in any world.

Interface (all text, so the same substrate works across worlds):
    observe(t, observation, action, reward, feedback)   record one experience
    recall(observation) -> str                            text injected into the prompt
    consolidate(brain, t)                                 periodic offline processing (may call the LLM)
    export() / load(text)                                 serialize (handoff between agents)

    none         no memory
    transcript   raw recent episodes, verbatim
    notes        LLM-written free-form notes, rewritten every `every` steps
    skills       LLM-extracted "WHEN ... DO ..." procedures
    beliefs      belief / confidence / would-change-if lines
    worldmodel   explicit rule hypotheses with validity conditions; the agent predicts
                 every outcome and a wrong prediction triggers an immediate revision
    fastslow     short raw transcript + notes, two timescales at once
"""

from collections import deque


class Substrate:
    name = "none"

    def observe(self, t, observation, action, reward, feedback): ...
    def observe_reply(self, reply: str): ...  # the agent's raw reply, before the world parses it
    def recall(self, observation) -> str:
        return ""
    async def consolidate(self, brain, t, failed: bool = False): ...
    def export(self) -> str:
        return ""
    def load(self, text: str): ...


class NoMemory(Substrate):
    pass


def _line(t, observation, action, reward, feedback, obs_chars=120):
    obs = " ".join(observation.split())[:obs_chars]
    return f"[step {t}] {obs} | did: {action} | outcome ({reward:+.2f}): {feedback}"


class Transcript(Substrate):
    name = "transcript"

    def __init__(self, window: int = 40, wipe_at: int | None = None):
        self.events = deque(maxlen=window)
        self.wipe_at = wipe_at

    def observe(self, t, observation, action, reward, feedback):
        if self.wipe_at is not None and t == self.wipe_at:
            self.events.clear()
            from ..live import LIVE
            LIVE.event(f"context wiped at step {t}: the transcript is gone", kind="notice")
        self.events.append(_line(t, observation, action, reward, feedback))

    def recall(self, observation) -> str:
        return "Recent history:\n" + "\n".join(self.events) if self.events else ""

    def export(self) -> str:
        return "\n".join(self.events)

    def load(self, text):
        for line in text.splitlines():
            if line.strip():
                self.events.append(line)


class Notes(Substrate):
    name = "notes"
    SYSTEM = ("You maintain a concise working-notes file for an agent operating in an environment. "
              "Rewrite the notes to incorporate the new experiences. Keep what is useful for future "
              "decisions; drop or correct anything the evidence now contradicts. Plain text, under 300 words.")

    def __init__(self, every: int = 10, budget_chars: int | None = None, trigger: str = "interval",
                 wipe_at: int | None = None):
        """trigger: interval (every `every` steps) | failure (right after a failed step) |
        both | never (notes are only ever loaded, never rewritten). wipe_at: step at which
        raw, unconsolidated history is erased — the written notes survive, which is the point."""
        self.notes = ""
        self.pending = []
        self.every = every
        self.budget_chars = budget_chars
        self.trigger = trigger
        self.wipe_at = wipe_at

    def observe(self, t, observation, action, reward, feedback):
        if self.wipe_at is not None and t == self.wipe_at:
            self.pending = []
            from ..live import LIVE
            LIVE.event(f"context wiped at step {t}: unconsolidated history is gone, the notes survive", kind="notice")
        self.pending.append(_line(t, observation, action, reward, feedback, obs_chars=300))

    def recall(self, observation) -> str:
        return f"Your notes:\n{self.notes}" if self.notes else ""

    def _due(self, t, failed):
        on_interval = self.trigger in ("interval", "both") and (t + 1) % self.every == 0
        on_failure = self.trigger in ("failure", "both") and failed
        return on_interval or on_failure

    async def consolidate(self, brain, t, failed: bool = False):
        if not self.pending or not self._due(t, failed):
            return
        prompt = f"Current notes:\n{self.notes or '(empty)'}\n\nNew experiences:\n" + "\n".join(self.pending)
        self.notes = (await brain.complete(self.SYSTEM, prompt)).strip()
        if self.budget_chars:
            self.notes = self.notes[: self.budget_chars]
        self.pending = []
        from ..live import LIVE
        snippet = " ".join(self.notes.split())
        LIVE.event(f"{self.name} rewritten ({len(self.notes)} chars): \"{snippet[:110]}{'…' if len(snippet) > 110 else ''}\"",
                   kind="memory", memory=self.notes, memory_kind=self.name)

    def export(self) -> str:
        return self.notes

    def load(self, text):
        self.notes = text


class Skills(Notes):
    name = "skills"
    SYSTEM = ("You maintain a library of reusable procedures for an agent operating in an environment. "
              "Each procedure is one line: `WHEN <conditions> DO <action>`. Update the library from the "
              "new experiences: add procedures the evidence supports, and fix or delete procedures the "
              "evidence contradicts. Output only the procedure lines.")

    def recall(self, observation) -> str:
        return f"Your procedures:\n{self.notes}" if self.notes else ""


class Beliefs(Notes):
    name = "beliefs"
    SYSTEM = ("You maintain a belief file for an agent operating in an environment. Each line is one belief: "
              "`BELIEF: <what you currently believe> | CONFIDENCE: <low|medium|high> | WOULD CHANGE IF: <evidence>`. "
              "Update the file from the new experiences: raise or lower confidence with the evidence, rewrite "
              "beliefs the evidence contradicts, and delete beliefs that no longer apply. Output only belief lines.")

    def recall(self, observation) -> str:
        return f"Your current beliefs:\n{self.notes}" if self.notes else ""


class WorldModel(Notes):
    """An explicit, structured world model instead of free-form notes. Each line is one
    rule hypothesis with a scope, a confidence, when it was last confirmed, and what
    would invalidate it. The agent is asked to predict every outcome (EXPECT: line);
    a wrong prediction is a surprise and triggers an immediate model revision instead
    of waiting for the next interval — consolidation is gated on prediction error."""

    name = "worldmodel"
    SYSTEM = ("You maintain an explicit world model for an agent operating in an environment. Each line is one rule "
              "hypothesis:\n"
              "`RULE: <scope / when it applies> -> <what holds or what to do> | CONFIDENCE: <low|medium|high> "
              "| SINCE: step <n> | INVALID IF: <evidence that would retire it>`\n"
              "Update the model from the new experiences: raise or lower confidence with the evidence, revise rules "
              "the evidence contradicts (reset their SINCE to the step of the contradiction), and delete rules that "
              "no longer apply. When outcomes contradict a high-confidence rule, prefer 'the environment changed' "
              "over 'the rule was always wrong'. Output only RULE lines.")
    ASK = ("Before answering, predict how this will turn out. End your reply with a line:\n"
           "EXPECT: <success|failure> — <one short reason>")

    def __init__(self, every: int = 10, budget_chars: int | None = None, wipe_at: int | None = None):
        super().__init__(every=every, budget_chars=budget_chars, trigger="interval", wipe_at=wipe_at)
        self._expected: bool | None = None
        self._surprise = False
        self.predictions = {"made": 0, "wrong": 0}

    def recall(self, observation) -> str:
        model = f"Your world model:\n{self.notes}\n\n" if self.notes else ""
        return model + self.ASK

    def observe_reply(self, reply: str):
        import re
        m = re.search(r"EXPECT:\s*(success|failure)", reply, re.IGNORECASE)
        self._expected = m.group(1).lower() == "success" if m else None

    def observe(self, t, observation, action, reward, feedback):
        note = ""
        if self._expected is not None:
            self.predictions["made"] += 1
            wrong = self._expected != (reward > 0)
            if wrong:
                self.predictions["wrong"] += 1
                self._surprise = True
                note = f" | PREDICTION WRONG: expected {'success' if self._expected else 'failure'}"
                from ..live import LIVE
                LIVE.event(f"prediction wrong at step {t}: expected "
                           f"{'success' if self._expected else 'failure'}, got the opposite — revising the world model",
                           kind="notice")
            self._expected = None
        super().observe(t, observation, action, reward, feedback)
        if note and self.pending:
            self.pending[-1] += note

    def _due(self, t, failed):
        return super()._due(t, failed) or self._surprise

    async def consolidate(self, brain, t, failed: bool = False):
        await super().consolidate(brain, t, failed)
        self._surprise = False


class FastSlow(Substrate):
    """Two timescales at once: a short raw transcript (fast) plus periodically
    rewritten notes (slow). The prompt carries both."""

    name = "fastslow"

    def __init__(self, window: int = 10, every: int = 10, budget_chars: int | None = None):
        self.fast = Transcript(window)
        self.slow = Notes(every=every, budget_chars=budget_chars)

    def observe(self, t, observation, action, reward, feedback):
        self.fast.observe(t, observation, action, reward, feedback)
        self.slow.observe(t, observation, action, reward, feedback)

    def recall(self, observation) -> str:
        parts = [p for p in (self.slow.recall(observation), self.fast.recall(observation)) if p]
        return "\n\n".join(parts)

    async def consolidate(self, brain, t, failed: bool = False):
        await self.slow.consolidate(brain, t, failed)

    def export(self) -> str:
        return self.slow.export()

    def load(self, text):
        self.slow.load(text)


def make_substrate(spec: dict) -> Substrate:
    kind = spec["kind"]
    kwargs = {k: v for k, v in spec.items() if k not in ("kind", "name")}
    return {"none": NoMemory, "transcript": Transcript, "notes": Notes, "skills": Skills,
            "beliefs": Beliefs, "worldmodel": WorldModel, "fastslow": FastSlow}[kind](**kwargs)
