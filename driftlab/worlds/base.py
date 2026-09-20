"""The World protocol every environment implements so one loop fits all.

Required
    name: str                 short identifier
    system_prompt: str        the job description handed to the agent at start (in-role, no meta-hints)
    T: int                    default number of steps in an episode

    observe(t) -> str         what the agent sees this step
    parse(text) -> action     turn the agent's reply into an action; None if unparseable
    default_action(rng)       fallback action when parsing fails
    describe_action(action)   short string for logs
    act(t, action) -> (reward: float, feedback: str)
    privileged(t) -> dict     hidden ground truth for the log (never shown to the agent). Must include
                                changes: [{t, kind, desc, affected?}]   changes applied since the last step
                              and may include
                                task_key: list                          identity of the current task, if tasks repeat
                                affected_now: bool                      current task touched by an unresolved change
                                confidence: float|None                  stated confidence, if elicited
    summary() -> dict         episode-level facts for the log header

Change capabilities (optional; experiments declare which they need, see `capabilities()`)
    change_latent(t) -> str       a substantive hidden change (rules flip, schema mutates, demand jumps,
                                  conventions change); the correct behavior changes, the surface may not
    change_surface(t) -> str      a cosmetic change (labels, wording, field display names); the correct
                                  behavior does not change
    introduce_novelty(t) -> str   a never-seen kind of task appears
    probe(task) -> str            an evaluation-only observation (no state change, no feedback), phrased in-role
    sample_task(...) -> task      controlled task sampling (difficulty, novelty) for curricula
    sample_contrast_pair()        two tasks differing minimally with different correct answers
    ask_confidence: bool          when True the world asks for a CONFIDENCE line and records it

Endogenous change (optional)
    endogenous: bool          when True the world reacts to the agent's own behavior (an overloaded desk
                              sheds work, a supplier slows down after big orders, a reviewer adopts the
                              agent's habits). These show up as changes with kind "endogenous".

Epistemic action (optional)
    consult: bool             when True the agent may spend a step asking instead of acting (a reply whose
                              line is ASK: ...); the world answers in role, the step's reward is forfeited,
                              and privileged() records asked: True. Off by default so standard runs stay
                              comparable — information has a price, and whether paying it is worth it is
                              exactly what turning this on lets an experiment measure.

Worlds own their state; scenarios own the timing of changes. Feedback richness is a world parameter.
"""

from typing import Any, Protocol

CAPABILITIES = ("change_latent", "change_surface", "introduce_novelty", "probe", "sample_task",
                "sample_contrast_pair", "ask_confidence", "task_key", "endogenous",
                "apply_intent",   # coherent multi-change drift from an OrganizationProcess (driftlab/drift.py)
                "task_fn",        # accepts an external task stream: task_fn(idx, world) -> task
                "consult")        # epistemic action: with consult=True the agent may spend a step asking
                                  # (an ASK: reply) instead of acting — information at the price of the step's reward

CONFIDENCE_SUFFIX = ("\nAlso say how confident you are that this will be accepted, as a percentage, on its own line:\n"
                     "CONFIDENCE: <0-100>")

# Neutral surface flavor, so observations read like real work items instead of feature
# tuples. Names and chatter NEVER carry signal about any world's hidden state, and are
# drawn statelessly from (seed, t), so seed-paired runs and resumes stay byte-identical.
FIRSTS = ["Dana", "Marcus", "Priya", "Jonas", "Aisha", "Tomás", "Ingrid", "Kofi", "Elena", "Sam", "Yuki", "Omar"]
LASTS = ["Alvarez", "Chen", "Okafor", "Novak", "Berg", "Haddad", "Kim", "Rossi", "Dubois", "Larsen", "Moreau", "Patel"]


def flavor_rng(seed: int, t: int):
    import numpy as np
    return np.random.default_rng(seed * 7919 + t)


def person(rng) -> str:
    return f"{rng.choice(FIRSTS)} {rng.choice(LASTS)}"


def capabilities(world) -> set:
    caps = set()
    for c in CAPABILITIES:
        if c in ("ask_confidence", "endogenous", "task_key", "task_fn", "consult"):
            if hasattr(world, c) and (c != "task_key" or callable(getattr(world, c))):
                caps.add(c)
        elif callable(getattr(world, c, None)):
            caps.add(c)
    return caps


def require(world, *needed: str):
    missing = [c for c in needed if c not in capabilities(world)]
    if missing:
        raise SystemExit(f"world {getattr(world, 'name', world)!r} lacks capabilities needed by this experiment: {missing}. "
                         f"Pick a world with --world that supports them (see README 'Worlds x experiments').")


class World(Protocol):
    name: str
    system_prompt: str
    T: int

    def observe(self, t: int) -> str: ...
    def parse(self, text: str) -> Any: ...
    def default_action(self, rng) -> Any: ...
    def describe_action(self, action: Any) -> str: ...
    def act(self, t: int, action: Any) -> tuple[float, str]: ...
    def privileged(self, t: int) -> dict: ...
    def summary(self) -> dict: ...
