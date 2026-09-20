# driftlab

A testbed for one question: how do agents cope when the world keeps changing
under them? driftlab supplies the changing worlds, the experiments, and the
measurements. It assumes nothing about how an agent remembers or learns — an
agent is anything that answers text with text:

```python
class Agent(Protocol):
    async def act(self, prompt: str) -> str                         # observation in, reply out
    async def observe(self, feedback: str, reward: float) -> None   # outcome of the last action
```

Every world is seeded and knows its own hidden state, so learning is measured
against ground truth rather than a benchmark score. The same measurements
apply whether the agent is an in-process LLM loop, a lookup-table baseline, or
an external coding agent playing through the MCP server.

This file is the short tour. [DESIGN.html](DESIGN.html) is the system map —
how every part works and connects (open it in a browser).
[CONTRIBUTING.md](CONTRIBUTING.md) has the checklists for adding worlds,
experiments, hypotheses and agents. `python -m driftlab.doctor` enforces the
contracts.

## Setup

```bash
uv sync --all-extras                 # numpy/scipy + openai (API agents) + mcp (harness server)
uv run python -m driftlab            # interactive menu over everything
uv run python -m experiments.exp02_detect_adapt_lag --smoke --mock   # end-to-end check, no credentials
```

API keys go in `.env`: `OPENAI_API_KEY` for OpenAI, `OPENROUTER_API_KEY` for
any vendor-prefixed slug (`openai/gpt-5.6-luna`, `anthropic/claude-sonnet-5`),
`OPENAI_BASE_URL` for other compatible providers. Harness CLIs (`claude`,
`codex`, `agy`, `gemini`) are found on PATH — always pin their model.

## The worlds

Six jobs at fictional companies. The agent is never told it is in an
experiment: changes arrive unannounced, notices read as office memos, probes
as a supervisor's question.

| World | The job | What really changes |
|---|---|---|
| rule_world | intake desk: route requests | a hidden routing rule flips |
| form_filler | entering orders from prose intake notes | the schema mutates |
| inventory | buyer for one product line | the demand rate jumps |
| codebase | developer building on an internal library | the style guide is revised; a helper is deprecated |
| claims_desk | approve or reject insurance claims | the fraud ring moves against the agent |
| campaign_desk | pick each day's outreach angle | audience tastes are redrawn; feedback only arrives pooled |

Each world can also change cosmetically (only the wording moves — reacting is
the mistake) and, opt-in, endogenously (the agent's own behavior triggers the
change).

## Experiments and hypotheses

Twenty experiment protocols, each named for the question it asks (does
detecting a rule change predict recovering from it? do agents over-react to
purely cosmetic change? does prediction-gated revision reduce over-updating?). Each has a versioned registry
entry with its design and, where checkable, a registered prediction:

```bash
python -m experiments.registry           # one line per experiment
python -m experiments.registry exp18     # the full entry
```

Claims live in `research/hypotheses/` (H001–H008), one per file, accumulating
evidence across experiments. The preferred way to test one is a **hypothesis
run** — it runs exactly the regimes the registered predictions need and ends
with a scope-qualified verdict:

```bash
python -m experiments.run_hypothesis H008 --plan     # what would run, and why
python -m experiments.run_hypothesis H008 --model openai/gpt-5.6-luna --budget-usd 15 --resume
```

To paint the whole picture for one agent cheaply, a **survey** plays every
experiment's full regime grid at 3 seeds and half length:

```bash
python -m experiments.survey --plan                                # the bill, nothing runs
python -m experiments.survey --model openai/gpt-5.6-luna --quick --resume
```

It fills the benchmark matrix, every experiment table, and a first read on
every hypothesis in one pass — 3 seeds is the verdict floor, where bootstrap
intervals are wide and only large effects turn conclusive. Because runs resume
at episode granularity, a later `run_hypothesis --seeds 5 --resume` confirms
by running only the extension.

## Agents

`agents.json` holds the roster; any experiment takes it with
`--agent-spec agents.json`. In-process agents pair a model with a memory
substrate (`none`, `transcript`, `notes`, `skills`, `beliefs`, `worldmodel` —
explicit rule hypotheses the agent revises the moment a prediction fails —
and `fastslow`). External harnesses play two ways: autonomously over MCP, or
invoked once per step like any other agent. Every LLM call is priced,
budgeted and cached.

Worlds can also price information itself: with `consult=True` an agent may
spend a step asking (the floor manager, the audit desk, the reviewer) instead
of acting — the answer is real and the step's reward is forfeited.

```bash
python -m experiments.exp02_detect_adapt_lag --agent tabular                       # non-LLM baseline
python -m experiments.exp02_detect_adapt_lag --agent cli:claude:claude-sonnet-5    # a harness, per step
python -m experiments.benchmark --agent-spec agents.json         # the yardstick: 5 experiments, 4 worlds
```

## Measurement

Every run directory reduces to the same standardized report (`profile.json`):
raw metrics (detection and recovery latency, retention, stale-memory and
over-update rates, calibration, cost per success — unmeasured stays `—`,
never zero), normalized into four 0–1 dimensions. The capability dimensions
(adaptation, knowledge, epistemics) average into the **driftlab score**;
efficiency — instruction-following and dollars — is reported beside it, never
blended in, so token pricing cannot move a capability ranking. Because every
agent plays the same seeded worlds, differences come as **paired effects**
with 95% bootstrap intervals, never bare percentages.

```bash
python -m driftlab.profile runs/exp02/rule_world
python -m driftlab.viz.server --open        # the dashboard, computed entirely from the logs
```

## Run modes

- **full** — the real thing.
- **`--quick`** — half-length episodes, smaller task space; stays valid and
  **counts**. Confirm anything that matters at full length.
- **`--smoke`** — a tenth of everything, one seed: a plumbing check that
  never counts. `--smoke --mock` needs no credentials.

## Conventions

- Worlds expose ground truth to the logger, never to the agent.
- Everything is seeded: same seed, same episode, byte for byte.
- Metrics are pure functions of the append-only logs; runs are written
  atomically, so anything resumes with `--resume`.
- `--budget-usd` aborts a run past its cap; `python -m driftlab.costs runs`
  reports all spend.
