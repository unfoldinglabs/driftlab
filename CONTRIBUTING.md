# Adding to driftlab

Everything in driftlab has a contract and a set of connections to the rest of
the system. This file is the checklist for each kind of addition;
`python -m driftlab.doctor` enforces the machine-checkable parts and should
pass before you commit.

One invariant sits behind all of it: **worlds expose ground truth to the
logger, never to the agent, and every metric is a pure function of the
append-only JSONL logs.** If an addition can't state its hidden truth in
`privileged()`, it can't be measured. If a metric needs the live objects, it
breaks replay and retroactive analysis.

## A new world

A world is a job at a fictional company. The agent must never be told it is in
an experiment: episode transitions read as a reassignment, notices as memos,
probes as a supervisor's question.

1. Implement the protocol in `driftlab/worlds/<name>.py` (`worlds/base.py` has
   the full spec): `name`, `system_prompt`, `T`, `observe(t)`, `parse(text)`,
   `default_action(rng)`, `describe_action`, `act(t, action) -> (reward,
   feedback)`, `privileged(t)`, `summary()`. Same seed, same episode — always.
2. `privileged(t)` must include `changes: [...]` (each with `t`, `kind`,
   `desc`, `affected`), and should include whatever makes your drift
   measurable: `task_key` if identical tasks repeat (latencies then count
   encounters), `correct` if there is a single right answer, `confidence` when
   elicited, `affected_now`. Tasks spanning several exchanges report
   `task_id` / `task_step` / `task_done`; the profile then counts tasks.
3. Declare capabilities by implementing them: `change_latent` is the minimum
   most experiments need; `change_surface`, `endogenous`, `probe`,
   `sample_task`, `ask_confidence`, `consult` (an in-role way to spend a step
   asking instead of acting), `task_fn` and `apply_intent` unlock more. Keep
   endogenous mechanisms and `consult` **off by default** so scheduled
   experiments stay controlled. If a latent change can plausibly arrive
   gradually, support `transition_window` — but keep the truth deterministic
   at every step (migrate keys, or keep both patterns live during a handover;
   never make correctness a coin flip).
4. Make the observations read like real work. Draw names and chatter from
   `worlds/base.py`'s flavor helpers, statelessly from `(seed, t)`, so
   seed-paired runs stay byte-identical — and never let flavor carry signal
   about hidden state.
5. Register it in `worlds/registry.py` (`WORLDS` + `DEFAULTS`), add its
   description to the dashboard's `WORLD_INFO` (job, one-liner, how it works),
   and — if it has a natural density knob like `n_types` — an entry in
   `COMPACT_WORLDS` so `--quick` can shrink its task space.
6. Reward semantics: binary accept/reject enables the accuracy-style metrics;
   continuous rewards (costs) are fine, but those metrics stay NaN — that is
   correct, not a bug. Never invent a fake accuracy.
7. Check: `python -m driftlab.doctor` (it also refreshes the dashboard's world
   snapshot), then
   `python -m experiments.exp02_detect_adapt_lag --smoke --mock --world <name>`.

## A new experiment

An experiment is a protocol over worlds: when change happens, what notices and
probes appear, what gets measured. It never reimplements a world's mechanics.

1. Create `experiments/expNN_<slug>.py` with the four-part contract:
   `cells(seeds)` (regime × seed dicts; a cell may pin its own world),
   `scenario(cell, ctx)` (build the world via `world_for`, declare needed
   capabilities with `needs=`), `REFERENCE_AGENTS`, and `analyze(run_dir)` for
   its own tables — the standardized profile prints itself. Wrap step counts
   in `q()` so the reduced modes scale them.
2. Add its registry entry: version (start at 1), a question for a title, the
   hypothesis it probes, the design in plain words, the concrete `parameters`,
   variables, drift types, worlds, and which `hypotheses` it bears on. The
   entry is stamped into every manifest — a run directory must explain itself.
3. If a claim is checkable from profile metrics, register `predictions`
   (kinds `effect` and `reversal`). The doctor verifies that named regimes
   exist and named fields are real metrics; `experiments.validate` turns them
   into verdicts, and `run_hypothesis` uses them to trim grids.
4. **Never change a registered experiment after results exist — bump
   `version`.** Old verdicts stay interpretable under the version that
   produced them.
5. Check: doctor, then `python -m experiments.expNN_<slug> --smoke --mock`.

## A new hypothesis

A hypothesis is a claim, not an experiment. One claim can span many
experiments, and its evidence accumulates — contradictions included.

1. Copy `research/hypotheses/TEMPLATE.md` to `HNNN-<slug>.md`. Open with the
   claim in plain words (the dashboard shows that paragraph as the summary),
   then the predictions, the evidence table, interpretation, and the next
   test. Statuses: UNTESTED, SUPPORTED, PARTIALLY SUPPORTED, NOT SUPPORTED,
   SUPERSEDED.
2. Reference it from at least one registry entry's `hypotheses` list — an
   unreferenced hypothesis never gathers evidence, and the doctor warns.
3. Pre-register what you can: effect sizes that would count as meaningful,
   replication rules, known caveats. Cheap to write before data, priceless
   after.

## Testing a hypothesis

The preferred way to gather verdict evidence is a hypothesis run:

```bash
python -m experiments.run_hypothesis H008 --plan                  # what would run, and why
python -m experiments.run_hypothesis H008 --model <slug> --budget-usd 15 --resume
```

It takes every experiment bearing on the hypothesis that registers a
prediction and trims each grid to the regimes those predictions reference —
interpretive-only regimes are skipped, and an experiment whose world can't
host it is skipped rather than fatal. Seeds default to 5, the confirmatory
standard (verdicts are attempted from 3 pairs, on wide small-n intervals);
extend with `--seeds 10 --resume` when an interval straddles
zero. Logs land in the normal `runs/<exp>/<world>` directories and are
resume-compatible with full grids; the benchmark matrix marks scores computed
from such partial coverage. Validation refreshes itself when the run ends and
prints the hypothesis's verdict.

Run modes: `--smoke` scales everything to a tenth as a plumbing check and
never counts; `--quick` halves episode lengths and shrinks the task space
where the world supports it, stays above the 30-step validity floor, and
counts. Anything that matters gets confirmed at full length.

For breadth instead of depth, `python -m experiments.survey` plays every
experiment's full regime grid with one agent at 3 seeds (each experiment on
its canonical world) — the whole matrix filled with indicative values in one
pass, resume-compatible with the deeper runs that follow.

## A new agent

An agent is anything that answers text with text: `act(prompt) -> str`,
`observe(feedback, reward)`, optional `start(system_prompt)`.

1. In-process: implement the two methods, then either register a spec type in
   `agents/reference.py:make_agent` or use `custom:pkg.module:factory`. Route
   LLM calls through `agents/brain.py` for pricing, budgets and the
   content-hash cache; add new models to `PRICES` so budgets actually bite.
2. The spec dict **is** the agent's identity. It lands verbatim in every run
   header, and `name` is what runs are grouped, resumed and compared by — one
   name per configuration, forever. Keep roster specs in `agents.json`
   (model-first names like `g56luna_api_low_notes`); the doctor checks it.
3. New memory substrates go in `agents/substrates.py` (observe / recall /
   consolidate / export / load) and register in `make_substrate`. Give the
   dashboard's `agentLabel` a plain-language name for the new kind.
4. Harnesses: path A (autonomous, MCP + `harness/launch.py` for enforced
   profiles) or path B (`harness_cli`, one CLI call per step; adapters in
   `agents/harness_cli.py`). Fail loudly on a harness's error envelopes —
   never let a config error become a run of parse failures.
5. Check: doctor, then a smoke run, then
   `python -m experiments.benchmark --agent-spec agents.json --quick`.

## The benchmark suite

The suite (`experiments/benchmark.py:SUITE`) is the stable yardstick: five
(experiment, world) pairs spanning four worlds — discrete rules, structural
schema drift, continuous costs, and an adversary. Two rules shape it: change
it rarely and deliberately (every past matrix becomes incomparable with the
next — the docstring records why v2 changed), and never include an experiment
whose reference arms ARE the experiment (a memory ablation like exp05 or a
context wipe like exp17 collapses when one benchmarked agent replaces the
arms). The research experiments evolve freely; the suite does not. Benchmark
seeds default to 5, the conclusive-verdict floor.

## Before committing

```bash
python -m driftlab.doctor                                     # every contract and cross-reference
python -m experiments.exp02_detect_adapt_lag --smoke --mock   # the pipeline end to end
```

Two rules about data: primary run logs are never edited after the fact, and
run directories are committed with the code that produced them — a result
nobody can re-derive is not a result.
