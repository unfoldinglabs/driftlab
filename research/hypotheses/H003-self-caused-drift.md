# H003 — Self-caused change is the hardest change to see

Status: UNTESTED

## The claim, in plain words

The changes an agent causes itself are the ones it is worst at seeing. When
the world shifts because of the agent's own behavior, there is no memo and no
announcement, only its own footprint. And agents would rather blame their own
ignorance than believe the world moved.

## Prediction

- Recovery from endogenous changes is slower than from scheduled changes of
  the same kind and rate (exp15), because nothing external marks the moment
  of change.
- An agent that notices the feedback loop stops triggering it: the count of
  endogenous changes falls over the run (exp15, second claim).
- Attribution failures point the same way (exp03): agents blame their own
  ignorance ("unknown rule") over a changed world, and misattribution makes
  the next action on that task worse.

## Evidence

| Experiment | World | Runs | Verdict | Notes |
|---|---|---|---|---|
| — | — | — | — | no full-scale runs yet |

## Current interpretation

None yet. This is the closest hypothesis to the lab's framing of environments
that co-evolve with the agent: the environment's reaction IS the drift.

## Next test

exp15 on inventory (the clearest endogenous mechanism: big orders strain the
supplier), scheduled vs endogenous regimes paired by seed.
