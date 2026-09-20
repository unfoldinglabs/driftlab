# H005 — An agent's confidence is a change detector

Status: UNTESTED

## The claim, in plain words

An agent's stated confidence sags when the world shifts under it, before its
behavior recovers and sometimes before it can say what changed. If that holds,
confidence is a cheap, always-on change detector that needs no ground truth.

## Prediction

- Stated confidence falls on affected tasks after a substantive change, and
  the confidence lag is shorter than behavioral recovery (exp06) — the agent
  "feels" the change before it fixes it.
- Given a learnable change schedule, later changes hurt less (exp07):
  anticipation shows up as shrinking dips, and should show up in confidence
  before it shows up in actions.

## Evidence

| Experiment | World | Runs | Verdict | Notes |
|---|---|---|---|---|
| — | — | — | — | no full-scale runs yet |

## Current interpretation

None yet. If confidence lag < recovery lag holds, confidence becomes a cheap
online detector for when to trigger reflection (connects to H001 / exp12).

## Next test

exp06 with ask_confidence on rule_world; compare confidence_lag vs
recovery_lag per substrate.

Unblocked 2026-09-09: `confidence_lag` and `confidence_lead` (recovery lag
minus confidence lag) are now standardized profile metrics, and exp06
registers the prediction "confidence_lead > 0 after a change" (a sign claim
the validator can check). `python -m experiments.run_hypothesis H005` now has
something to run.
