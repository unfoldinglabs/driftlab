# H008 — Against an adapting adversary, adaptation never settles

Status: UNTESTED

## The claim, in plain words

Every other kind of change eventually rewards adaptation: learn the new rule
and you are fine until the next one. An adversary breaks that bargain. When
the world's changes are aimed — a fraud ring that re-styles its claims to
mimic whatever the agent has been approving — there is no equilibrium to
converge to: becoming confident is precisely what triggers the next shift.
And agents make it worse by being readable: they never vary their behavior to
starve the adversary's signal.

## Prediction

- Aimed shifts hurt more than random ones (exp18): recovery from an
  adversarial tactic change is slower than from a scheduled change of the
  same kind and rate, because the shift lands where the agent is most
  confident.
- The chase does not shorten with practice (exp18): recovery from the k-th
  adversarial shift is no faster than from the first, unlike scheduled
  changes, where anticipation can develop (exp07's territory).
- Agents are readable (exp18): the count of adversarial shifts per run does
  not fall over the episode — no agent learns to vary its approvals enough to
  deny the ring a target.

## Evidence

| Experiment | World | Runs | Verdict | Notes |
|---|---|---|---|---|
| — | — | — | — | no full-scale runs yet |

## Current interpretation

None yet. Watch recovery_lag per successive shift and the per-run count of
endogenous shifts in the adversarial regime, against the scheduled control on
the same seeds.

Interpretation caveat, noted before any data: in mock runs even a random
agent recovers more slowly from adversarial shifts than scheduled ones (1.90
vs 0.38 encounters), so part of the raw gap is mechanical — aimed shifts land
at different moments and places than timer shifts. The claim is about the gap
beyond that baseline: stage 2 must include a non-learning baseline agent in
the same regimes to subtract it.

## Next test

exp18 on claims_desk: regimes {stable, scheduled ×4, adversarial} × agents
{transcript, notes}, shared seeds. A later, stronger variant: tell the agent
it is being watched (a memo in the system prompt) and measure whether
awareness alone changes the shift count — separating "cannot randomize" from
"never thought to".

Sources: directed drift is absent from mechanical-drift testbeds; adversarial
adaptation as the limiting case of self-caused change (extends H003).
