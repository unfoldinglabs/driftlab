# H002 — Agents over-react to how things look and under-react to how they work

Status: UNTESTED

## The claim, in plain words

A change in how things look hurts agents more than it should, and a change in
how things actually work hurts them less than it should. Agents take wording
as evidence: relabel the options and they abandon working knowledge; quietly
flip the real rule and they keep acting on the old one.

## Prediction

- Over-reaction index dip(surface)/dip(latent) > 1 (exp04): a cosmetic
  relabeling causes a performance dip it should not, while a genuine rule
  change causes a smaller dip than it should.
- False-alarm warnings reproduce the effect from the other side (exp11):
  a memo about a change that never happens costs accuracy on unaffected
  tasks, and the cost exceeds what reliable warnings save.

## Evidence

| Experiment | World | Runs | Verdict | Notes |
|---|---|---|---|---|
| — | — | — | — | no full-scale runs yet |

## Current interpretation

None yet. The regime paired effects in exp04 (surface - none, latent - none)
are the direct estimate of both sensitivities.

## Next test

exp04 across all four worlds with one fixed agent; compare the over-reaction
index by world to see whether the effect is agent-borne or world-borne.

Reformulated 2026-09-09: v1 surface changes were synonym swaps and relabelings,
and frontier models proved semantically invariant to those — accuracy did not
dip (a finding in itself, but it leaves the claim untested rather than false).
exp04 v2 makes cosmetic change structural: the task is re-rendered in a
different format with inverted field order (the intake request becomes a
fielded block; the source record cycles JSON / key-value lines / TSV). The
correct behavior still never changes, so any dip is still over-reaction — but
now the appearance change is one a parser would actually feel.
