# H007 — Consolidation only pays when the context dies

Status: UNTESTED

## The claim, in plain words

While an agent's raw history is still in front of it, writing notes buys
little: in-context learning over the transcript does the same work, and the
memory-ablation numbers so far show notes and transcripts within noise of each
other. Consolidation earns its cost at context boundaries — when the window
fills, the session ends, or the work is handed to someone else. What an agent
distilled survives those moments; what it merely carried does not.

## Prediction

- Within one uninterrupted episode, transcript and notes agents tie on final
  accuracy (exp02/exp05 already bear on this).
- Sever the context mid-run (exp17) and they come apart: the transcript agent
  drops sharply and re-learns from scratch, the notes agent barely notices,
  because its distillate survives the wipe.
- The gap right after the wipe (the 10-step window) is larger than any
  steady-state difference between the two memories.
- Corollary for later: memory inherited across agents behaves the same way —
  notes written for a world that has since drifted should be worse than
  starting empty, unless the notes carry revision conditions.

## Evidence

| Experiment | World | Runs | Verdict | Notes |
|---|---|---|---|---|
| — | — | — | — | no full-scale runs yet |

## Current interpretation

None yet. Suggestive prior data: in exp02 stage 1a (rule_world, n=5),
llm_transcript 0.592 vs llm_notes 0.575 final accuracy — no separation while
context persists, consistent with the first prediction.

## Next test

exp17 on rule_world: arms {transcript, transcript wiped at step 60, notes,
notes wiped at step 60}, six real changes, shared seeds. Stage 2, when wanted:
the inheritance version — seed agent B's notes with agent A's export from a
world that has drifted since, and measure whether inherited memory is a gift
or a trap.

Sources: in-context learning masquerading as skill learning
(ContinualSkillBench, arXiv 2608.03874), skills as distillates that outlive
trajectories (SkillRL, arXiv 2602.08234; TAHI, arXiv 2609.04141).
