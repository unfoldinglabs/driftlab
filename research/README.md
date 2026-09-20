# Research log

The hypothesis notebook: driftlab's cumulative memory of what was claimed,
what was tested, and what the evidence said. Experiments produce numbers;
this directory is where they accumulate into knowledge.

```
research/
  hypotheses/   HNNN-<slug>.md  one scientific claim each, with status and evidence
  findings/     FNNN-<slug>.md  confirmed results worth citing (created as they land)
```

Rules of the log:

- A **hypothesis** is a claim, not an experiment. One hypothesis can span many
  experiments (`experiments/registry.py` maps experiments -> hypotheses), and
  evidence for it accumulates across them — including contradictions.
- Every evidence row names the experiment, world, run directory and verdict,
  so any claim can be traced back to raw trajectories (the JSONL logs are the
  primary scientific artifact; everything here is interpretation).
- Statuses: `UNTESTED`, `SUPPORTED`, `PARTIALLY SUPPORTED`, `NOT SUPPORTED`,
  `SUPERSEDED (by HNNN)`. Update the status when evidence lands; never delete
  the history that led to it.
- Registered experiments carry a version; cite it in evidence rows
  (`exp04 v1`) so a result stays interpretable after an experiment changes.

Start a new hypothesis from `hypotheses/TEMPLATE.md`.
