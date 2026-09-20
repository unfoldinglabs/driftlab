# H001 — How an agent remembers decides how it adapts

Status: UNTESTED

## The claim, in plain words

How an agent stores what it learns (raw history, distilled notes, extracted
procedures, or nothing) matters more for coping with a changing world than
which model it is. And there is no best design: each one trades fast noticing,
fast recovery, durable knowledge and resistance to outdated beliefs against
the others, so the right memory depends on how the world changes.

## Prediction

How an agent remembers determines the split between detecting a change and
recovering from it, and the trade between retention and staleness:

- Detection and recovery lags come apart by substrate (exp02): raw transcripts
  detect fast but recover slowly; consolidated notes detect slower but recover
  in fewer encounters once they do.
- The best retention setting moves with drift rate (exp05): short windows win
  at high drift, long windows at zero drift — the crossover is measurable.
  (Caveat 2026-09-09: exp05 v1 sampled the full task space, so a 5-step window
  almost never met the same case twice — the short arms were starved, not
  tested. exp05 v2 uses a recurring caseload; read v1 rows under that flaw.)
- Event-triggered reflection beats time-triggered at equal cost (exp12).
- Whatever is acquired transfers: a trained agent re-adapts to fresh rules
  faster than a fresh agent, even when day-one accuracy collapses (exp14).
- Under delayed feedback the memory benefit shrinks or reverses (exp09),
  because stale associations get written before outcomes arrive.
- Under pooled feedback the memory benefit becomes absolute (exp19): when
  outcomes arrive only as periodic totals, a memoryless agent has nothing to
  connect a report to and stays at chance; an agent that records its own
  choices can correlate them with the totals.

## Evidence

| Experiment | World | Runs | Verdict | Notes |
|---|---|---|---|---|
| — | — | — | — | no full-scale runs yet |

## Current interpretation

None yet. The profile metrics to watch: detection_lag vs recovery_lag gap,
retention, stale_rate, and the paired memory-vs-no-memory effect per regime.

## Next test (phase 1 protocol, pre-registered 2026-09-02, before any data)

exp02 + exp05 with the reference substrates (none / transcript / notes /
skills; same model, gpt-5-mini at low effort) on shared seeds — the memory
architecture is the only thing that varies.

- Stage 1: 5 seeds on rule_world (pilot the substrate contrasts).
- Stage 2: extend to 10 seeds and replicate on form_filler. A status change
  requires the effect to hold in both worlds (the replication rule; no
  correction for multiple comparisons otherwise).
- Effect sizes of interest, chosen in advance: a substrate must shift
  recovery lag by >= 2 encounters (exp02) or final accuracy by >= 0.05
  (exp05) for the difference to count as meaningful, CI aside.
- Predictions as registered: exp02 "recovery lag differs between memory
  architectures"; exp05 "the retention-setting ranking reverses between zero
  drift and high drift".
- Prior evidence to beat: tabular vs codex_g56luna (n=5, rule_world) did NOT
  separate on recovery lag — H001 needs the controlled ablation to show more
  than two arbitrary architectures did.
