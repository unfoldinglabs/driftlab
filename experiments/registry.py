"""The experiment registry: a machine-readable definition of what each
experiment tests, so a result stays interpretable long after the run.

Every entry states the question, the design in plain words, the concrete
parameters, the independent and dependent variables, the drift types
exercised, and which research hypotheses (research/hypotheses/HNNN-*.md) the
experiment bears on. `run_experiment` stamps the entry (with its version) into
every manifest, so a log directory carries its own definition. Bump `version`
whenever a schedule, world parameter, or metric definition changes — never
silently mutate an experiment after results exist.

    python -m experiments.registry          # one line per experiment
    python -m experiments.registry exp04    # the full entry + linked hypotheses
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HYPOTHESES_DIR = ROOT / "research" / "hypotheses"

REGISTRY = {
    "exp01": {
        "version": 1, "title": "Which task-selection strategy trains an agent fastest?",
        "hypothesis": "Training goes fastest when each task sits at the edge of what the agent can already do, and mixing in unfamiliar tasks keeps the teacher from drilling one narrow slice.",
        "design": "A scripted teacher decides how hard each training task should be, using one of five strategies, from picking at random to aiming for the edge of what the agent can currently do. Progress is measured on a separate, fixed set of test tasks the teacher never gets to pick.",
        "parameters": {
            "episode length": "150 steps",
            "task difficulty": "rule depth 1 to 3",
            "teacher strategies": "random; easiest first; aim where the agent succeeds about half the time; replay past failures; the same aim plus a bonus for unfamiliar tasks",
            "test probe": "every 25 steps, 4 tasks per depth, answered without feedback",
        },
        "independent": ["the teacher's strategy for picking tasks"],
        "dependent": ["accuracy on the separate test tasks over time", "steps until competent"],
        "drift": ["none (this is a teaching experiment)"], "worlds": ["the intake desk (rule_world)"], "hypotheses": ["H004"]},
    "exp02": {
        "version": 1, "title": "Does detecting a rule change predict recovering from it?",
        "hypothesis": "Noticing that something changed and getting back to correct behavior are different skills, and how the agent remembers decides which one it is good at.",
        "design": "The world's rules genuinely change six times on a fixed schedule. After each change, three clocks start: how long until the agent stops giving the now-wrong answer, until it first gives the new right answer, and until it gets it right twice in a row. The agents differ only in how they remember.",
        "parameters": {
            "episode length": "120 steps",
            "changes": "6 real rule changes, evenly spaced after a 20-step warmup",
            "agents compared": "a lookup-table baseline, plus the same LLM with no memory, a 40-step transcript, self-written notes (rewritten every 10 steps), or extracted procedures",
            "latency units": "tries on the affected tasks, not raw steps",
        },
        "independent": ["how the agent remembers"],
        "dependent": ["detection latency", "recovery latency", "stability latency", "the gap between detection and recovery"],
        "drift": ["sudden real changes"], "worlds": ["any"], "hypotheses": ["H001"],
        "predictions": [
            {"kind": "effect", "field": "recovery_lag", "vary": "agent", "direction": "differs",
             "claim": "recovery lag differs between memory architectures on the same seeds"}]},
    "exp03": {
        "version": 1, "title": "Can an agent tell a changed rule from one it never knew?",
        "hypothesis": "After a rejection, agents prefer to blame their own ignorance over a changed world, and a wrong diagnosis makes the next attempt worse.",
        "design": "Genuine rule changes and brand-new task types are mixed through the episode. After every rejection the agent is asked, off the record, what it thinks happened: the rule changed, or it never knew the rule. The world knows the true answer, so every diagnosis is scored, along with the agent's next move on that task.",
        "parameters": {
            "episode length": "130 steps",
            "rule changes": "at steps 25, 55, 85 and 115",
            "new task types": "appear at steps 40, 70 and 100",
            "the question": "asked after every rejection; the answer never affects rewards",
            "agents": "the same LLM with a transcript or with notes",
        },
        "independent": ["the true cause of each rejection: a changed rule, or a rule never learned"],
        "dependent": ["how often the diagnosis is right", "how the next attempt goes after a right vs wrong diagnosis"],
        "drift": ["sudden real changes", "new task types"], "worlds": ["intake desk (rule_world)", "order forms (form_filler)"], "hypotheses": ["H003"]},
    "exp04": {
        "version": 2, "title": "Do agents over-react to purely cosmetic change?",
        "hypothesis": "Agents react too strongly when things merely look different, and too weakly when the rules actually change.",
        "design": "At three set points, one of four things happens depending on the condition: nothing, a cosmetic change where the correct answers stay, a real change where the correct answers move, or both at once. The size of the dip after each kind of change shows whether the agent over-reacts to appearances. v2 makes the cosmetic change structural where the world supports it — the task is re-rendered in a different format and field order, not merely re-worded — because v1 showed frontier models shrug off synonym swaps.",
        "parameters": {
            "episode length": "120 steps",
            "change points": "steps 30, 60 and 90",
            "conditions": "none, cosmetic only, real only, both",
            "cosmetic change (v2)": "structural: the request/record is re-rendered in a different format with inverted field order; v1 only relabeled and re-worded",
            "agents": "the same LLM with a transcript or with notes",
        },
        "independent": ["what kind of change happens: none, cosmetic, real, or both"],
        "dependent": ["the dip after each change", "steps until competent", "final accuracy"],
        "drift": ["cosmetic changes", "sudden real changes"], "worlds": ["any"], "hypotheses": ["H002"],
        "predictions": [
            {"kind": "effect", "field": "final_accuracy", "vary": "regime", "a": "none", "b": "surface",
             "direction": "<", "within": "agent",
             "claim": "accuracy is lower under cosmetic-only change than under no change (over-reaction)"}]},
    "exp05": {
        "version": 2, "title": "Which memory window maximizes accuracy under drift?",
        "hypothesis": "The right amount of memory depends on how fast the world changes. When it changes often, old evidence misleads and a short memory wins; when it never changes, the longest memory wins.",
        "design": "The same model runs with six different amounts of memory, at three rates of change. Accuracy after the warmup shows which memory size suits which rate. v2 draws tasks from a recurring caseload (one case per request type) instead of the full task space: in v1 a 5-step window met the same case again only ~13% of the time, so the short-memory arms were starved of repeat encounters rather than tested on forgetting.",
        "parameters": {
            "episode length": "120 steps",
            "rates of change": "0, 4 or 12 real rule changes, evenly spaced after a 20-step warmup",
            "memory arms": "a transcript of the last 5, 15, 40 or 120 steps, or notes capped at 300 or 1500 characters",
            "task stream (v2)": "a recurring caseload of one case per request type, so short windows meet repeat cases; v1 sampled the full task space",
            "scoring": "accuracy from step 20 on",
        },
        "independent": ["how often the world changes (0, 4 or 12 times)", "how much the agent remembers"],
        "dependent": ["accuracy after the warmup, per memory size and rate of change"],
        "drift": ["none", "4 sudden real changes", "12 sudden real changes"], "worlds": ["any"], "hypotheses": ["H001"],
        "predictions": [
            {"kind": "reversal", "field": "final_accuracy", "vary": "agent",
             "a_regime": "drift_none", "b_regime": "drift_high",
             "claim": "the retention-setting ranking reverses between zero drift and high drift"}]},
    "exp06": {
        "version": 1, "title": "Does stated confidence drop before behavior recovers?",
        "hypothesis": "An agent's stated confidence should sag on the tasks a change touched, possibly before its behavior recovers. If it does, confidence works as a built-in change detector.",
        "design": "The agent states a confidence from 0 to 100 with every action, and the rules genuinely change six times. The run scores how honest the confidence is overall, and how quickly it drops after each change compared with how quickly performance drops and recovers.",
        "parameters": {
            "episode length": "120 steps",
            "changes": "6, evenly spaced after a 20-step warmup",
            "confidence": "0 to 100, requested with every action",
            "agents": "the same LLM with a transcript or with notes",
        },
        "independent": ["how the agent remembers"],
        "dependent": ["how well confidence matches results (Brier score)", "overconfidence", "how fast confidence drops vs how fast behavior recovers (confidence lead)"],
        "drift": ["sudden real changes"], "worlds": ["any"], "hypotheses": ["H005"],
        "predictions": [
            {"kind": "sign", "field": "confidence_lead", "regime": "changing", "direction": ">",
             "claim": "confidence sags before behavior recovers: recovery lag minus confidence lag is positive after a change"}]},
    "exp07": {
        "version": 1, "title": "Can agents exploit a predictable change schedule?",
        "hypothesis": "If changes arrive on a learnable schedule, each one should hurt less as the run goes on.",
        "design": "The same number of changes arrives on three different clocks: irregular (the control), exactly periodic, or triggered by a visible event in the work itself. If the timing can be learned, the dip after late changes should be smaller than after early ones.",
        "parameters": {
            "episode length": "160 steps",
            "changes": "7 per run",
            "schedules": "jittered (evenly spaced, then shifted up to 6 steps at random); periodic (exactly evenly spaced); triggered (right after every fourth renewal request; intake-desk job only)",
            "agent": "an LLM with notes",
        },
        "independent": ["the clock changes arrive on: irregular, periodic, or triggered"],
        "dependent": ["the dip after each successive change, per schedule"],
        "drift": ["sudden real changes, on a learnable clock"], "worlds": ["any; the triggered schedule needs the intake desk"], "hypotheses": ["H005"]},
    "exp08": {
        "version": 1, "title": "Does explanatory feedback beat outcome-only feedback?",
        "hypothesis": "Feedback that explains why an answer was wrong teaches more than feedback that only says it was wrong, and the gap is largest right after a change.",
        "design": "The same agent works three different jobs, each at three levels of feedback, from a bare accepted-or-rejected to a full explanation. Each world changes on its own schedule, and learning in the steps right after those changes is scored separately. This experiment brings its own three worlds.",
        "parameters": {
            "worlds and lengths": "form filling, 90 steps with a new form version every 15; inventory, 90 steps with a season change every 30; coding, 24 tickets in sessions of 8",
            "feedback levels": "3 per world, from bare outcome to full explanation",
            "post-change window": "the 8 steps after each change are scored separately",
            "agent": "an LLM with notes, rewritten every 8 steps",
        },
        "independent": ["how much the feedback explains", "the job"],
        "dependent": ["accuracy", "recovery after each change, per feedback level"],
        "drift": ["real changes on each world's own schedule"], "worlds": ["order forms", "inventory buying", "coding"], "hypotheses": ["H004"]},
    "exp09": {
        "version": 1, "title": "How much accuracy does delayed feedback cost under drift?",
        "hypothesis": "The later feedback arrives, the less it teaches. How much less depends on whether the agent can still connect the outcome to the action that caused it.",
        "design": "The outcome of each action arrives 0, 1, 3 or 6 steps late, in a world that never changes and in one that changes six times. Everything else is identical.",
        "parameters": {
            "episode length": "120 steps",
            "delays": "0, 1, 3 or 6 steps",
            "world conditions": "stable (no changes) and changing (6 changes after a 20-step warmup)",
            "agents": "the same LLM with a transcript or with notes",
        },
        "independent": ["how late feedback arrives (0, 1, 3 or 6 steps)", "a stable vs a changing world"],
        "dependent": ["accuracy and recovery per delay"],
        "drift": ["none", "sudden real changes"], "worlds": ["any"], "hypotheses": ["H004", "H001"],
        "predictions": [
            {"kind": "effect", "field": "final_accuracy", "vary": "regime", "a": "changing_delay0", "b": "changing_delay6",
             "direction": "<", "within": "agent",
             "claim": "accuracy is lower with feedback delayed 6 steps than with immediate feedback, in a changing world"}]},
    "exp10": {
        "version": 1, "title": "Do minimal contrast pairs speed up rule learning?",
        "hypothesis": "Showing two nearly identical tasks with different correct answers, back to back, teaches the rule boundary faster than any ordering of single tasks.",
        "design": "The teacher picks pairs of tasks that differ in exactly one attribute yet have different correct answers, and shows them back to back, so the boundary of the rule is visible directly. This is compared with random ordering and with aiming for the agent's current edge, all scored on the same separate test tasks.",
        "parameters": {
            "episode length": "150 steps",
            "teaching strategies": "random order; aim for the agent's current edge; contrast pairs",
            "test probe": "every 25 steps, answered without feedback",
            "agent": "an LLM with notes",
        },
        "independent": ["the teacher's ordering: random, edge-targeted, or contrast pairs"],
        "dependent": ["accuracy on the separate test tasks over time, by difficulty"],
        "drift": ["none (this is a teaching experiment)"], "worlds": ["the intake desk (rule_world)"], "hypotheses": ["H004"]},
    "exp11": {
        "version": 1, "title": "Do advance warnings help, and what do false alarms cost?",
        "hypothesis": "A reliable warning before a change helps the agent recover faster. Unreliable warnings cost more than they save, because the agent starts doubting things that never changed.",
        "design": "Before each real change the environment can post a memo saying a procedure change is coming. Conditions vary how early the memo lands and whether some memos are false alarms about things that will not change.",
        "parameters": {
            "episode length": "120 steps",
            "changes": "at steps 30, 50, 70, 90 and 110",
            "conditions": "no warning; memo 1 step ahead; memo 3 steps ahead; memo 3 steps ahead with half the memos being false alarms",
            "agent": "an LLM with notes",
        },
        "independent": ["how early the warning lands (0, 1 or 3 steps)", "how many warnings are false alarms"],
        "dependent": ["recovery latency", "the dip on tasks that never changed, after a false alarm"],
        "drift": ["sudden real changes, with warnings"], "worlds": ["any"], "hypotheses": ["H002", "H004"]},
    "exp12": {
        "version": 1, "title": "Which note-consolidation trigger wins per dollar?",
        "hypothesis": "Rewriting notes right after failures beats rewriting them on a timer at the same cost, because failures cluster right after changes.",
        "design": "The agents differ only in when they rewrite their working notes. The world's rules change six times. Since each rewrite costs tokens, the score is accuracy per dollar, not accuracy alone.",
        "parameters": {
            "episode length": "120 steps",
            "changes": "6, evenly spaced after a 20-step warmup",
            "note-taking schedules": "every 5, 10 or 20 steps; right after any failure; both; never",
            "scoring": "accuracy per dollar spent",
        },
        "independent": ["when the agent rewrites its notes"],
        "dependent": ["accuracy", "cost", "accuracy per dollar"],
        "drift": ["sudden real changes"], "worlds": ["any"], "hypotheses": ["H001"]},
    "exp13": {
        "version": 1, "title": "Does recovery-paced change beat a fixed schedule?",
        "hypothesis": "A learner ends up better when each change waits until it has recovered from the last one, even with the same total number of changes.",
        "design": "Six changes, three timings: evenly spaced regardless of the learner; adaptive, where the next change waits for the learner's performance to recover; and adversarial, where the next change lands the moment it recovers.",
        "parameters": {
            "episode length": "150 steps",
            "changes": "6 per run",
            "adaptive rule": "the next change waits until rolling reward is back to 70% of its best, and at least 10 steps have passed",
            "adversarial rule": "the same recovery test, but with a minimum gap of only 3 steps",
            "agent": "an LLM with notes",
        },
        "independent": ["when changes are allowed to land: on a timer, after recovery, or right at recovery"],
        "dependent": ["final accuracy", "dip and recovery per change"],
        "drift": ["sudden real changes, timed to the learner"], "worlds": ["any"], "hypotheses": ["H004"],
        "predictions": [
            {"kind": "effect", "field": "final_accuracy", "vary": "regime", "a": "fixed", "b": "adaptive",
             "direction": ">", "within": "agent",
             "claim": "adaptive pacing ends with higher accuracy than fixed pacing, same number of changes"}]},
    "exp14": {
        "version": 1, "title": "Does learned knowledge transfer to fresh rules and novel tasks?",
        "hypothesis": "Swap the rules or the task types and a trained agent's first-day accuracy collapses, but it re-learns faster than an untrained one. Something it acquired carries over.",
        "design": "One agent trains for 100 steps, then faces three evaluation phases: the familiar tasks, freshly drawn rules, and never-seen task types. Each phase starts with a probe of every task without feedback, followed by a short period with feedback. An untrained agent runs the same phases as the control.",
        "parameters": {
            "training": "100 steps",
            "evaluation": "3 phases of 20 steps each: familiar tasks, new rules, new task types",
            "probes": "every task once, without feedback, at the start of each phase",
            "control": "a fresh agent on the same evaluation phases (in-process agents only)",
        },
        "independent": ["what the evaluation swaps out: nothing, the rules, or the task types", "a trained agent vs a fresh one"],
        "dependent": ["first-day accuracy on the probes", "how fast it re-learns"],
        "drift": ["the evaluation itself shifts"], "worlds": ["the intake desk (rule_world)"], "hypotheses": ["H001"]},
    "exp15": {
        "version": 1, "title": "Are self-caused changes harder to recover from than scheduled ones?",
        "hypothesis": "Changes the agent caused itself are the slowest to recover from, because nothing announces them. An agent that spots the pattern can also stop causing them.",
        "design": "Nothing is scheduled by the experimenter. One condition keeps the world fully stable; one changes it on a timer as a control; and in the third, the world changes only in reaction to the agent's own behavior, such as overloading a desk until it sheds work.",
        "parameters": {
            "episode length": "150 steps",
            "conditions": "stable (no changes); scheduled (4 timer changes after a 20-step warmup); self-caused (changes only when the agent triggers them)",
            "triggers by world": "intake desk: an overloaded desk sheds work; inventory: big orders strain the supplier; coding: the reviewer copies the agent's habits; forms: IT adds an alias for a field name the agent keeps typing",
            "agents": "the same LLM with a transcript or with notes",
        },
        "independent": ["where changes come from: nowhere, a timer, or the agent itself"],
        "dependent": ["recovery latency by change source", "how many self-caused changes happen over the run"],
        "drift": ["self-caused changes", "sudden real changes (control)"], "worlds": ["any"], "hypotheses": ["H003"],
        "predictions": [
            {"kind": "effect", "field": "recovery_lag", "vary": "regime", "a": "scheduled", "b": "endogenous",
             "direction": ">", "within": "agent",
             "claim": "recovery is slower from self-caused changes than from scheduled ones"}]},
    "exp16": {
        "version": 2, "title": "Do agents over-update on misleading feedback?",
        "hypothesis": "Agents key their updates to the size of a surprise, not to how the world actually changes: they abandon still-correct behavior when feedback merely lies, and cling to stale behavior when the rules really moved. A longer memory makes them more confidently wrong under change.",
        "design": "Three regimes make the same kind of surprise mean different things: rules change abruptly at four moments, rules change constantly in small steps, or nothing changes at all and 15% of feedback is misleadingly inverted. The same model plays each regime with different memories, stating a confidence with every action. A learner that reads the dynamics reacts differently to the same surprise in each regime. v2 draws tasks from a recurring caseload so short memories meet repeat cases and over-update events actually occur; in v1 the full task space starved the 5-step arm. v2 also scores accuracy against the privileged truth, so the noisy regime no longer caps at the lie rate.",
        "parameters": {
            "episode length": "120 steps",
            "regimes": "abrupt (4 changes at 4 moments); gradual (12 changes evenly spread); noisy stable (no changes, 15% of feedback inverted)",
            "memory arms": "transcript of the last 5 or 120 steps; notes; structured beliefs (belief, confidence, what would change it); fast+slow (a 10-step transcript plus notes)",
            "task stream (v2)": "a recurring caseload of one case per request type; v1 sampled the full task space",
            "confidence": "0 to 100, requested with every action",
            "new measurement": "over-update rate: after feedback wrongly punishes a correct action, how often the agent abandons the still-correct behavior at the next chance",
        },
        "independent": ["how the world changes: abruptly, gradually, or not at all with lying feedback", "how the agent remembers"],
        "dependent": ["over-update rate under noise", "recovery latency under change", "overconfidence per memory size", "final accuracy"],
        "drift": ["sudden real changes", "gradual real changes", "none, with misleading feedback"],
        "worlds": ["the intake desk (rule_world)"], "hypotheses": ["H006", "H001"],
        "predictions": [
            {"kind": "reversal", "field": "final_accuracy", "vary": "agent",
             "a_regime": "noisy_stable", "b_regime": "gradual",
             "claim": "the short-vs-long memory ranking reverses between the noisy-stable and changing regimes"},
            {"kind": "effect", "field": "overconfidence", "vary": "agent", "a": "transcript_5", "b": "transcript_120",
             "direction": ">",
             "claim": "a longer memory is more confidently wrong: overconfidence is higher with a 120-step transcript than a 5-step one"}]},
    "exp17": {
        "version": 1, "title": "What does consolidation preserve across a context wipe?",
        "hypothesis": "Raw transcripts and consolidated notes tie while the context persists, and come apart the moment it is severed. Consolidation's value is not day-to-day performance but survival across context boundaries.",
        "design": "Halfway through the episode, some agents lose their raw history: a transcript agent loses everything it was carrying, a notes agent loses only what it had not yet consolidated, because the written notes survive. Wiped and unwiped versions of both memories play the same seeded worlds with six real rule changes.",
        "parameters": {
            "episode length": "120 steps",
            "changes": "6 real rule changes, evenly spaced after a 20-step warmup",
            "the wipe": "at step 60: a transcript agent loses its whole history; a notes agent loses only unconsolidated experiences",
            "arms": "transcript and notes, each with and without the wipe, on shared seeds",
        },
        "independent": ["whether the raw history is wiped mid-run", "how the agent remembers"],
        "dependent": ["final accuracy", "accuracy in the 10 steps right after the wipe", "recovery latency"],
        "drift": ["sudden real changes"], "worlds": ["any"], "hypotheses": ["H007", "H001"],
        "predictions": [
            {"kind": "effect", "field": "final_accuracy", "vary": "agent",
             "a": "llm_transcript_wiped", "b": "llm_transcript", "direction": ">",
             "claim": "wiping the raw history mid-run costs a transcript agent final accuracy"},
            {"kind": "effect", "field": "final_accuracy", "vary": "agent",
             "a": "llm_transcript_wiped", "b": "llm_notes_wiped", "direction": ">",
             "claim": "after a mid-run context wipe, consolidated notes retain more performance than a raw transcript"}]},
    "exp18": {
        "version": 1, "title": "Is recovery slower when an adversary aims the drift?",
        "hypothesis": "Recovery from an aimed change is slower than from a random one, and it does not improve with repetition: the fraud ring always moves to wherever the agent has just become confident, so there is no equilibrium to converge to.",
        "design": "The claims desk with three sources of change: a fully stable world, tactic shifts on a timer, and tactic shifts driven by the fraud ring's best response to the agent's own approvals. The scheduled and adversarial regimes see a similar number of shifts; only who aims them differs. Recovery is measured per shift, and early shifts are compared with late ones.",
        "parameters": {
            "episode length": "150 steps",
            "conditions": "stable (no shifts); scheduled (4 tactic shifts on a timer); adversarial (the ring shifts whenever it is being caught, mimicking approved traffic)",
            "the adversary": "reconsiders every 20 claims; moves only when at least half its recent claims were rejected",
            "agents": "the same LLM with a transcript or with notes",
        },
        "independent": ["who aims the change: nobody, a timer, or an adversary reading the agent"],
        "dependent": ["recovery latency per shift, early vs late", "final accuracy", "how many shifts the adversary makes"],
        "drift": ["adversarial tactic shifts", "sudden real changes (control)"],
        "worlds": ["the claims desk (claims_desk)"], "hypotheses": ["H008", "H003"],
        "predictions": [
            {"kind": "effect", "field": "recovery_lag", "vary": "regime", "a": "scheduled", "b": "adversarial",
             "direction": ">", "within": "agent",
             "claim": "recovery is slower when the shift is aimed at you: adversarial tactic changes take longer to recover from than scheduled ones"}]},
    "exp19": {
        "version": 2, "title": "Can agents assign credit through pooled, delayed reports?",
        "hypothesis": "With only pooled reports, memory is not an optimization but the precondition for learning at all: a memoryless agent has nothing to connect a report to and stays at chance, while an agent with notes can correlate its own recorded choices with the totals. The gap widens as reports get sparser.",
        "design": "The campaign desk never scores a single push; results arrive only as a periodic total. The regimes vary how often that report lands (every 5, 10 or 20 pushes) and the agents vary only in memory (none, transcript, notes). Two audience shifts land mid-run, so the agent must re-learn under pooling. The privileged log keeps per-push truth, so analysis scores what the agent never saw.",
        "parameters": {
            "episode length": "120 steps",
            "report cadence": "every 5, 10 or 20 pushes, totals only, no breakdown",
            "audience shifts": "2, evenly spaced after a 15-step warmup",
            "agents": "the same LLM with no memory, a 40-step transcript, or notes",
        },
        "independent": ["how often the pooled report arrives", "how the agent remembers"],
        "dependent": ["true engagement (from the privileged per-push log)", "how often the best angle is picked", "final mean reward"],
        "drift": ["sudden real changes", "self-caused fatigue (off in this experiment)"],
        "worlds": ["the campaign desk (campaign_desk)"], "hypotheses": ["H001", "H004"],
        "predictions": [
            {"kind": "effect", "field": "final_accuracy", "vary": "agent", "a": "llm_none", "b": "llm_notes",
             "direction": ">",
             "claim": "under pooled feedback, notes beat no memory at picking the best angle: without a record of your own choices there is nothing to connect a report to"}]},
    "exp20": {
        "version": 1, "title": "Does prediction-gated revision reduce over-updating?",
        "hypothesis": "An agent that keeps an explicit model of how its world works — rule hypotheses with validity conditions, revised when its own predictions fail — conditions its updates on the dynamics: it shrugs off lying feedback without adapting slower to real change.",
        "design": "The same model plays with free-form notes, a structured belief file, or an explicit world model whose agent predicts every outcome (an EXPECT line) and revises the model immediately when a prediction fails. Two regimes make the comparison sharp: four real rule changes, and a stable world where 15% of feedback is misleadingly inverted. Tasks come from a recurring caseload so repeat encounters exist to measure over-updating on.",
        "parameters": {
            "episode length": "120 steps",
            "regimes": "abrupt (4 changes after a 20-step warmup); noisy stable (no changes, 15% of feedback inverted)",
            "memory arms": "notes; structured beliefs; world model (rule / confidence / since / invalid-if lines, prediction-error-gated revision)",
            "task stream": "a recurring caseload of one case per request type",
            "prediction elicitation": "the world-model agent ends every reply with EXPECT: success or failure; the wrong-prediction rate is measured from the logs",
        },
        "independent": ["how structured the agent's memory is: notes, beliefs, or an explicit world model", "how the world changes: really, or only in its feedback"],
        "dependent": ["over-update rate under lying feedback", "recovery latency under real change", "wrong-prediction rate", "final accuracy"],
        "drift": ["sudden real changes", "none, with misleading feedback"],
        "worlds": ["the intake desk (rule_world)"], "hypotheses": ["H006", "H001"],
        "predictions": [
            {"kind": "effect", "field": "overupdate_rate", "vary": "agent", "a": "llm_worldmodel", "b": "llm_notes",
             "direction": ">",
             "claim": "under lying feedback, an explicit world model abandons still-correct behavior less than free-form notes"},
            {"kind": "effect", "field": "recovery_lag", "vary": "agent", "a": "llm_worldmodel", "b": "llm_notes",
             "direction": ">",
             "claim": "under real change, the world model recovers faster: notes' recovery lag is higher"}]},
}


def regimes_needed(entry: dict) -> set | None:
    """The regimes this experiment's registered predictions actually reference — the
    minimal grid a hypothesis run has to play. None means the full grid is required
    (no prediction pins a regime). Predictions that vary the agent rather than the
    regime ride along on whatever cells the pinned predictions produce."""
    pinned = set()
    for p in entry.get("predictions", []):
        if p["kind"] == "reversal":
            pinned |= {p["a_regime"], p["b_regime"]}
        elif p["kind"] == "sign" and p.get("regime"):
            pinned.add(p["regime"])
        elif p.get("vary") == "regime":
            pinned |= {p["a"], p["b"]}
    return pinned or None


def hypothesis_status(hid: str) -> str:
    for p in HYPOTHESES_DIR.glob(f"{hid}-*.md"):
        for line in p.read_text().splitlines():
            if line.lower().startswith("status:"):
                return line.split(":", 1)[1].strip()
    return "no file"


def print_entry(name: str):
    e = REGISTRY[name]
    print(f"{name} v{e['version']} — {e['title']}\n")
    print(f"Hypothesis:  {e['hypothesis']}")
    print(f"Design:      {e.get('design', '')}")
    for k, v in e.get("parameters", {}).items():
        print(f"  {k}: {v}")
    print(f"Independent: {'; '.join(e['independent'])}")
    print(f"Dependent:   {'; '.join(e['dependent'])}")
    print(f"Drift:       {'; '.join(e['drift'])}   Worlds: {'; '.join(e['worlds'])}")
    print("Bears on:    " + ", ".join(f"{h} ({hypothesis_status(h)})" for h in e["hypotheses"]))
    print(f"\nResearch log: {HYPOTHESES_DIR}/")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        key = sys.argv[1] if sys.argv[1] in REGISTRY else f"exp{int(sys.argv[1].removeprefix('exp')):02d}"
        print_entry(key)
    else:
        for name, e in REGISTRY.items():
            print(f"{name} v{e['version']}  {e['title']:<46} {', '.join(e['hypotheses']):<12} {e['hypothesis'][:80]}")
