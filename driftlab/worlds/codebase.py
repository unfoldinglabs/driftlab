"""CodebaseWorld: a developer building on an evolving internal library while the
team's style guide drifts.

Each task asks the agent to implement one small function against a hidden
test suite. A third of the tickets build on the team's internal `util` module
(identifier, date, text and money helpers, provided in scope); with
`api_drift=True` a latent change may rename one of those helpers — the old
name keeps working but is deprecated, the onboarding docs in the system
prompt silently go stale, and review feedback is how the agent finds out.

Correctness is checked by running the tests; *conventions* are
checked by static inspection of the submitted code (type hints, docstrings,
no print, line length, using the provided `log()` helper). The convention set
changes every `session_length` tasks. That is the drift: the tests never
change, but "how we write code here" does, and the agent is only told what
the feedback level allows:

    tests_only   pass/fail counts only
    names        plus the names of violated conventions
    verbose      plus the failing test messages and a one-line convention rule

Reward: +1 if all tests pass and no convention is violated, +0.5 if tests
pass but conventions fail, 0 otherwise. Privileged log: active conventions,
per-test results, violations, session index.

Change capabilities: change_latent (the active convention set is swapped now),
change_surface (tickets are re-worded: a new ticket template, same requirements),
ask_confidence, consult (with `consult=True` the agent may reply `ASK: STYLE`;
the reviewer names one rule currently in force — partial information — and the
step's reward is forfeited). `session_length` schedules latent changes
automatically when a scenario does not.

Endogenous change: with `endogenous=True` the reviewer picks up the agent's
habits. If the last three submissions all satisfy a convention that is not
currently enforced (say, they all have docstrings), the reviewer starts
requiring it: the agent's own consistency raised the bar.

Submitted code is executed in a subprocess with a timeout. This is not a
security sandbox — run it only with models you trust, on a machine you
control.
"""

import ast
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field

import numpy as np

from .base import CONFIDENCE_SUFFIX, flavor_rng, person

# signal-free ticket chatter: realistic surface, no information about the style guide
TICKET_NOTES = ["Needed for the reporting pipeline.", "Blocks a downstream cleanup ticket.",
                "Nice-to-have for the ops dashboard.", "Follow-up from last week's incident review.",
                "Requested by the data team.", "Small one, should be quick."]
PRIORITIES = ["P2", "P3", "P3", "P4"]
from .policy import ConditionalPolicy

HELPERS = '''
_LOG = []
def log(msg):
    """Record an event. House convention: call once per function invocation."""
    _LOG.append(str(msg))
'''

# The team's internal library: a second module the tickets build on. Each concept has a
# sequence of names its function has carried over the library's history; under API drift
# the current name advances and the old one keeps working but is deprecated — the docs
# the agent saw at episode start silently go stale, exactly like a real internal repo.
API_CONCEPTS = {
    "identifier": {"names": ["fmt_id", "format_id", "format_identifier"], "params": "x",
                   "body": "return 'MD-' + str(int(x)).rjust(6, '0')",
                   "doc": "display reference for a numeric id, e.g. 7 -> 'MD-000007'"},
    "date": {"names": ["parse_dt", "parse_date", "read_date"], "params": "s",
             "body": "y, m, d = s.split('-'); return (int(y), int(m), int(d))",
             "doc": "split an ISO date string into an (y, m, d) tuple of ints"},
    "text": {"names": ["clean", "normalize_text", "canonical_text"], "params": "s",
             "body": "return ' '.join(s.split()).lower()",
             "doc": "collapse whitespace and lowercase"},
    "money": {"names": ["to_cents", "amount_cents", "as_cents"], "params": "amount",
              "body": "return int(round(float(amount) * 100))",
              "doc": "convert a currency amount to integer cents"},
}

TASKS = [
    ("slugify", "Return a URL slug: lowercase, spaces/underscores become '-', drop other non-alphanumerics, collapse repeated '-'.",
     ["assert slugify('Hello World') == 'hello-world'", "assert slugify('a__b  c!') == 'a-b-c'", "assert slugify('--x--') == 'x'"]),
    ("chunk", "Split list xs into consecutive chunks of size n (last chunk may be shorter).",
     ["assert chunk([1,2,3,4,5], 2) == [[1,2],[3,4],[5]]", "assert chunk([], 3) == []", "assert chunk([1,2], 5) == [[1,2]]"]),
    ("parse_kv", "Parse 'a=1;b=2' into a dict of strings; ignore empty segments and surrounding whitespace.",
     ["assert parse_kv('a=1;b=2') == {'a':'1','b':'2'}", "assert parse_kv(' x = y ;; ') == {'x':'y'}", "assert parse_kv('') == {}"]),
    ("running_mean", "Return the running (cumulative) mean of a list of numbers as a list of floats.",
     ["assert running_mean([2,4,6]) == [2.0,3.0,4.0]", "assert running_mean([]) == []"]),
    ("dedupe", "Remove duplicates from a list, keeping first occurrences in order.",
     ["assert dedupe([3,1,3,2,1]) == [3,1,2]", "assert dedupe([]) == []"]),
    ("word_freq", "Return a dict of lowercase word -> count for a string; words are alphabetic runs.",
     ["assert word_freq('A a b!') == {'a':2,'b':1}", "assert word_freq('') == {}"]),
    ("safe_div", "Divide a by b, returning default when b is zero.",
     ["assert safe_div(6, 3, default=0) == 2", "assert safe_div(1, 0, default=-1) == -1"]),
    ("flatten", "Flatten one level of nesting in a list of lists.",
     ["assert flatten([[1,2],[3],[]]) == [1,2,3]", "assert flatten([]) == []"]),
    # tickets that build on the team's internal library (module `util`, in scope)
    ("order_ref", "Return the display reference for order number n, using the team identifier helper.",
     ["assert order_ref(7) == 'MD-000007'", "assert order_ref(123456) == 'MD-123456'"]),
    ("order_year", "Return the year of an ISO order date string, using the team date helper.",
     ["assert order_year('2026-03-14') == 2026", "assert order_year('1999-12-31') == 1999"]),
    ("norm_names", "Normalize a list of customer names with the team text helper, keeping order.",
     ["assert norm_names(['  Ada  OKAFOR ', 'Bo  Li']) == ['ada okafor', 'bo li']", "assert norm_names([]) == []"]),
    ("invoice_cents", "Sum a list of currency amounts into integer cents, using the team money helper.",
     ["assert invoice_cents([1.10, 2.05]) == 315", "assert invoice_cents([]) == 0"]),
]

SIGNATURES = {"slugify": "slugify(s)", "chunk": "chunk(xs, n)", "parse_kv": "parse_kv(s)",
              "running_mean": "running_mean(xs)", "dedupe": "dedupe(xs)", "word_freq": "word_freq(s)",
              "safe_div": "safe_div(a, b, default)", "flatten": "flatten(xss)",
              "order_ref": "order_ref(n)", "order_year": "order_year(s)",
              "norm_names": "norm_names(names)", "invoice_cents": "invoice_cents(amounts)"}

CONVENTIONS = {
    "type_hints": "every parameter and the return value must have a type annotation",
    "docstring": "the function must have a docstring",
    "no_print": "do not call print()",
    "max_line_80": "no line longer than 80 characters",
    "must_log": "call log(...) exactly once at the start of the function body",
    "single_return": "the function must have exactly one return statement",
}


def check_conventions(src: str, fn_name: str, active: list[str], deprecated: dict | None = None) -> list[str]:
    """`deprecated` maps an obsolete util name to its current one; calling one is a violation."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return ["syntax_error"]
    if deprecated:
        called = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                called.add(getattr(n.func, "id", None) or getattr(n.func, "attr", None))
        called.discard(None)
        extra = ["deprecated_api"] if called & set(deprecated) else []
        return extra + check_conventions(src, fn_name, active)
    fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == fn_name), None)
    if fn is None:
        return ["function_missing"]
    bad = []
    if "type_hints" in active and (fn.returns is None or any(a.annotation is None for a in fn.args.args)):
        bad.append("type_hints")
    if "docstring" in active and ast.get_docstring(fn) is None:
        bad.append("docstring")
    if "no_print" in active and any(isinstance(n, ast.Call) and getattr(n.func, "id", "") == "print" for n in ast.walk(tree)):
        bad.append("no_print")
    if "max_line_80" in active and any(len(l) > 80 for l in src.splitlines()):
        bad.append("max_line_80")
    if "must_log" in active:
        body = fn.body[1:] if ast.get_docstring(fn) is not None else fn.body
        first = body[0] if body else None
        starts_with_log = (isinstance(first, ast.Expr) and isinstance(first.value, ast.Call)
                           and getattr(first.value.func, "id", "") == "log")
        n_logs = sum(1 for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "log")
        if not starts_with_log or n_logs != 1:
            bad.append("must_log")
    if "single_return" in active and sum(1 for n in ast.walk(fn) if isinstance(n, ast.Return)) != 1:
        bad.append("single_return")
    return bad


def run_tests(src: str, tests: list[str], timeout: float = 5.0, prelude: str = "") -> tuple[int, int, list[str]]:
    program = HELPERS + "\n" + prelude + "\n" + src + "\n\n_results = []\n"
    for i, t in enumerate(tests):
        program += (f"try:\n    {t}\n    _results.append(('ok', {i}, ''))\n"
                    f"except Exception as e:\n    _results.append(('fail', {i}, repr(e)[:120]))\n")
    program += "for r in _results:\n    print('RESULT', r[0], r[1], r[2])\n"
    try:
        out = subprocess.run([sys.executable, "-I", "-c", program], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 0, len(tests), ["timeout"]
    if out.returncode != 0 and "RESULT" not in out.stdout:
        return 0, len(tests), [out.stderr.strip().splitlines()[-1][:160] if out.stderr.strip() else "crashed"]
    passed, failures = 0, []
    for line in out.stdout.splitlines():
        if line.startswith("RESULT"):
            _, status, idx, *msg = line.split(" ", 3)
            if status == "ok":
                passed += 1
            else:
                failures.append(f"test {idx}: {msg[0] if msg else ''}")
    return passed, len(tests), failures


@dataclass
class CodebaseWorld:
    seed: int
    T: int = 32
    session_length: int = 8
    conventions_per_session: int = 2
    feedback: str = "names"     # tests_only | names | verbose
    api_drift: bool = False      # latent changes may also rename an internal-library function
    ask_confidence: bool = False
    consult: bool = False        # epistemic action: ASK: STYLE forfeits the step, the reviewer names one live rule
    endogenous: bool = False
    review_rounds: int = 1       # >1: a rejected ticket comes back with feedback; reward lands when it resolves
    round_penalty: float = 0.15  # each extra review round shaves this off the final reward
    conditional_conventions: int = 0  # hidden IF-THEN conventions over the ticket (rules as a program)
    task_fn: object = None       # (ticket_idx, world) -> index into TASKS; default: seeded permutation
    name: str = "codebase"

    system_prompt: str = field(init=False)
    changes: list = field(init=False, default_factory=list)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        # the internal library: per concept, the index of its current name; older names keep
        # working but are deprecated, so the docs below silently go stale under API drift
        self._api = {c: 0 for c in API_CONCEPTS}
        api_doc = "; ".join(f"{self._api_name(c)}({API_CONCEPTS[c]['params']}) — {API_CONCEPTS[c]['doc']}"
                            for c in API_CONCEPTS)
        self.system_prompt = ("You are a developer on the platform team at Meridian. Tickets ask you to implement small "
                              "functions against the team's internal library; CI runs the tests and a reviewer checks the "
                              "code against the team's style guide, which is not written down anywhere you can see. "
                              "A helper `log(msg)` is available in scope, along with the team utilities (as of your "
                              f"onboarding): {api_doc}. "
                              "Reply with the complete function source inside one code block:\n```python\n<your code>\n```")
        if self.consult:
            self.system_prompt += ("\nIf you are unsure about the house style, you may ask instead of submitting: "
                                   "reply with just the line\nASK: STYLE\nThe reviewer will tell you one rule currently "
                                   "in force; the ticket waits and you get no credit for the step.")
        self.system_prompt += CONFIDENCE_SUFFIX if self.ask_confidence else ""
        self._ask_i = 0
        self._asked = False
        self.n_asks = 0
        self.order = [int(i) for i in self.rng.permutation(len(TASKS))]
        self.sessions = self._make_sessions()
        self.active: list[str] = list(self.sessions[0])
        self._session_seen = 0
        self.template = 0
        self.results: list[dict] = []
        self._new: list = []
        self._last_conf = None
        self._changed_at: int | None = None
        self._ticket = 0            # tickets started; the task id
        self._round = 0             # review rounds used on the open ticket
        self._last_fb: str | None = None
        self._task_info = {"task_id": 0, "task_step": 0, "task_done": False}
        self.policy = ConditionalPolicy(self.seed, self.conditional_conventions, predicates={
            "multi_param": lambda name: SIGNATURES[name].count(",") >= 1,
            "string_task": lambda name: name in ("slugify", "parse_kv", "word_freq"),
            "collection_task": lambda name: name in ("chunk", "dedupe", "flatten", "running_mean"),
        }, effects=list(CONVENTIONS)) if self.conditional_conventions else None

    def _record(self, t, kind, desc, **extra):
        ch = {"t": t, "kind": kind, "desc": desc, "affected": [], **extra}
        self.changes.append(ch)
        self._new.append(ch)
        if kind in ("latent", "endogenous"):
            self._changed_at = t
        return desc

    # ---- the internal library --------------------------------------------------------
    def _api_name(self, concept: str) -> str:
        return API_CONCEPTS[concept]["names"][self._api[concept]]

    def _deprecated(self) -> dict:
        """obsolete util name -> current name, across every concept's history."""
        out = {}
        for c, spec in API_CONCEPTS.items():
            for old in spec["names"][: self._api[c]]:
                out[old] = self._api_name(c)
        return out

    def _util_source(self) -> str:
        lines = []
        for c, spec in API_CONCEPTS.items():
            cur = self._api_name(c)
            lines.append(f"def {cur}({spec['params']}):\n    {spec['body']}")
            for old in spec["names"][: self._api[c]]:  # deprecated names keep working
                lines.append(f"{old} = {cur}")
        return "\n".join(lines)

    # ---- change capabilities ---------------------------------------------------------
    def change_latent(self, t: int) -> str:
        if self.api_drift:
            movable = [c for c in API_CONCEPTS if self._api[c] + 1 < len(API_CONCEPTS[c]["names"])]
            if movable and self.rng.random() < 0.5:
                c = movable[int(self.rng.integers(len(movable)))]
                old = self._api_name(c)
                self._api[c] += 1
                return self._record(t, "latent", f"internal library updated: {old} is deprecated, "
                                                 f"use {self._api_name(c)}", api={k: self._api[k] for k in self._api})
        if self.policy and self.rng.random() < 0.4:
            return self._record(t, "latent", f"style guide revised: {self.policy.mutate()}")
        names = [c for c in CONVENTIONS if c not in self.active]
        new = [str(c) for c in self.rng.choice(names, size=min(self.conventions_per_session, len(names)), replace=False)]
        old, self.active = self.active, new
        return self._record(t, "latent", f"style guide revised: now {new} (was {old})", conventions=new)

    def change_surface(self, t: int) -> str:
        self.template = (self.template + 1) % 3
        return self._record(t, "surface", "ticket template re-worded (requirements unchanged)")

    def _endogenous_check(self, t: int, src: str, fn_name: str):
        if not self.endogenous:
            return
        self._habits = getattr(self, "_habits", [])
        positive = ("docstring", "type_hints", "must_log", "single_return")   # habits a reviewer would notice and adopt
        satisfied = {c for c in positive if c not in self.active and not check_conventions(src, fn_name, [c])}
        self._habits.append(satisfied)
        if len(self._habits) >= 3:
            common = set.intersection(*self._habits[-3:]) - set(self.active)
            if common:
                c = sorted(common)[0]
                self.active = self.active + [c]
                self._habits = []
                self._record(t, "endogenous", f"reviewer now expects '{c}' after seeing it in your last three submissions",
                             conventions=list(self.active))

    def _make_sessions(self):
        n_sessions = -(-self.T // self.session_length)
        names = list(CONVENTIONS)
        out = []
        for _ in range(n_sessions):
            out.append([str(c) for c in self.rng.choice(names, size=self.conventions_per_session, replace=False)])
        return out

    def _task(self, _t: int = 0):
        idx = self.task_fn(self._ticket, self) if self.task_fn else self.order[self._ticket % len(TASKS)]
        return TASKS[int(idx) % len(TASKS)]

    def _active(self, t: int) -> list[str]:
        name = self._task()[0]
        extra = [c for c in self.policy.active(name) if c not in self.active] if self.policy else []
        return self.active + extra

    def _maybe_new_session(self, t: int):
        if t and t % self.session_length == 0:
            self.change_latent(t)

    # ---- World protocol -----------------------------------------------------
    def observe(self, t: int) -> str:
        self._maybe_new_session(t)
        name, spec, _ = self._task()
        templates = [f"Ticket #{1000 + self._ticket}: implement `{SIGNATURES[name]}`. {spec}",
                     f"[TASK-{1000 + self._ticket}] New helper needed: `{SIGNATURES[name]}`\nAcceptance: {spec}",
                     f"Hey, can you add `{SIGNATURES[name]}`? Should {spec[0].lower() + spec[1:]}"]
        rng = flavor_rng(self.seed, t)
        text = (f"(reported by {person(rng)} · {rng.choice(PRIORITIES)} · \"{rng.choice(TICKET_NOTES)}\")\n"
                + templates[self.template])
        if self._round and self._last_fb:
            text += (f"\n\nReview round {self._round + 1} of {self.review_rounds} for this ticket. "
                     f"Your last submission was not merged:\n{self._last_fb}")
        return text

    def parse(self, text: str):
        c = re.search(r"CONFIDENCE:\s*(\d{1,3})", text)
        self._last_conf = min(float(c.group(1)), 100.0) / 100 if c else None
        m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
        src = m.group(1) if m else (text if "def " in text else None)
        if src is None and self.consult and re.search(r"^ASK:", text, re.M):
            return "ASK"
        return textwrap.dedent(src).strip() if src else None

    def default_action(self, rng):
        return "def _missing():\n    pass"

    def describe_action(self, action) -> str:
        return action.replace("\n", "\\n")[:200]

    def act(self, t: int, action: str) -> tuple[float, str]:
        name, _, tests = self._task()
        task_id, task_step = self._ticket, self._round
        active = self._active(t)
        deprecated = self._deprecated()
        if action == "ASK":
            self._asked = True
            self.n_asks += 1
            rule = active[self._ask_i % len(active)]
            self._ask_i += 1
            self._task_info = {"task_id": task_id, "task_step": task_step, "task_done": False}
            self.results.append({"t": t, "task": name, "passed": 0, "total": 0, "violations": [],
                                 "active": active, "reward": 0.0})
            return 0.0, (f"Reviewer: one thing we do enforce right now — {CONVENTIONS.get(rule, rule)}. "
                         "The ticket is still waiting; no credit for the step.")
        passed, total, failures = run_tests(action, tests, prelude=self._util_source())
        violations = check_conventions(action, name, active, deprecated)
        self._endogenous_check(t, action, name)
        tests_ok, conv_ok = passed == total, not violations
        base = 1.0 if tests_ok and conv_ok else (0.5 if tests_ok else 0.0)
        fb = f"Tests: {passed}/{total} passed."
        if self.feedback in ("names", "verbose") and violations:
            fb += " Convention violations: " + ", ".join(violations) + "."
        elif violations and self.feedback == "tests_only":
            fb += " Code review: not accepted (house conventions)."
        if self.feedback == "verbose":
            if failures:
                fb += "\nFailures:\n" + "\n".join(f"- {f}" for f in failures)
            for v in violations:
                if v in CONVENTIONS:
                    fb += f"\nRule '{v}': {CONVENTIONS[v]}."
                elif v == "deprecated_api":
                    used = [f"{old} (use {new})" for old, new in deprecated.items()]
                    fb += f"\nRule 'deprecated_api': deprecated helpers: {', '.join(used)}."
        done = (tests_ok and conv_ok) or (task_step + 1 >= self.review_rounds)
        if done:
            reward = base * max(0.0, 1.0 - self.round_penalty * task_step) if self.review_rounds > 1 else base
            self._ticket += 1
            self._round, self._last_fb = 0, None
        else:  # the ticket comes back for another round; the reward lands when it resolves
            reward = 0.0
            self._round, self._last_fb = task_step + 1, fb
            fb += "\nNot merged yet; the reviewer sent it back — revise and resubmit."
        self._task_info = {"task_id": task_id, "task_step": task_step, "task_done": done}
        self.results.append({"t": t, "task": name, "passed": passed, "total": total, "violations": violations,
                             "active": active, "reward": reward})
        return reward, fb

    def privileged(self, t: int) -> dict:
        r = self.results[-1]
        new, self._new = self._new, []
        asked, self._asked = self._asked, False
        return {"task": r["task"], "passed": r["passed"], "total": r["total"], "violations": r["violations"],
                "api_versions": dict(self._api),
                "active_conventions": list(r["active"]), "session": t // self.session_length, "confidence": self._last_conf,
                "affected_now": self._changed_at is not None and t - self._changed_at <= 3, "asked": asked,
                **self._task_info, "changes": new,
                "conditionals": self.policy.describe() if self.policy else []}

    def summary(self) -> dict:
        return {"changes": self.changes, "n_endogenous": sum(1 for c in self.changes if c["kind"] == "endogenous"),
                "n_asks": self.n_asks,
                "mean_reward": float(np.mean([r["reward"] for r in self.results])) if self.results else None}
