"""FormWorld: data entry into a company order form whose validation drifts.

Each task gives the agent a source record and the form's visible field names.
By default the record arrives as a prose intake note (a typed-up phone order,
a voicemail transcription) in which every value appears verbatim and quoted,
so extraction reads like real work while the truth stays exactly recoverable;
`record_style="json"` gives the raw record instead. The agent submits a JSON
object. Hidden validation decides which fields are
required, what format each expects (date style, country code, integer ranges,
enum values) and how record keys map onto form fields. The agent only learns
these from the error list, whose richness is a parameter:

    terse     "3 field(s) invalid."
    fields    names the invalid fields
    verbose   names the field and states the rule it broke

Change capabilities: change_latent (a schema mutation: rename, enum, required,
date format, country style, quantity cap), change_surface (the source record's
key names are re-labeled; the form and its rules do not change — and with
`surface_strength="structural"` the record's rendering also cycles between
JSON, key-value lines and a TSV table), introduce_novelty (a new optional
field with its own rule appears), probe, ask_confidence. `version_every`
schedules latent changes automatically when a scenario does not.

Endogenous change: with `endogenous=True`, when the agent submits the same
unknown field name three times, IT adds it as an accepted alias of the closest
form field ("users kept typing it, so we allowed it"): a schema change the
agent caused, and one that makes its own habit correct.
"""

import difflib
import json
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .base import CONFIDENCE_SUFFIX, flavor_rng, person

# signal-free intake chatter: realistic surface, no information about the schema
INTAKE_NOTES = ["Customer called to confirm the order details.", "Came in through the partner portal overnight.",
                "Marked ordinary priority by the intake team.", "Original was handwritten; transcribed by reception.",
                "Duplicate check already done, this one is new.", "Customer asked for a confirmation email."]
from .policy import ConditionalPolicy

COUNTRIES = {"France": "FR", "Germany": "DE", "Spain": "ES", "Italy": "IT", "Poland": "PL"}
PRIORITIES = [["low", "normal", "high"], ["P3", "P2", "P1"], ["standard", "rush"]]
DATE_FORMATS = ["YYYY-MM-DD", "DD/MM/YYYY", "MM-DD-YYYY"]
RENAMES = {"qty": ["qty", "quantity", "units"], "cust": ["cust", "customer_name", "buyer"],
           "email": ["email", "contact_email"], "ship_country": ["ship_country", "country", "destination"],
           "order_date": ["order_date", "date", "placed_on"], "priority": ["priority", "urgency"]}
RECORD_LABELS = [
    {"customer": "customer", "email": "email", "quantity": "quantity", "shipping_country": "shipping_country",
     "order_date": "order_date", "priority_level": "priority_level"},
    {"customer": "client", "email": "contact", "quantity": "units_ordered", "shipping_country": "deliver_to",
     "order_date": "placed", "priority_level": "urgency_code"},
    {"customer": "account_holder", "email": "e_mail", "quantity": "count", "shipping_country": "ship_to_country",
     "order_date": "ordered_on", "priority_level": "prio"},
]
SYSTEM = ("You are a data-entry specialist at Meridian Logistics. You receive order records and must enter each "
          "one into the company's order form. The system rejects invalid submissions and reports errors; the form "
          "has no manual. Reply with the completed form as a single JSON object on its own line, prefixed by FORM:\n"
          'FORM: {"field": "value", ...}')


def fmt_date(iso: str, style: str) -> str:
    y, m, d = iso.split("-")
    return {"YYYY-MM-DD": iso, "DD/MM/YYYY": f"{d}/{m}/{y}", "MM-DD-YYYY": f"{m}-{d}-{y}"}[style]


@dataclass
class FormWorld:
    seed: int
    T: int = 90
    version_every: int = 15
    feedback: str = "fields"
    max_attempts: int = 3
    ask_confidence: bool = False
    endogenous: bool = False
    conditional_rules: int = 0       # hidden IF-THEN requirements over the source record (rules as a program)
    task_fn: object = None           # (task_idx, world) -> source record; default: seeded random (streams plug in here)
    attempt_penalty: float = 0.0     # long-horizon grading: success reward = 1 - penalty * failed attempts
    surface_strength: str = "label"  # label: columns renamed | structural: the record's format also changes
    record_style: str = "document"   # document: a prose intake note | json: the raw record
    name: str = "form_filler"

    system_prompt: str = field(init=False)
    schema: dict = field(init=False)
    changes: list = field(init=False, default_factory=list)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.system_prompt = SYSTEM + (CONFIDENCE_SUFFIX if self.ask_confidence else "")
        self.schema = {"names": {k: v[0] for k, v in RENAMES.items()}, "required": {"qty", "cust", "email", "ship_country"},
                       "date_format": DATE_FORMATS[0], "country_style": "code", "priority_values": PRIORITIES[0],
                       "qty_max": 100, "aliases": {}, "extra": None}
        self.version = 1
        self.record_labels = 0
        self.task_idx = 0
        self.attempt = 0
        self.last_errors: list[str] = []
        self.successes = 0
        self._unknown = Counter()
        self._new: list = []
        self._last_conf = None
        self._last_truth: list = []
        self._unresolved_since: int | None = None
        self._task_info = {"task_id": 0, "task_step": 0, "task_done": False}
        self.policy = ConditionalPolicy(self.seed, self.conditional_rules, predicates={
            "large_order": lambda rec: rec["quantity"] > 60,
            "rush_order": lambda rec: rec["priority_level"] == 2,
            "export_order": lambda rec: rec["shipping_country"] in ("Germany", "France"),
            "late_month": lambda rec: int(rec["order_date"].split("-")[2]) > 20,
        }, effects=["order_date", "priority"]) if self.conditional_rules else None
        self.task = self.task_fn(0, self) if self.task_fn else self._new_task()

    def _record(self, t, kind, desc, **extra):
        ch = {"t": t, "kind": kind, "desc": desc, "affected": [], **extra}
        self.changes.append(ch)
        self._new.append(ch)
        if kind in ("latent", "novelty", "endogenous"):
            self._unresolved_since = t
        return desc

    # ---- change capabilities ----------------------------------------------------
    def change_latent(self, t: int) -> str:
        kinds = ["rename", "enum", "required", "date", "country", "qty_max"] + (["conditional"] if self.policy else [])
        kind = str(self.rng.choice(kinds))
        s = self.schema
        if kind == "conditional":
            desc = self.policy.mutate()
        elif kind == "rename":
            k = str(self.rng.choice(list(RENAMES)))
            s["names"][k] = str(self.rng.choice([n for n in RENAMES[k] if n != s["names"][k]]))
            desc = f"form field renamed: {k} is now '{s['names'][k]}'"
        elif kind == "enum":
            s["priority_values"] = PRIORITIES[int(self.rng.integers(len(PRIORITIES)))]
            desc = f"priority values changed to {s['priority_values']}"
        elif kind == "required":
            opt = [k for k in RENAMES if k not in s["required"]]
            if opt:
                k = str(self.rng.choice(opt)); s["required"].add(k); desc = f"field now required: {k}"
            else:
                s["required"].discard("priority"); desc = "priority no longer required"
        elif kind == "date":
            s["date_format"] = str(self.rng.choice([f for f in DATE_FORMATS if f != s["date_format"]]))
            desc = f"date format changed to {s['date_format']}"
        elif kind == "country":
            s["country_style"] = "name" if s["country_style"] == "code" else "code"
            desc = f"country must now be given as {s['country_style']}"
        else:
            s["qty_max"] = int(self.rng.choice([20, 50, 100, 500])); desc = f"quantity cap changed to {s['qty_max']}"
        self.version += 1
        return self._record(t, "latent", f"form v{self.version}: {desc}", version=self.version)

    def change_surface(self, t: int) -> str:
        self.record_labels = (self.record_labels + 1) % len(RECORD_LABELS)
        if self.surface_strength == "structural":
            self.record_format = (getattr(self, "record_format", 0) + 1) % 3
            return self._record(t, "surface", "source records arrive in a different format and with different "
                                              "column names (the form is unchanged)")
        return self._record(t, "surface", "source records now use different column names (form unchanged)")

    OPENERS = ["Typed up from a phone order taken by the front desk:",
               "Transcribed from the customer's voicemail:",
               "From an emailed order, pasted below:"]
    CONNECTORS = ['the {k} is "{v}"', '{k} given as "{v}"', 'their {k} reads "{v}"']

    def _record_text(self) -> str:
        rec = self._visible_record()
        fmt = getattr(self, "record_format", 0)
        if fmt == 1:
            return "\n".join(f"{k}: {v}" for k, v in rec.items())
        if fmt == 2:
            return "\t".join(rec) + "\n" + "\t".join(str(v) for v in rec.values())
        if self.record_style == "document":
            # a prose intake note; every value appears verbatim and quoted, so the truth
            # stays exactly recoverable while the record reads like a real document
            rng = flavor_rng(self.seed, 60_000 + self.task_idx)
            parts = [str(rng.choice(self.CONNECTORS)).format(k=k, v=v) for k, v in rec.items()]
            return str(rng.choice(self.OPENERS)) + " " + "; ".join(parts) + "."
        return json.dumps(rec, indent=2)

    def introduce_novelty(self, t: int) -> str:
        if self.schema["extra"]:
            return ""
        self.schema["extra"] = {"name": "gift_wrap", "values": ["yes", "no"]}
        return self._record(t, "novelty", "new optional form field appears: gift_wrap (yes/no)")

    def _endogenous_check(self, t: int, submission: dict):
        if not self.endogenous:
            return
        for k in submission:
            if k not in self.schema["names"].values() and k not in self.schema["aliases"]:
                self._unknown[k] += 1
                if self._unknown[k] >= 3:
                    synonyms = {syn: canon for canon, syns in RENAMES.items() for syn in syns}
                    match = difflib.get_close_matches(k, list(synonyms), n=1, cutoff=0.6)
                    if match:
                        canon = synonyms[match[0]]
                        self.schema["aliases"][k] = canon
                        self._record(t, "endogenous", f"IT added '{k}' as an accepted alias for '{self.schema['names'][canon]}' "
                                                      "after repeated submissions", alias=k)

    # ---- tasks -----------------------------------------------------------------------
    def _new_task(self) -> dict:
        r = self.rng
        return {"customer": f"{r.choice(['Ada', 'Bo', 'Cy', 'Dee', 'Eli'])} {r.choice(['Marsh', 'Okafor', 'Reyes', 'Sato'])}",
                "email": f"user{r.integers(100, 999)}@example.com", "quantity": int(r.integers(1, 120)),
                "shipping_country": str(r.choice(list(COUNTRIES))), "order_date": f"2026-{r.integers(1, 13):02d}-{r.integers(1, 29):02d}",
                "priority_level": int(r.integers(3))}

    def _visible_record(self) -> dict:
        labels = RECORD_LABELS[self.record_labels]
        return {labels[k]: v for k, v in self.task.items()}

    def _expected(self) -> dict:
        s, rec, n = self.schema, self.task, self.schema["names"]
        return {n["qty"]: rec["quantity"], n["cust"]: rec["customer"], n["email"]: rec["email"],
                n["ship_country"]: COUNTRIES[rec["shipping_country"]] if s["country_style"] == "code" else rec["shipping_country"],
                n["order_date"]: fmt_date(rec["order_date"], s["date_format"]),
                n["priority"]: s["priority_values"][min(rec["priority_level"], len(s["priority_values"]) - 1)]}

    def _validate(self, submission: dict) -> list[tuple[str, str]]:
        s, n, exp = self.schema, self.schema["names"], self._expected()
        required = set(s["required"]) | (set(self.policy.active(self.task)) if self.policy else set())
        sub = {}
        for k, v in submission.items():
            canon = s["aliases"].get(k)
            sub[n[canon] if canon else k] = v
        errors = []
        for canon, visible in n.items():
            val = sub.get(visible)
            if val is None or val == "":
                if canon in required:
                    errors.append((visible, "required field missing"))
                continue
            if canon == "qty":
                try:
                    qv = int(val)
                except (TypeError, ValueError):
                    errors.append((visible, "must be an integer")); continue
                if qv != exp[visible]:
                    errors.append((visible, "does not match the source record"))
                elif qv > s["qty_max"]:
                    errors.append((visible, f"exceeds maximum of {s['qty_max']}"))
            elif canon == "priority":
                if str(val) not in s["priority_values"]:
                    errors.append((visible, f"must be one of {s['priority_values']}"))
            elif canon == "order_date":
                if str(val) != exp[visible]:
                    errors.append((visible, f"must use format {s['date_format']}"))
            elif canon == "ship_country":
                if str(val) != exp[visible]:
                    errors.append((visible, "must be a 2-letter ISO code" if s["country_style"] == "code" else "must be the full country name"))
            elif str(val).strip() != str(exp[visible]):
                errors.append((visible, "does not match the source record"))
        extra = s["extra"]
        for k, v in sub.items():
            if k in n.values():
                continue
            if extra and k == extra["name"]:
                if str(v) not in extra["values"]:
                    errors.append((k, f"must be one of {extra['values']}"))
                continue
            errors.append((k, "unknown field"))
        return errors

    # ---- World protocol ---------------------------------------------------------------
    def _fields_text(self) -> str:
        names = sorted(self.schema["names"].values())
        if self.schema["extra"]:
            names.append(self.schema["extra"]["name"])
        return ", ".join(names)

    def observe(self, t: int) -> str:
        prev = ""
        if self.attempt and self.last_errors:
            prev = "\n\nYour previous submission for this record was rejected:\n" + "\n".join(self.last_errors)
        rng = flavor_rng(self.seed, t)
        intro = (f"Intake batch ORD-{5100 + t}, forwarded by {person(rng)} at the front desk. "
                 f"\"{rng.choice(INTAKE_NOTES)}\"\n")
        return (f"{intro}Order form (version {self.version}). Fields: {self._fields_text()}.\n"
                f"Attempt {self.attempt + 1} of {self.max_attempts}.\n\nSource record:\n"
                f"{self._record_text()}{prev}")

    def probe(self, task: dict | None = None) -> str:
        rec = task or self._new_task()
        labels = RECORD_LABELS[self.record_labels]
        return ("Quick check from your team lead (nothing will be submitted; just show how you would fill it in):\n"
                f"Fields: {self._fields_text()}.\nSource record:\n{json.dumps({labels[k]: v for k, v in rec.items()}, indent=2)}")

    def parse(self, text: str):
        c = re.search(r"CONFIDENCE:\s*(\d{1,3})", text)
        self._last_conf = min(float(c.group(1)), 100.0) / 100 if c else None
        m = re.search(r"FORM:\s*(\{.*?\})\s*(?:\n|$)", text, re.S) or re.search(r"(\{.*\})", text, re.S)
        if not m:
            return None
        try:
            obj = json.loads(m.group(1))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None

    def default_action(self, rng):
        return {}

    def describe_action(self, action) -> str:
        return json.dumps(action, sort_keys=True)[:200]

    def act(self, t: int, action: dict) -> tuple[float, str]:
        task_id, task_step = self.task_idx, self.attempt
        self._endogenous_check(t, action)
        errors = self._validate(action)
        self._last_truth = errors
        if not errors:
            reward, fb = max(0.2, 1.0 - self.attempt_penalty * task_step), "Accepted."
            self.successes += 1
            self._unresolved_since = None
            self._advance(t)
        else:
            reward = -0.2
            if self.feedback == "terse":
                fb = f"Rejected: {len(errors)} field(s) invalid."
            elif self.feedback == "fields":
                fb = "Rejected. Invalid fields: " + ", ".join(sorted({e[0] for e in errors}))
            else:
                fb = "Rejected:\n" + "\n".join(f"- {f}: {why}" for f, why in errors)
            self.last_errors = fb.splitlines()[1:] if "\n" in fb else [fb]
            self.attempt += 1
            if self.attempt >= self.max_attempts:
                fb += "\nNo attempts left; the record goes to manual review and you get the next one."
                self._advance(t)
        self._task_info = {"task_id": task_id, "task_step": task_step, "task_done": self.task_idx != task_id}
        return reward, fb

    def _advance(self, t: int):
        self.task_idx += 1
        self.attempt = 0
        self.last_errors = []
        if self.task_idx % self.version_every == 0:
            self.change_latent(t)
        self.task = self.task_fn(self.task_idx, self) if self.task_fn else self._new_task()

    def privileged(self, t: int) -> dict:
        new, self._new = self._new, []
        return {"version": self.version, "task_idx": self.task_idx, "attempt": self.attempt, "confidence": self._last_conf,
                "error_truth": [list(e) for e in self._last_truth], "affected_now": self._unresolved_since is not None,
                **self._task_info, "changes": new, "schema": json.dumps(self.schema, default=list, sort_keys=True),
                "conditionals": self.policy.describe() if self.policy else []}

    def summary(self) -> dict:
        return {"successes": self.successes, "tasks_seen": self.task_idx, "changes": self.changes, "final_version": self.version,
                "n_endogenous": sum(1 for c in self.changes if c["kind"] == "endogenous")}
