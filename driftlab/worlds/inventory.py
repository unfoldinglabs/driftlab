"""InventoryWorld: a buyer restocking one product line under shifting demand.

Each day the agent sees stock on hand, orders in transit, and yesterday's
sales (demand censored by what was in stock). It orders a quantity that
arrives after `lead_time` days. Costs: holding per unit-day, a stockout
penalty per lost sale, a fixed cost per non-zero order. Demand is Poisson with
a rate that follows a season schedule (abrupt), drifts (gradual), is flat, or\nfollows a retail-shaped series (regime "retail": a weekday cycle over a moving\nbase level with occasional promotion spikes).

Change capabilities: change_latent (the demand rate jumps to a new level from
today on), change_surface (the daily report is re-worded: units become
"pcs", columns reordered; nothing about demand changes), probe (a hypothetical
"what would you order if..." from the manager), ask_confidence.

Endogenous change: with `endogenous=True`, an order above `big_order` units
strains the supplier: lead time rises by one day for `strain_days` days. The
agent's own ordering pattern changes the dynamics it must plan around.

The reference policy `BaseStockOracle` uses the TRUE current rate (a regret
floor); `MovingAverageAgent` is an achievable baseline.
"""

import re
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import poisson

from .base import CONFIDENCE_SUFFIX, flavor_rng

# signal-free warehouse chatter: realistic surface, no information about demand
OPS_NOTES = ["The warehouse team rotated the racking over the weekend.", "Forklift certification day on Thursday.",
             "The morning count matched the system, no adjustments.", "Carrier pickup moved to 3pm this week.",
             "New temp started in receiving today.", "Cycle count scheduled for the end of the month."]

SYSTEM_TMPL = ("You are the buyer for one product line at Meridian Retail. Each morning you see stock on hand, orders "
               "in transit and yesterday's sales, and place today's order (it arrives in {lead} days unless the report "
               "says otherwise). Your job is to keep total cost down. Costs: {h} per unit held per day, {s} per unit of "
               "demand you could not serve, {o} fixed cost per non-zero order. End your reply with a line:\nORDER: <integer>")
UNIT_WORDS = ["units", "pcs", "items"]


@dataclass
class InventoryWorld:
    seed: int
    T: int = 90
    regime: str = "seasonal"      # seasonal (abrupt) | drifting (gradual) | stationary | retail (weekday cycle + promos)
    rates: tuple = (6.0, 14.0, 3.0)
    season_length: int = 30
    lead_time: int = 2
    holding: float = 0.1
    stockout: float = 1.0
    order_cost: float = 0.5
    feedback: str = "standard"
    ask_confidence: bool = False
    endogenous: bool = False
    big_order: int = 40
    strain_days: int = 7
    name: str = "inventory"

    system_prompt: str = field(init=False)
    changes: list = field(init=False, default_factory=list)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.system_prompt = SYSTEM_TMPL.format(lead=self.lead_time, h=self.holding, s=self.stockout, o=self.order_cost)
        self.system_prompt += CONFIDENCE_SUFFIX if self.ask_confidence else ""
        self.stock = int(self.rates[0] * (self.lead_time + 1))
        self.base_lead = self.lead_time
        self.pipeline = [0] * self.lead_time
        self.true_rates = self._make_rates()
        self._retail_level = float(self.rates[0])   # retail: the base level, before the weekly cycle
        self.history: list[dict] = []
        self.total_cost = 0.0
        self.unit_word = 0
        self.strain_until = -1
        self._new: list = []
        self._last_conf = None
        self._changed_at: int | None = None

    WEEK = np.array([1.0, 0.9, 0.85, 0.95, 1.15, 1.45, 1.3])  # Mon..Sun retail cycle

    def _make_rates(self) -> np.ndarray:
        if self.regime == "stationary":
            return np.full(self.T, float(self.rates[0]))
        if self.regime == "seasonal":
            return np.array([self.rates[(t // self.season_length) % len(self.rates)] for t in range(self.T)], dtype=float)
        if self.regime == "retail":
            # a realistic retail shape: weekday cycle over a slowly moving base level,
            # plus occasional 3-day promotion spikes; fully determined by the seed
            base = np.interp(np.arange(self.T), np.linspace(0, self.T - 1, len(self.rates)), list(self.rates))
            rates = base * self.WEEK[np.arange(self.T) % 7]
            promo = np.random.default_rng(self.seed + 30_000)
            n_promos = max(1, self.T // 45)
            for start in promo.choice(np.arange(10, max(11, self.T - 4)), size=n_promos, replace=False):
                rates[start:start + 3] *= 1.8
            return rates
        knots = np.linspace(0, self.T - 1, len(self.rates) + 1)
        return np.interp(np.arange(self.T), knots, list(self.rates) + [self.rates[0]])

    def _record(self, t, kind, desc, **extra):
        ch = {"t": t, "kind": kind, "desc": desc, "affected": [], **extra}
        self.changes.append(ch)
        self._new.append(ch)
        if kind in ("latent", "endogenous"):
            self._changed_at = t
        return desc

    # ---- change capabilities ----------------------------------------------------------
    def change_latent(self, t: int) -> str:
        if self.regime == "retail":
            # the jump moves the BASE level (not the weekday-modulated rate), so the weekly shape survives
            cur = self._retail_level
            new = float(self.rng.choice([r for r in self.rates if abs(r - cur) > 1e-9] or [cur * 2]))
            self.true_rates[t:] *= new / cur
            self._retail_level = new
            return self._record(t, "latent", f"demand level jumps {cur:.1f} -> {new:.1f} per day (before the weekly cycle)", rate=new)
        cur = float(self.true_rates[t]) if t < self.T else float(self.rates[0])
        new = float(self.rng.choice([r for r in self.rates if abs(r - cur) > 1e-9] or [cur * 2]))
        self.true_rates[t:] = new
        return self._record(t, "latent", f"demand rate jumps {cur:.1f} -> {new:.1f} per day", rate=new)

    def change_surface(self, t: int) -> str:
        self.unit_word = (self.unit_word + 1) % len(UNIT_WORDS)
        return self._record(t, "surface", f"daily report re-worded (quantities now shown as '{UNIT_WORDS[self.unit_word]}')")

    def probe(self, task: dict | None = None) -> str:
        stock = (task or {}).get("stock", self.stock)
        return (f"Your manager asks, hypothetically: if you had {stock} {UNIT_WORDS[self.unit_word]} on hand this morning "
                f"and nothing in transit, how many would you order today? (Nothing will be placed.)")

    def _endogenous_check(self, t: int, order: int):
        if self.endogenous and order >= self.big_order and t > self.strain_until:
            self.strain_until = t + self.strain_days
            self.pipeline.append(0)   # one extra day of transit for everything from now
            self._record(t, "endogenous", f"supplier strained by a {order}-unit order: lead time is {self.base_lead + 1} days "
                                          f"for the next {self.strain_days} days")

    def _relax_strain(self, t: int):
        if self.strain_until >= 0 and t > self.strain_until and len(self.pipeline) > self.base_lead:
            # merge the extra transit day back
            extra = self.pipeline.pop()
            self.pipeline[-1] += extra
            self.strain_until = -1
            self._record(t, "surface", f"supplier back to normal: lead time {self.base_lead} days")

    # ---- World protocol ------------------------------------------------------------------
    def observe(self, t: int) -> str:
        u = UNIT_WORDS[self.unit_word]
        last = self.history[-1] if self.history else None
        sales = "n/a" if not last else (f"{last['sales']} {u} (sold out)" if last["lost"] > 0 else f"{last['sales']} {u}")
        recent = ", ".join(str(h["sales"]) for h in self.history[-14:]) or "none yet"
        lead_note = f" Current supplier lead time: {len(self.pipeline)} days." if len(self.pipeline) != self.base_lead else ""
        note = flavor_rng(self.seed, t).choice(OPS_NOTES)
        weekday = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][t % 7]
        return (f"Day {t + 1}, {weekday}. Stock on hand: {self.stock} {u}. Arriving over the next {len(self.pipeline)} days: {self.pipeline}."
                f"{lead_note}\nYesterday's sales: {sales}. Sales over the last 14 days: {recent}.\n"
                f"Ops note: {note}\nHow many {u} do you order today?")

    def parse(self, text: str):
        c = re.search(r"CONFIDENCE:\s*(\d{1,3})", text)
        self._last_conf = min(float(c.group(1)), 100.0) / 100 if c else None
        m = re.search(r"ORDER:\s*(\d+)", text)
        return int(m.group(1)) if m else None

    def default_action(self, rng):
        return int(rng.integers(0, 15))

    def describe_action(self, action) -> str:
        return f"order {action}"

    def act(self, t: int, action: int) -> tuple[float, str]:
        order = max(0, int(action))
        self._relax_strain(t)
        self.stock += self.pipeline.pop(0)
        self.pipeline.append(order)
        self._endogenous_check(t, order)
        demand = int(self.rng.poisson(self.true_rates[t]))
        sales = min(demand, self.stock)
        lost = demand - sales
        self.stock -= sales
        h_cost, s_cost, o_cost = self.holding * self.stock, self.stockout * lost, (self.order_cost if order > 0 else 0.0)
        cost = h_cost + s_cost + o_cost
        self.total_cost += cost
        self.history.append({"t": t, "order": order, "demand": demand, "sales": sales, "lost": lost, "stock_end": self.stock, "cost": cost})
        if self.feedback == "terse":
            fb = f"Day cost: {cost:.2f}."
        elif self.feedback == "standard":
            fb = f"Day cost {cost:.2f} (holding {h_cost:.2f}, stockout {s_cost:.2f}, ordering {o_cost:.2f})."
        else:
            fb = (f"Day cost {cost:.2f} (holding {h_cost:.2f}, stockout {s_cost:.2f}, ordering {o_cost:.2f}). "
                  + (f"Stockout: {lost} units of demand were lost." if lost else "No stockout."))
        return -cost, fb

    def privileged(self, t: int) -> dict:
        h = self.history[-1]
        new, self._new = self._new, []
        return {"true_rate": float(self.true_rates[t]), "demand": h["demand"], "lost": h["lost"], "stock_end": h["stock_end"],
                "cost": h["cost"], "season": int(t // self.season_length) if self.regime == "seasonal" else None,
                "lead_time_now": len(self.pipeline), "confidence": self._last_conf,
                "affected_now": self._changed_at is not None and t - self._changed_at <= self.lead_time + 7, "changes": new}

    def summary(self) -> dict:
        return {"total_cost": self.total_cost, "regime": self.regime, "rates": list(self.rates), "changes": self.changes,
                "n_endogenous": sum(1 for c in self.changes if c["kind"] == "endogenous")}


class BaseStockOracle:
    """Orders up to the newsvendor base-stock level using the TRUE current rate (regret floor)."""

    def __init__(self, world: InventoryWorld):
        self.w = world
        self.crit = world.stockout / (world.stockout + world.holding)

    def choose(self, t: int) -> int:
        L = len(self.w.pipeline)
        rate = float(np.mean(self.w.true_rates[t: t + L + 1])) * (L + 1)
        return max(0, int(poisson.ppf(self.crit, rate)) - (self.w.stock + sum(self.w.pipeline)))


class MovingAverageAgent:
    """Achievable baseline: base-stock policy on a moving average of observed sales."""

    def __init__(self, world: InventoryWorld, window: int = 7):
        self.w, self.window = world, window
        self.crit = world.stockout / (world.stockout + world.holding)

    def choose(self, t: int) -> int:
        sales = [h["sales"] for h in self.w.history[-self.window:]] or [self.w.rates[0]]
        L = len(self.w.pipeline)
        rate = float(np.mean(sales)) * (L + 1)
        return max(0, int(poisson.ppf(self.crit, max(rate, 0.5))) - (self.w.stock + sum(self.w.pipeline)))
