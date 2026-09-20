"""World registry: build any world by name, and report what it can do."""

from .base import capabilities
from .campaign import CampaignWorld
from .claims import ClaimsWorld
from .codebase import CodebaseWorld
from .formfiller import FormWorld
from .gridtext import RuleWorld
from .inventory import InventoryWorld

WORLDS = {"rule_world": RuleWorld, "form_filler": FormWorld, "inventory": InventoryWorld, "codebase": CodebaseWorld,
          "claims_desk": ClaimsWorld, "campaign_desk": CampaignWorld}

# reasonable per-world defaults for a "standard" episode, used when an experiment does not care
DEFAULTS = {
    "rule_world": {"T": 120, "depth": 2},
    "form_filler": {"T": 90, "version_every": 10_000},   # changes are scheduled by scenarios
    # retail texture (weekday cycle + promo spikes) over a FLAT base: the level only moves
    # when a scenario schedules change_latent, so controlled protocols stay controlled
    "inventory": {"T": 90, "regime": "retail", "rates": (6.0,)},
    # api_drift: scheduled substantive changes mix style-guide revisions with library deprecations
    "codebase": {"T": 24, "session_length": 10_000, "api_drift": True},
    "claims_desk": {"T": 120},
    "campaign_desk": {"T": 90, "report_every": 10},
}


def make_world(name: str, seed: int, **kw):
    cls = WORLDS[name]
    params = {**DEFAULTS.get(name, {}), **kw}
    accepted = set(cls.__dataclass_fields__)
    return cls(seed=seed, **{k: v for k, v in params.items() if k in accepted})


def capability_table() -> dict:
    return {name: sorted(capabilities(cls(seed=0))) for name, cls in WORLDS.items()}
