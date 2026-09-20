"""LLM brain: a thin, cached, budget-capped wrapper around the OpenAI Responses API,
with a process-wide cost ledger.

Every (model, effort, system, prompt) triple is hashed and cached on disk, so
re-running an experiment with unchanged config costs zero API calls and is
exactly reproducible. Cache hits are counted separately in the ledger (they
cost nothing but are reported as "would-have-cost" so a cold rerun is
predictable).

Default model: gpt-5-mini, reasoning effort "low".
"""

import asyncio
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path

DEFAULT_MODEL = "gpt-5-mini"


def _load_dotenv():
    """Read KEY=VALUE lines from the repo's .env into the environment (existing vars win)."""
    env = Path(__file__).resolve().parents[2] / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

# USD per 1M tokens: (input, cached input, output). Reasoning tokens bill as output.
# Edit here when prices change; unknown models are costed at 0 and flagged.
PRICES = {
    "gpt-5": (1.25, 0.125, 10.00),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-5-nano": (0.05, 0.005, 0.40),
    "gpt-4.1": (2.00, 0.50, 8.00),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4.1-nano": (0.10, 0.025, 0.40),
    # quoted by the OpenRouter models API, 2026-09-05
    "gpt-5.6-luna": (0.20, 0.02, 1.20),
    "gemini-3.6-flash": (0.75, 0.075, 3.75),
    "claude-sonnet-5": (2.00, 0.20, 10.00),
    # quoted by the OpenRouter models API, 2026-09-14
    "deepseek-v4.1-flash": (0.15, 0.003, 0.60),
    "qwen3.8-27b": (0.214, 0.15, 2.55),
}


def price_call(model: str, input_tokens: int, cached_tokens: int, output_tokens: int) -> float:
    model = model.split("/")[-1]  # vendor-prefixed slugs (openai/gpt-5-mini) price as the bare model
    key = next((k for k in sorted(PRICES, key=len, reverse=True) if model.startswith(k)), None)
    if key is None:
        return 0.0
    p_in, p_cached, p_out = PRICES[key]
    uncached = max(input_tokens - cached_tokens, 0)
    return (uncached * p_in + cached_tokens * p_cached + output_tokens * p_out) / 1e6


class BudgetExceeded(RuntimeError):
    pass


class CostLedger:
    """Aggregates usage and cost across every Brain in the process.

    Mutated only from the single asyncio event loop (LLM calls and harness
    subprocesses all await on it) with no awaits inside record(), so the plain
    dict updates are safe without a lock. If record() is ever called from a
    thread, add one."""

    FIELDS = ("calls", "cache_hits", "input_tokens", "cached_tokens", "output_tokens",
              "reasoning_tokens", "cost_usd", "cache_saved_usd")

    def __init__(self):
        self.by_model: dict = {}
        self.by_run: dict = {}       # run id -> {model -> bucket}: each episode's own calls only
        self.budget_usd: float | None = None
        self.unpriced_models: set = set()

    def _bucket(self, model, store: dict | None = None):
        store = self.by_model if store is None else store
        return store.setdefault(model, {f: 0 for f in self.FIELDS})

    def _buckets(self, model):
        # the global bucket plus the current episode's, read off the run contextvar,
        # so concurrent episodes never absorb each other's calls
        from ..live import current_run
        return (self._bucket(model), self._bucket(model, self.by_run.setdefault(current_run.get(), {})))

    def record(self, model, usage: dict, cached_hit: bool):
        cost = price_call(model, usage["input_tokens"], usage["cached_tokens"], usage["output_tokens"])
        if cost == 0 and usage["input_tokens"] and model not in PRICES:
            self.unpriced_models.add(model)
        for b in self._buckets(model):
            if cached_hit:
                b["cache_hits"] += 1
                b["cache_saved_usd"] += cost
            else:
                b["calls"] += 1
                for f in ("input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens"):
                    b[f] += usage[f]
                b["cost_usd"] += cost
        if not cached_hit:
            self._check_budget()

    def record_costed(self, model: str, cost_usd: float, input_tokens: int = 0, output_tokens: int = 0):
        """Record a call whose price is already known (a harness CLI reporting its own spend)."""
        for b in self._buckets(model):
            b["calls"] += 1
            b["input_tokens"] += input_tokens
            b["output_tokens"] += output_tokens
            b["cost_usd"] += cost_usd
        self._check_budget()

    def start_run(self, rid: str):
        """Forget any earlier attempt's numbers for this run id (a --resume replay)."""
        self.by_run.pop(rid, None)

    def take_run(self, rid: str) -> dict:
        """This episode's own cost block, regardless of what ran concurrently."""
        return self.by_run.pop(rid, {})

    def _check_budget(self):
        if self.budget_usd is not None and self.total_cost() > self.budget_usd:
            raise BudgetExceeded(f"spent ${self.total_cost():.2f} > budget ${self.budget_usd:.2f}")

    def total_cost(self) -> float:
        return sum(b["cost_usd"] for b in self.by_model.values())

    def snapshot(self) -> dict:
        return deepcopy(self.by_model)

    @staticmethod
    def diff(after: dict, before: dict) -> dict:
        out = {}
        for model, b in after.items():
            a = before.get(model, {})
            d = {f: b[f] - a.get(f, 0) for f in CostLedger.FIELDS}
            if d["calls"] or d["cache_hits"]:
                out[model] = d
        return out


LEDGER = CostLedger()

# every live AsyncOpenAI client, so the runner can close them before an event loop
# exits; unclosed clients GC after loop teardown and raise 'Event loop is closed'
import weakref
_CLIENTS: "weakref.WeakSet" = weakref.WeakSet()


async def aclose_clients():
    for c in list(_CLIENTS):
        try:
            await c.close()
        except Exception:
            pass


class Brain:
    def __init__(self, model: str = DEFAULT_MODEL, cache_dir: str = "runs/llm_cache",
                 max_calls: int = 2000, max_output_tokens: int = 1500, reasoning_effort: str = "low",
                 base_url: str | None = None, api: str | None = None, timeout_s: float = 180.0):
        from openai import AsyncOpenAI

        self.model = model
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_calls = max_calls
        self.calls_made = 0
        self.tokens_used = 0
        # Any OpenAI-compatible provider works: set base_url in the agent spec (or
        # OPENAI_BASE_URL in .env) and put that provider's key in OPENAI_API_KEY.
        # OPENROUTER_API_KEY is recognized directly: a vendor-prefixed model slug
        # (openai/gpt-5-mini, anthropic/...) routes to OpenRouter with that key.
        # OpenAI itself speaks the Responses API; other providers get Chat
        # Completions unless the spec forces api="responses".
        effective_url = base_url or os.environ.get("OPENAI_BASE_URL", "")
        # a stalled provider connection must fail fast into the retry/backoff path, not
        # hold a concurrency slot for the SDK's 600s default
        if base_url:
            self._client = AsyncOpenAI(base_url=base_url, timeout=timeout_s)
        elif not effective_url and "/" in model and os.environ.get("OPENROUTER_API_KEY"):
            effective_url = "https://openrouter.ai/api/v1"
            self._client = AsyncOpenAI(base_url=effective_url, api_key=os.environ["OPENROUTER_API_KEY"], timeout=timeout_s)
        else:
            self._client = AsyncOpenAI(timeout=timeout_s)  # reads OPENAI_API_KEY (and OPENAI_BASE_URL)
        self.api = api or ("chat" if effective_url and "openai.com" not in effective_url else "responses")
        _CLIENTS.add(self._client)

    def _cache_path(self, system: str, prompt: str) -> Path:
        # max_output_tokens is part of the key: a truncated (possibly empty) reply under a
        # small cap must not be replayed for a run configured with a larger one
        key = hashlib.sha256(f"{self.model}\x00{self.reasoning_effort}\x00{self.max_output_tokens}"
                             f"\x00{system}\x00{prompt}".encode()).hexdigest()
        return self.cache_dir / f"{key}.json"

    @staticmethod
    def _usage(resp) -> dict:
        u = getattr(resp, "usage", None)
        g = lambda obj, name: getattr(obj, name, 0) or 0  # noqa: E731
        return {
            "input_tokens": g(u, "input_tokens"),
            "output_tokens": g(u, "output_tokens"),
            "cached_tokens": g(getattr(u, "input_tokens_details", None), "cached_tokens"),
            "reasoning_tokens": g(getattr(u, "output_tokens_details", None), "reasoning_tokens"),
        }

    # transient transport failures (a rate-limited shared pool, a dropped connection, a 5xx)
    # are retried with backoff; anything else raises immediately
    RETRY_DELAYS = (2, 8, 30)

    async def _call_with_backoff(self, system: str, prompt: str) -> tuple[str, dict]:
        import openai
        for attempt in range(len(self.RETRY_DELAYS) + 1):
            try:
                return await self._call(system, prompt)
            except (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError) as e:
                if attempt == len(self.RETRY_DELAYS):
                    raise
                delay = self.RETRY_DELAYS[attempt]
                print(f"{self.model}: {type(e).__name__}, retrying in {delay}s ({attempt + 1}/{len(self.RETRY_DELAYS)})")
                await asyncio.sleep(delay)
        raise RuntimeError("unreachable")

    async def _call(self, system: str, prompt: str) -> tuple[str, dict]:
        if self.api == "chat":
            kwargs = dict(model=self.model,
                          messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                          max_completion_tokens=self.max_output_tokens,
                          extra_body={"reasoning": {"effort": self.reasoning_effort}})  # OpenRouter-style; others ignore it
            resp = await self._client.chat.completions.create(**kwargs)
            text = (resp.choices[0].message.content or "") if resp.choices else ""
            u = getattr(resp, "usage", None)
            g = lambda obj, name: getattr(obj, name, 0) or 0  # noqa: E731
            usage = {"input_tokens": g(u, "prompt_tokens"), "output_tokens": g(u, "completion_tokens"),
                     "cached_tokens": g(getattr(u, "prompt_tokens_details", None), "cached_tokens"),
                     "reasoning_tokens": g(getattr(u, "completion_tokens_details", None), "reasoning_tokens")}
            return text, usage
        kwargs = dict(model=self.model, instructions=system, input=prompt, max_output_tokens=self.max_output_tokens)
        if self.model.startswith(("gpt-5", "o")):
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        resp = await self._client.responses.create(**kwargs)
        return resp.output_text or "", self._usage(resp)

    async def complete(self, system: str, prompt: str) -> str:
        path = self._cache_path(system, prompt)
        if path.exists():
            rec = json.loads(path.read_text())
            LEDGER.record(self.model, rec["usage"], cached_hit=True)
            self.tokens_used += rec["usage"]["input_tokens"] + rec["usage"]["output_tokens"]
            return rec["text"]
        if self.calls_made >= self.max_calls:
            raise BudgetExceeded(f"LLM call budget of {self.max_calls} exhausted")
        self.calls_made += 1
        text, usage = await self._call_with_backoff(system, prompt)
        self.tokens_used += usage["input_tokens"] + usage["output_tokens"]
        LEDGER.record(self.model, usage, cached_hit=False)
        path.write_text(json.dumps({"model": self.model, "system": system, "prompt": prompt,
                                    "text": text, "usage": usage}))
        return text


class MockBrain:
    """Credential-free stand-in. `responder(system, prompt) -> str` supplies the
    reply; the default returns "" so agents fall back to their parse-failure
    defaults. Reports estimated tokens to the ledger under model "mock" ($0)."""

    def __init__(self, responder=None, **_):
        self.responder = responder or (lambda s, p: "")
        self.model = "mock"
        self.calls_made = 0
        self.tokens_used = 0

    async def complete(self, system: str, prompt: str) -> str:
        self.calls_made += 1
        text = self.responder(system, prompt)
        usage = {"input_tokens": (len(system) + len(prompt)) // 4, "output_tokens": max(len(text) // 4, 1),
                 "cached_tokens": 0, "reasoning_tokens": 0}
        self.tokens_used += usage["input_tokens"] + usage["output_tokens"]
        LEDGER.record(self.model, usage, cached_hit=False)
        return text


# reasoning-heavy models need output headroom or they truncate to empty replies;
# used when a spec does not set max_output_tokens itself (matching on the bare model name)
DEFAULT_MAX_TOKENS = {"deepseek-v4.1-flash": 12000, "qwen3.8-27b": 6000}


def make_brain(spec: dict, cache_dir: str):
    if spec.get("mock"):
        return MockBrain()
    model = spec.get("model", DEFAULT_MODEL)
    default_cap = DEFAULT_MAX_TOKENS.get(model.split("/")[-1], 1500)
    return Brain(
        model=model,
        cache_dir=cache_dir,
        max_calls=spec.get("max_calls", 2000),
        max_output_tokens=spec.get("max_output_tokens", default_cap),
        reasoning_effort=spec.get("reasoning_effort", "low"),
        base_url=spec.get("base_url"),
        api=spec.get("api"),
        timeout_s=spec.get("timeout_s", 180.0),
    )
