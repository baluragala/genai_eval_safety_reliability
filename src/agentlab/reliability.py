"""Faults and the patterns that survive them. Notebook 05 builds these; Notebook 06 reuses them.

Every executor here has the shape `(world, name, args, key) -> ToolResult`,
so they wrap each other like middleware:

    executor = idempotent(with_retry(with_breaker(faulty(default_executor))))
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

from .agent import ProcessCrash, default_executor
from .tools import ToolResult
from .world import SimClock, World


# ---------------------------------------------------------------- errors

class TransientError(Exception):
    retryable = True


class RateLimited(TransientError):
    def __init__(self, msg, retry_after_ms=2000):
        super().__init__(msg)
        self.retry_after_ms = retry_after_ms


class PermanentError(Exception):
    retryable = False


class CircuitOpen(Exception):
    retryable = False


# ---------------------------------------------------------------- fault injection

@dataclass
class FaultInjector:
    """Wraps an executor and fails some calls, reproducibly (seeded).

    rates:  {"lookup_order": 0.3, "issue_refund": 0.2} probability of a fault per call
    kinds:  which faults to draw from: timeout | rate_limit | server_error | bad_request
    crash_after: {"issue_refund": 1}: kill the process right AFTER the Nth call has taken effect
    """
    inner: object = default_executor
    rates: dict = field(default_factory=dict)
    kinds: tuple = ("timeout", "rate_limit", "server_error")
    seed: int = 7
    crash_after: dict = field(default_factory=dict)
    log: list = field(default_factory=list)

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self.counts = {}

    def __call__(self, world: World, name, args, key):
        self.counts[name] = self.counts.get(name, 0) + 1
        if self.rng.random() < self.rates.get(name, 0.0):
            kind = self.rng.choice(self.kinds)
            self.log.append((name, kind))
            if kind == "timeout":
                world.clock.sleep(5000)
                raise TransientError(f"{name}: upstream timeout after 5000ms")
            if kind == "rate_limit":
                raise RateLimited(f"{name}: 429 too many requests", retry_after_ms=1500)
            if kind == "server_error":
                world.clock.sleep(200)
                raise TransientError(f"{name}: 503 service unavailable")
            raise PermanentError(f"{name}: 400 bad request")
        result = self.inner(world, name, args, key)
        if self.crash_after.get(name) == self.counts[name]:
            self.log.append((name, "CRASH"))
            raise ProcessCrash(f"worker died right after {name} took effect")
        return result


# ---------------------------------------------------------------- retry with backoff

@dataclass
class RetryPolicy:
    max_attempts: int = 4
    base_ms: float = 200
    max_backoff_ms: float = 5000
    deadline_ms: float = 20000
    jitter: bool = True
    seed: int = 7

    def backoff(self, attempt: int, rng: random.Random) -> float:
        cap = min(self.max_backoff_ms, self.base_ms * 2 ** attempt)
        return rng.uniform(0, cap) if self.jitter else cap   # "full jitter"


def with_retry(inner, policy: RetryPolicy = RetryPolicy(), *, retry_writes=False,
               idempotent_tools=("lookup_order", "search_kb", "recall")):
    """Retry retryable failures with exponential backoff and jitter, within a deadline.

    Writes are NOT retried unless `retry_writes=True`. Retrying a refund that
    may already have happened is how customers get paid twice. Turn it on only
    once the executor below is idempotent.
    """
    rng = random.Random(policy.seed)

    def run(world, name, args, key):
        start = world.clock.time()
        last = None
        can_retry = retry_writes or name in idempotent_tools
        for attempt in range(policy.max_attempts):
            try:
                return inner(world, name, args, key)
            except ProcessCrash:
                raise
            except Exception as e:
                last = e
                if not getattr(e, "retryable", False) or not can_retry:
                    raise
                wait = getattr(e, "retry_after_ms", None) or policy.backoff(attempt, rng)
                if world.clock.time() - start + wait > policy.deadline_ms:
                    break
                world.clock.sleep(wait)
        raise TransientError(f"{name}: gave up after {policy.max_attempts} attempts ({last})")
    run.stats = {}
    return run


# ---------------------------------------------------------------- circuit breaker

@dataclass
class CircuitBreaker:
    """closed → (failures ≥ threshold) → open → (cooldown elapsed) → half-open → closed | open"""
    clock: SimClock
    failure_threshold: int = 3
    cooldown_ms: float = 10000
    state: str = "closed"
    failures: int = 0
    opened_at: float = 0.0
    history: list = field(default_factory=list)

    def _to(self, s):
        if s != self.state:
            self.history.append((round(self.clock.time()), self.state, s))
            self.state = s

    def allow(self) -> bool:
        if self.state == "open" and self.clock.time() - self.opened_at >= self.cooldown_ms:
            self._to("half_open")
        return self.state != "open"

    def success(self):
        self.failures = 0
        self._to("closed")

    def failure(self):
        self.failures += 1
        if self.state == "half_open" or self.failures >= self.failure_threshold:
            self.opened_at = self.clock.time()
            self._to("open")


def with_breakers(inner, clock: SimClock, **kw):
    """One circuit breaker per tool (each downstream dependency fails on its own)."""
    breakers: dict[str, CircuitBreaker] = {}

    def run(world, name, args, key):
        b = breakers.setdefault(name, CircuitBreaker(clock, **kw))
        if not b.allow():
            raise CircuitOpen(f"{name}: circuit open, failing fast (dependency is down)")
        try:
            r = inner(world, name, args, key)
        except ProcessCrash:
            raise
        except Exception:
            b.failure()
            raise
        b.success()
        return r
    run.breakers = breakers
    return run


# ---------------------------------------------------------------- fallback & degradation

def with_fallback(inner, fallbacks: dict):
    """If a tool fails, try a named fallback (e.g. a read replica or cache), labelled as degraded."""
    def run(world, name, args, key):
        try:
            return inner(world, name, args, key)
        except ProcessCrash:
            raise
        except Exception as e:
            if name not in fallbacks:
                raise
            r = fallbacks[name](world, args)
            if isinstance(r.data, dict):
                r.data = {**r.data, "_degraded": f"served by fallback because: {e}"}
            return r
    return run


def cached_order_lookup(world: World, args) -> ToolResult:
    """A read replica that lags the primary a little. Good enough to answer 'where is my order'."""
    o = world.orders.get(str(args.get("order_id", "")).upper())
    if not o:
        return ToolResult(False, error="not in cache")
    return ToolResult(True, {"order_id": args["order_id"].upper(), **o, "_stale_by": "up to 15 minutes"})


def refunds_degraded(world: World, args) -> ToolResult:
    """Graceful degradation: payments are down, so queue the refund for a human. Don't fail and don't lie."""
    t = f"H-{len(world.escalations) + 1:03d}"
    world.escalations.append({"ticket": t, "order_id": str(args.get("order_id", "")).upper(),
                              "reason": f"refund of {args.get('amount')} queued: payments degraded",
                              "t": world.clock.time()})
    return ToolResult(True, {"state": "QUEUED_FOR_MANUAL_PROCESSING", "ticket": t,
                             "message": "Payments are temporarily unavailable; the refund has been "
                                        "queued and a human will complete it within 1 business day."})


# ---------------------------------------------------------------- idempotency

def idempotent(inner, side_effect_tools=("issue_refund", "send_email")):
    """Attach a stable idempotency key to every write, so a replay can't do it twice.

    The key is (task, tool_call_id): the checkpointed assistant message keeps
    its tool_call ids, so a resumed run replays the same call with the same key.
    """
    def run(world, name, args, key):
        if name in side_effect_tools:
            if name == "issue_refund":
                args = {**args, "idempotency_key": key}
            elif key in world.idempotency:
                return world.idempotency[key]
        r = inner(world, name, args, key)
        if name == "send_email":
            world.idempotency[key] = r
        return r
    return run


# ---------------------------------------------------------------- checkpointing

class Checkpointer:
    """Saves AgentState after every step. `latest()` is what you resume from after a crash."""

    def __init__(self):
        self.snaps: list[str] = []

    def save(self, state):
        self.snaps.append(json.dumps(state.snapshot()))

    def latest(self):
        from .agent import AgentState
        return AgentState.restore(json.loads(self.snaps[-1])) if self.snaps else None

    def __len__(self):
        return len(self.snaps)


# ---------------------------------------------------------------- model fallback

class ModelWithFallback:
    """Try the primary model; on API errors, use a fallback model."""

    def __init__(self, primary, fallback, *, fail_primary_times: int = 0):
        self.primary, self.fallback = primary, fallback
        self.name = f"{primary.name}→{fallback.name}"
        self._inject = fail_primary_times
        self.used = []

    def __call__(self, messages, tools):
        try:
            if self._inject > 0:
                self._inject -= 1
                raise TransientError("simulated 503 from primary model provider")
            r = self.primary(messages, tools)
            self.used.append(self.primary.name)
            return r
        except Exception:
            self.used.append(self.fallback.name)
            return self.fallback(messages, tools)


def compose(*layers):
    """compose(a, b, c)(base) == a(b(c(base))). Used to read executor stacks top-down."""
    def build(base):
        for layer in reversed(layers):
            base = layer(base)
        return base
    return build

