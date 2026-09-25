"""Notebook 05: Reliability engineering (block 5, 25 min, demonstration + guided practice)."""
from notebooks._builder import code, md, solution

NOTEBOOK = "05_reliability.ipynb"
TITLE = "05 · Reliability engineering: keeping the agent working when the world misbehaves"
MINUTES = 25

CELLS = [
md("""
## Where we are

The agent is correct (Notebook 01) and guarded (Notebook 04). In production its dependencies will still time out,
rate-limit, return 503s and crash halfway through a refund. This notebook adds the patterns production systems use
to cope with that:

| Pattern | Protects against |
|---|---|
| **Retry + exponential backoff + jitter** | transient faults (timeouts, 429, 503) |
| **Idempotency keys** | retries and replays doing a write twice |
| **Checkpointing** | a crash losing the run |
| **Circuit breaker** | hammering a dependency that's down |
| **Fallback / graceful degradation** | a dependency being down for a while |
| **Step and budget limits** | loops and runaway cost |

Each pattern is an **executor wrapper** with the same signature as the default tool executor, so they stack like middleware:

```python
executor = with_fallback(with_retry(with_breakers(idempotent(faulty_executor), clock)), fallbacks)
```

> ⏱️ **Virtual time.** Tool latency, backoff sleeps and breaker cooldowns advance `world.clock`, a simulated clock.
> A 30-second backoff costs nothing in class, and the numbers are still realistic. LLM latency is real.
"""),
code("""
from agentlab.reliability import (FaultInjector, TransientError, RateLimited, PermanentError, CircuitOpen,
                                  RetryPolicy, with_retry, CircuitBreaker, with_breakers, with_fallback,
                                  cached_order_lookup, refunds_degraded, idempotent, Checkpointer,
                                  ModelWithFallback)
from agentlab.guards import LoopGuard
T = {t.id: t for t in al.GOLDEN}
FLAKY = ("lookup_order", "search_kb", "issue_refund")

def suite(tasks, make_executor, rate, label, seed=0):
    \"\"\"Run tasks, each in a fresh world with its own seeded fault injector, and score them.\"\"\"
    rows = []
    for i, t in enumerate(tasks):
        w = al.World.fresh()
        fi = FaultInjector(rates={n: rate for n in FLAKY}, seed=1000 * seed + i)
        tr = al.run_agent(t, w, executor=make_executor(fi, w))
        res = al.check_task(t, tr, w)
        rows.append({"config": label, "fault_rate": rate, "task": t.id, "success": res["success"],
                     "faults": len(fi.log), "tool_calls": len(tr.tool_calls()),
                     "virtual_s": round(w.clock.time() / 1000, 1), "why": "; ".join(res["reasons"])[:80]})
    return pd.DataFrame(rows)
"""),
md("""
## 1 · The world fails in different ways

| Fault | Example | Retry? |
|---|---|---|
| timeout | upstream took > 5 s | ✅ (if safe) |
| 429 rate limit | `retry_after=1500ms` | ✅ after the delay it gives |
| 503 unavailable | deploy in progress | ✅ |
| 400 bad request | malformed args | ❌ retrying won't fix it |
| process crash | OOM, spot instance reclaimed | resume from a checkpoint |

`FaultInjector` wraps the executor and fails a seeded fraction of calls. Here's the golden set with **30%** of calls
to the three main tools failing, and no protection:
"""),
code("""
unprotected = suite(al.GOLDEN, lambda fi, w: fi, 0.3, "unprotected")
print("success rate:", unprotected.success.mean(), "| faults injected:", unprotected.faults.sum())
unprotected
"""),
md("""
The model sees the raw error and has to improvise: it retries, apologises, escalates or gives up. Reliability
shouldn't depend on how the model improvises. It belongs in the harness.
"""),
md("""
## 2 · Retry with exponential backoff and jitter

* **Exponential:** wait `base × 2^attempt`, capped at a maximum, so a struggling service gets time to recover.
* **Jitter:** randomise the wait. Otherwise a thousand clients that failed together retry together and knock the service over again.
* **Deadline:** stop once the total time spent is more than the customer would wait.
* **Only retry retryable errors.** A 400 will fail the same way every time.

### 🧪 Your turn (4 min): write it yourself first
"""),
code("""
import random

def my_retry(fn, *, clock, max_attempts=4, base_ms=200, cap_ms=5000, rng=random.Random(7)):
    \"\"\"Call fn(); on an exception with .retryable True, sleep (clock.sleep) and try again.\"\"\"
    # TODO: loop up to max_attempts; re-raise non-retryable errors immediately;
    #       wait = uniform(0, min(cap_ms, base_ms * 2**attempt)); honour e.retry_after_ms if present
    ...
"""),
solution("""
def my_retry(fn, *, clock, max_attempts=4, base_ms=200, cap_ms=5000, rng=random.Random(7)):
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            if not getattr(e, "retryable", False) or attempt == max_attempts - 1:
                raise
            wait = getattr(e, "retry_after_ms", None) or rng.uniform(0, min(cap_ms, base_ms * 2 ** attempt))
            print(f"  attempt {attempt + 1} failed ({e}); sleeping {wait:.0f} ms")
            clock.sleep(wait)

# A dependency that fails twice, then works.
clock, calls = al.SimClock(), {"n": 0}
def flaky():
    calls["n"] += 1
    if calls["n"] <= 2:
        raise TransientError("503 service unavailable")
    return "ok"
print(my_retry(flaky, clock=clock), f"after {calls['n']} calls and {clock.time():.0f} virtual ms")
"""),
md("### The backoff schedule, with and without jitter"),
code("""
rng = random.Random(7)
pd.DataFrame({"attempt": range(6),
              "no jitter (ms)": [RetryPolicy(jitter=False).backoff(a, rng) for a in range(6)],
              "full jitter (ms)": [round(RetryPolicy().backoff(a, rng)) for a in range(6)]}).set_index("attempt").T
"""),
md("""
### Why writes aren't retried by default
If `issue_refund` times out, you don't know whether the refund went through. Retrying blindly is how customers get
paid twice. `with_retry` retries only read-only tools unless you pass `retry_writes=True`, and you should only pass it
once writes are **idempotent** (§3).
"""),
code("""
retry_reads = lambda fi, w: with_retry(fi)
retried = suite(al.GOLDEN, retry_reads, 0.3, "retry (reads only)")
pd.DataFrame({"unprotected": [unprotected.success.mean(), unprotected.virtual_s.mean()],
              "retry reads": [retried.success.mean(), retried.virtual_s.mean()]},
             index=["success rate", "avg virtual seconds"])
"""),
code("""
# Which tasks still fail with read retries? Look at which tool failed.
retried[~retried.success][["task", "faults", "why"]]
"""),
md("""
Retrying reads alone often isn't enough: some of the faults land on `issue_refund`, and that write isn't retried yet.
Fixing that needs idempotency first.
"""),
md("""
## 3 · Checkpointing and idempotency: surviving a crash

LangGraph's checkpointers and "durable execution" engines like Temporal work the same way: **save the state after every step**,
and after a crash, **resume from the last checkpoint** instead of starting over.

The catch is a crash that lands *after* a side effect but *before* the checkpoint records it. On resume, the step runs again.
"""),
code("""
def crash_and_resume(use_idempotency):
    w, cp = al.World.fresh(), Checkpointer()
    wrap = idempotent if use_idempotency else (lambda ex: ex)
    crashy = FaultInjector(crash_after={"issue_refund": 1})        # dies right after the refund lands
    try:
        al.run_agent(T["T01"], w, executor=wrap(crashy), checkpointer=cp)
    except al.ProcessCrash as e:
        print(f"💥 {e}  (checkpoints saved: {len(cp)})")
    tr = al.run_agent(T["T01"], w, executor=wrap(al.default_executor), checkpointer=cp, state=cp.latest())
    print(f"   resumed → {tr.status}: {tr.final}")
    print(f"   ledger: {[(r['order_id'], r['amount']) for r in w.refunds]}")
    return len(w.refunds_for("A-1001"))

print("WITHOUT idempotency:"); n0 = crash_and_resume(False)
print("\\nWITH idempotency:");   n1 = crash_and_resume(True)
print(f"\\nrefunds issued for A-1001 → without: {n0}, with: {n1}")
"""),
md("""
**How it works:** the checkpoint holds the model's message, including its `tool_call_id`. On resume the same call replays
with the same id, so `idempotent()` attaches the same key, `task_id:tool_call_id`, and the payments system recognises
the key and returns the original result instead of paying again.

> The rule: **any write that might be replayed needs an idempotency key, and the key must be derived from the step,
> not generated fresh on each attempt.**
"""),
md("""
## 4 · Circuit breaker: fail fast when a dependency is down

When a dependency is down, retries make it worse: every request spends its full retry budget and adds load.
A circuit breaker counts failures and, past a threshold, **opens**. Calls then fail immediately, with no load and no waiting.
After a cooldown it goes **half-open** and lets one probe through. If the probe succeeds, it closes again.

```
 closed ──(3 failures)──▶ open ──(cooldown)──▶ half_open ──(success)──▶ closed
                            ▲                     │
                            └────(failure)────────┘
```
"""),
code("""
clock = al.SimClock()
breaker = CircuitBreaker(clock, failure_threshold=3, cooldown_ms=10_000)
dependency_up_at = 15_000     # the payments service comes back at t = 15 s

log = []
for i in range(25):
    t = clock.time()
    if not breaker.allow():
        log.append((t, breaker.state, "fast-fail"))
    elif t >= dependency_up_at:
        breaker.success(); log.append((t, breaker.state, "✅ ok"))
    else:
        breaker.failure(); log.append((t, breaker.state, "❌ error"))
    clock.sleep(1000)

print("transitions:", breaker.history)
pd.DataFrame(log, columns=["t_ms", "state_after", "result"]).T
"""),
md("""
### The same thing inside the agent: the orders database goes down
Breakers matter **across requests**. Five customers write in during the outage, and one breaker is shared by all of them:
"""),
code("""
OUTAGE_TASKS = [T[i] for i in ["T04", "T08", "T01", "T07", "T02"]]

def outage_run(with_breaker):
    w = al.World.fresh()
    fi = FaultInjector(rates={"lookup_order": 1.0}, kinds=("server_error",))
    inner = with_breakers(fi, w.clock, failure_threshold=3) if with_breaker else fi
    executor = with_retry(inner)                      # one executor, shared by every request
    statuses = [al.run_agent(t, w, executor=executor).status for t in OUTAGE_TASKS]
    return {"calls that hit the dead DB": fi.counts.get("lookup_order", 0),
            "virtual seconds (all 5)": round(w.clock.time() / 1000, 1),
            "runs finished": statuses.count("done")}

pd.DataFrame({"retry only": outage_run(False), "retry + breaker": outage_run(True)})
"""),
md("""
**🗣️ Discuss (2 min):** the breaker doesn't make this request succeed. What does it buy you, for this customer and for everyone else?
"""),
md("""
## 5 · Fallback and graceful degradation

When a dependency is down, do something useful that is **honestly labelled**:
* `lookup_order` → a read replica or cache, flagged `_stale_by: up to 15 minutes`
* `issue_refund` → **queue it for a human** and tell the customer so. Don't fail silently, and don't claim it's done.
"""),
code("""
FALLBACKS = {"lookup_order": cached_order_lookup, "issue_refund": refunds_degraded}

def degraded_run(task_id, down):
    w = al.World.fresh()
    fi = FaultInjector(rates={down: 1.0}, kinds=("server_error",))
    ex = with_fallback(with_retry(with_breakers(fi, w.clock)), FALLBACKS)
    tr = al.run_agent(T[task_id], w, executor=ex)
    print(f"━━ {task_id} with {down} DOWN → {tr.status}")
    print("   reply     :", tr.final)
    print("   ledger    :", w.refunds, "| human queue:", [e["reason"] for e in w.escalations])

degraded_run("T04", "lookup_order")     # status question answered from the replica
degraded_run("T01", "issue_refund")     # refund queued for a human, and the customer is told
"""),
md("""
The oracle marks the second run ❌, because no refund happened. **That's the correct outcome.** Graceful degradation
isn't success. It's failing honestly, with a way forward for the customer.

### The model is a dependency too
The primary model provider can have an outage like any other service. `ModelWithFallback` switches to a second model on API errors.
We'll make the primary fail once:
"""),
code("""
primary = al.OpenAIAgentModel()
backup = al.OpenAIAgentModel(al.config.FALLBACK_MODEL)
m = ModelWithFallback(primary, backup, fail_primary_times=1)
w = al.World.fresh()
tr = al.run_agent(T["T04"], w, model=m)
print("models used per step:", m.used)
print("reply:", tr.final)
"""),
md("""
Fallback models behave differently. **Run the golden set against the fallback too.** A fallback nobody has
evaluated is an untested code path that only runs during an incident.
"""),
md("""
## 6 · Stop conditions: steps, budget, loops

A loop that never ends is a reliability failure *and* a cost failure. Three independent brakes:
"""),
code("""
def stuck(**kw):
    w = al.World.fresh(outages={"lookup_order"})          # returns 'please retry' forever
    tr = al.run_agent(T["T04"], w, **kw)
    return {"status": tr.status, "llm calls": len(tr.llm_calls()),
            "lookup calls": tr.tool_names().count("lookup_order"),
            "cost $": round(tr.cost_usd, 5), "reply": (tr.final or "")[:70]}

pd.DataFrame({
    "max_steps=4": stuck(max_steps=4),
    "budget_usd=0.0005": stuck(budget_usd=0.0005),
    "LoopGuard(2)": stuck(guards=[LoopGuard(max_repeats=2)]),
}).T
"""),
md("""
## 7 · Chaos experiment: which patterns earn their keep?

Same tasks, fault rates from 0% to 40%, three configurations. Every injector is seeded, so the faults hit the same
calls in every configuration.
"""),
code("""
CHAOS_TASKS = [T[i] for i in ["T01", "T02", "T04", "T07", "T08"]]
CONFIGS = {
    "unprotected": lambda fi, w: fi,
    "retry (reads)": lambda fi, w: with_retry(fi),
    "full stack": lambda fi, w: with_fallback(
        with_retry(with_breakers(idempotent(fi), w.clock), retry_writes=True), FALLBACKS),
}
chaos = pd.concat([suite(CHAOS_TASKS, make, rate, label, seed=k)
                   for rate in [0.0, 0.2, 0.4] for label, make in CONFIGS.items()
                   for k in [int(rate * 10)]], ignore_index=True)
chaos_table = chaos.pivot_table(index="fault_rate", columns="config", values="success", aggfunc="mean")
chaos_table
"""),
code("""
import matplotlib.pyplot as plt
ax = chaos_table.plot(marker="o", figsize=(7, 4), ylim=(-0.05, 1.05))
ax.set_xlabel("fraction of tool calls that fail"); ax.set_ylabel("task success rate")
ax.set_title("Chaos experiment: success rate vs. injected fault rate"); ax.grid(alpha=.3)
plt.show()
chaos.pivot_table(index="fault_rate", columns="config", values="virtual_s", aggfunc="mean").round(1)
"""),
md("""
Read both tables. Reliability patterns trade **latency** (virtual seconds spent backing off) for **success**. At some point
waiting longer is worse than degrading, and that point is a product decision, not an engineering constant.
"""),
md("""
### 🧪 Your turn (3 min): a per-call time budget
Write `with_time_budget(inner, max_ms)`. If a single call takes longer than `max_ms` on the world clock, raise
`TransientError` so the retry layer above can try again. Plug it in under `with_retry` and re-run §1's suite at 30%.
"""),
code("""
def with_time_budget(inner, max_ms=2000):
    def run(world, name, args, key):
        # TODO
        return inner(world, name, args, key)
    return run
"""),
solution("""
def with_time_budget(inner, max_ms=2000):
    def run(world, name, args, key):
        start = world.clock.time()
        result = inner(world, name, args, key)
        if world.clock.time() - start > max_ms:
            raise TransientError(f"{name}: exceeded {max_ms} ms budget")
        return result
    return run
# (A real timeout would cancel the call. Here we can only notice afterwards, which is also why a
#  timed-out WRITE may already have happened. That's the idempotency lesson again.)

budgeted = suite(al.GOLDEN, lambda fi, w: with_retry(with_time_budget(fi, 2000)), 0.3, "retry + time budget")
print("success:", budgeted.success.mean(), "| avg virtual s:", budgeted.virtual_s.mean().round(1))
"""),
md("""
## The same patterns in LangGraph

| Pattern here | LangGraph |
|---|---|
| `with_retry(RetryPolicy(...))` | `RetryPolicy(max_attempts=..., backoff_factor=..., jitter=True)` on a node |
| `Checkpointer` + `state=cp.latest()` | a checkpointer (`MemorySaver`, `SqliteSaver`, `PostgresSaver`) + `thread_id` |
| `idempotent()` keyed on the step | make tool nodes idempotent; key on thread + task id |
| `ApprovalGate` → `resume_agent` | `interrupt()` in a node, then `Command(resume=...)` |
| `max_steps` | `recursion_limit` |

See *Fault tolerance* in the LangChain docs (reading list).

## ✅ Takeaways
* **Retry only what is retryable**, with exponential backoff, jitter and a deadline.
* **Never retry a write without an idempotency key** derived from the step, not from the attempt.
* **Checkpoint after every step** so a crash costs one step, not the whole conversation.
* **Circuit breakers** fail fast so that one outage doesn't take everything else down with it.
* **Degrade honestly**: a queued refund plus a truthful reply beats a silent failure every time.
* **Measure it:** a chaos experiment shows which patterns earn their latency.

➡️ Next: where do humans stay in the loop? Notebook 06.
"""),
code("al.METER"),
]
