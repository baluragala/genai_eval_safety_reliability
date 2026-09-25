"""Notebook 05: Reliability engineering (block 5, 25 min, demonstration + guided practice)."""
from notebooks._builder import (exercise, explain, glossary, md, predict, reading, run, so_what, solution,
                                step)

NOTEBOOK = "05_reliability.ipynb"
TITLE = "05 · Reliability engineering: keeping the agent working when the world misbehaves"
MINUTES = 25

CONTEXT = {
    "problem": ("A correct, guarded agent still depends on systems that time out, rate-limit, return 503s and crash halfway "
                "through a refund. In production, **reliability failures turn into correctness failures**: a lost refund, "
                "a double refund, a customer stuck in a loop. We need the agent to keep working, or fail honestly, when the "
                "world misbehaves."),
    "start": ("From Notebooks 01 and 04: the golden set and oracle for measuring success, and the guards (a loop guard reappears here). "
              "Tools are now wrapped in a **fault injector** that fails a seeded share of calls."),
    "learn": ["Tell retryable faults from permanent ones, and retry with exponential backoff and jitter within a deadline",
              "Explain why writes need **idempotency keys** before they can be retried or replayed",
              "Checkpoint agent state and resume after a crash without doing a side effect twice",
              "Use circuit breakers, fallbacks and graceful degradation, plus step and budget limits",
              "Run a chaos experiment and compare configurations by success rate and latency"],
    "do": ("Break the world on purpose (timeouts, 429s, 503s, crashes), then add one pattern at a time as an executor wrapper "
           "and measure the difference. It ends with a chaos experiment across fault rates."),
    "given": ("Given: the fault injector, a virtual clock (so backoff waits cost no class time) and reference wrappers "
              "in `agentlab.reliability`. You write: a retry function, and you assemble the wrapper stack."),
}


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

Each pattern is an **executor wrapper**: a function with the same signature as the default tool executor,
`(world, tool_name, args, key) -> ToolResult`. Because they share that shape they stack like middleware:

```python
executor = with_fallback(with_retry(with_breakers(idempotent(faulty_executor), clock)), fallbacks)
```

> ⏱️ **Virtual time.** Tool latency, backoff sleeps and breaker cooldowns advance `world.clock`, a simulated clock.
> A 30-second backoff costs nothing in class, and the numbers are still realistic. LLM latency is real.
"""),
glossary([
    ("T", "the golden tasks by id, e.g. T['T01'] (from al.GOLDEN, src/agentlab/tasks.py)"),
    ("FaultInjector", "wraps an executor and fails a seeded share of tool calls (src/agentlab/reliability.py)"),
    ("suite(...)", "helper defined below: runs tasks under a fault injector and scores each with the oracle"),
    ("unprotected / retried", "suite results (one row per task) with no protection / with read retries"),
    ("Checkpointer", "saves the agent state after every step, for resume after a crash"),
    ("breaker / CircuitBreaker", "the closed → open → half-open state machine"),
    ("FALLBACKS", "tool → fallback function: read replica for lookups, human queue for refunds"),
    ("chaos / chaos_table", "the final experiment: every task × fault rate × configuration"),
    ("world.clock", "virtual time in ms; backoff and cooldowns advance it instantly"),
]),
step("0 · Load the reliability toolkit and a test harness",
     "Every experiment below runs a set of tasks under injected faults and scores them the same way. "
     "Defining that once as `suite()` means each later comparison changes exactly one thing: the executor stack.",
     [("agentlab.reliability", "fault injection and the wrapper patterns, from `src/agentlab/reliability.py`"),
      ("LoopGuard", "the loop guard from Notebook 04 (`src/agentlab/guards.py`)"),
      ("al.GOLDEN", "the 10 golden tickets from Notebook 01 (`src/agentlab/tasks.py`)"),
      ("FLAKY", "the three tools we let fail: order lookups, help-centre search and refunds, the dependencies a real support agent has"),
      ("seed", "each task gets its own seed (1000×seed + task index), so a rerun injects the same faults at the same calls and comparisons are fair")]),
run("""
from agentlab.reliability import (FaultInjector, TransientError, RateLimited, PermanentError, CircuitOpen,
                                  RetryPolicy, with_retry, CircuitBreaker, with_breakers, with_fallback,
                                  cached_order_lookup, refunds_degraded, idempotent, Checkpointer,
                                  ModelWithFallback)
from agentlab.guards import LoopGuard
import random, re
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

print("toolkit loaded · flaky tools:", FLAKY)
"""),
reading(["No table yet, just a confirmation line. `suite()` returns one row per task: `success` (oracle verdict), "
         "`faults` (how many calls the injector failed), `tool_calls`, `virtual_s` (simulated seconds) and `why` (the oracle's reason for a ❌)."],
        expect="The line prints the three flaky tools. If the import fails, re-run the setup cell at the top."),

md("""
## 1 · The world fails in different ways

| Fault | Example | Retry? |
|---|---|---|
| timeout | upstream took > 5 s | ✅ (if safe) |
| 429 rate limit | `retry_after=1500ms` | ✅ after the delay it gives |
| 503 unavailable | deploy in progress | ✅ |
| 400 bad request | malformed args | ❌ retrying won't fix it |
| process crash | OOM, spot instance reclaimed | resume from a checkpoint |
"""),
predict("With **30%** of calls to the three main tools failing and no protection at all, roughly what share of the "
        "10 golden tasks will still pass?",
        ["Almost all: the model will just retry", "About half", "Very few"],
        hint="A refund task makes about three calls to flaky tools. What's the chance all three succeed at 70% each?"),
step("1.1 · Run the golden set with 30% faults and no protection",
     "This is the baseline every pattern below is measured against. Without it there's no way to say a pattern helped.",
     [("al.GOLDEN", "the same 10 tickets as Notebook 01"),
      ("rate = 0.3", "each call to a flaky tool fails with probability 0.3 (timeout, 429 or 503, drawn at random)"),
      ("lambda fi, w: fi", "the executor is the raw fault injector: errors go straight to the model"),
      ("live model", "gpt-4o-mini decides what to do with each error message")]),
run("""
unprotected = suite(al.GOLDEN, lambda fi, w: fi, 0.3, "unprotected")
unprotected
"""),
reading(["`faults`: how many calls the injector failed for that task. 0 means the task ran clean.",
         "`success`: the oracle's verdict. A task can pass despite faults if the model retried on its own, "
         "or fail from a single fault if that fault hit the refund.",
         "`virtual_s`: timeouts add 5 s of virtual time each, so faulty tasks look slow."],
        expect=("Fewer passes than the fault-free 10/10 from Notebook 01. The exact count depends on where the seeded "
                "faults land and how the live model reacts to each error message."),
        deeper=("The model is improvising. It sees `503 service unavailable` as the tool result and might retry, apologise, "
                "escalate or give up, and it can make a different choice next time. Reliability that depends on "
                "improvisation can't be guaranteed. The patterns below move reliability out of the model and into the harness, "
                "where it's deterministic.")),
explain("al.explain.explain_faults(unprotected, 'unprotected, 30% faults')"),
so_what("The model can't be relied on to recover from faults, so the harness has to."),

md("""
## 2 · Retry with exponential backoff and jitter

* **Exponential:** wait `base × 2^attempt`, capped at a maximum, so a struggling service gets time to recover.
* **Jitter:** randomise the wait. Otherwise a thousand clients that failed together retry together and knock the service over again.
* **Deadline:** stop once the total time spent is more than the customer would wait.
* **Only retry retryable errors.** A 400 will fail the same way every time.
"""),
*exercise("write a retry function (4 min)",
          "Implement `my_retry`. Call `fn()`; if it raises an exception whose `.retryable` is True, sleep on the virtual "
          "clock and try again, up to `max_attempts`. Re-raise non-retryable errors immediately. The solution tests it "
          "on a dependency that fails twice and then works.",
          """
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
# Expect "ok after 3 calls": two failures, each followed by a jittered sleep, then success.
"""),
step("2.1 · The backoff schedule, with and without jitter",
     "Before trusting the library's retry, look at the waits it will actually use.",
     [("RetryPolicy", "defaults from `src/agentlab/reliability.py`: base 200 ms, cap 5000 ms, full jitter"),
      ("rng = Random(7)", "seeded so the jittered numbers are the same on every run")]),
run("""
rng = random.Random(7)
backoff_table = pd.DataFrame({"attempt": range(6),
              "no jitter (ms)": [RetryPolicy(jitter=False).backoff(a, rng) for a in range(6)],
              "full jitter (ms)": [round(RetryPolicy().backoff(a, rng)) for a in range(6)]}).set_index("attempt").T
backoff_table
"""),
reading(["Row 1 (no jitter): 200, 400, 800, … doubling until the 5000 ms cap. This is deterministic.",
         "Row 2 (full jitter): a random wait between 0 and the same ceiling, so it can come out *shorter* than the unjittered wait."],
        expect="The first row is identical everywhere. The second row is fixed by the seed but looks irregular, and that irregularity is the point.",
        deeper=("Why not always wait the full ceiling? After an outage, every client that failed at the same moment would "
                "retry at the same moment too: a synchronised 'thundering herd' that knocks the service over again. "
                "Full jitter spreads those retries across the whole window.")),
explain("al.explain.explain_backoff(backoff_table)"),

md("""
### Why writes aren't retried by default
If `issue_refund` times out, you don't know whether the refund went through. Retrying blindly is how customers get
paid twice. `with_retry` retries only read-only tools unless you pass `retry_writes=True`, and you should only pass it
once writes are **idempotent** (§3).
"""),
predict("Now wrap the executor in `with_retry` (reads only). Will retrying fix *every* failure from §1.1?",
        ["Yes, all 10 pass", "Most, but some refund tasks still fail", "No change"],
        hint="Which of the three flaky tools is a write?"),
step("2.2 · Retry reads, then compare with the baseline",
     "Change exactly one thing (add read retries), keeping the same seeds, so any difference is caused by the retries.",
     [("with_retry(fi)", "retries lookup_order, search_kb and recall on retryable errors; never issue_refund (`src/agentlab/reliability.py`)"),
      ("unprotected", "the baseline from step 1.1"),
      ("same seeds", "suite() reuses the per-task seeds, so the same calls are hit by the same faults")]),
run("""
retried = suite(al.GOLDEN, lambda fi, w: with_retry(fi), 0.3, "retry (reads only)")
pd.DataFrame({"unprotected": [unprotected.success.mean(), unprotected.virtual_s.mean()],
              "retry reads": [retried.success.mean(), retried.virtual_s.mean()]},
             index=["success rate", "avg virtual seconds"])
"""),
reading(["`success rate`: the fraction of the 10 tasks that passed.",
         "`avg virtual seconds`: time spent, including backoff sleeps. Retries turn failures into waiting."],
        expect=("Success should go up and latency with it. Any remaining failures are most likely on refund tasks, "
                "because the refund is a write and isn't retried."),
        deeper=("Retries can also lower success on a given run. A retried call makes the conversation longer, and the "
                "live model may then behave differently. That's why each change gets measured, not assumed.")),
explain("al.explain.explain_retry_effect(unprotected, retried)"),
step("2.3 · Which tasks still fail with read retries?",
     "Find the remaining failures, because they point to the next pattern we need.",
     [("retried", "the results from step 2.2")]),
run("""
retried[~retried.success][["task", "faults", "why"]]
"""),
reading(["Each row is a task that still failed. `why` is the oracle's reason.",
         "An empty table means every fault on this seed landed on a read."],
        expect="If refund tasks appear here, their failed call was issue_refund, which with_retry deliberately leaves alone."),
so_what("Retrying writes safely needs idempotency, which is next."),

md("""
## 3 · Checkpointing and idempotency: surviving a crash

LangGraph's checkpointers and "durable execution" engines like Temporal work the same way: **save the state after every step**,
and after a crash, **resume from the last checkpoint** instead of starting over.

The catch is a crash that lands *after* a side effect but *before* the checkpoint records it. On resume, the step runs again.
"""),
predict("The worker crashes right after the T01 refund ($129) lands. We resume from the last checkpoint with **no** "
        "idempotency. How many refunds for A-1001 will be in the ledger?",
        ["1", "2", "0"]),
step("3.1 · Crash after the refund, then resume, with and without idempotency",
     "This is the most expensive reliability bug in agent systems: a replayed write. The demo reproduces it exactly, then fixes it.",
     [("FaultInjector(crash_after={'issue_refund': 1})", "raises ProcessCrash right after the 1st refund has taken effect (`src/agentlab/reliability.py`)"),
      ("Checkpointer()", "saves AgentState after every model step and every tool result; `cp.latest()` is where we resume"),
      ("idempotent(...)", "adds key = task_id:tool_call_id to each refund; the payments system (`issue_refund` in `src/agentlab/tools.py`) dedupes on it"),
      ("T['T01']", "Priya's $129 shoe refund, an eligible refund")]),
run("""
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
"""),
reading(["`💥 …`: the simulated crash, and how many checkpoints were saved before it.",
         "`resumed →`: the second run, started from `cp.latest()`, finishing the conversation.",
         "`ledger`: every refund the payments system recorded."],
        expect=("Without idempotency: two refunds of $129. With it: one. This doesn't depend on the model. The last "
                "checkpoint holds the model's request to refund, so resume replays that exact call."),
        deeper=("The checkpoint holds the model's message, including its `tool_call_id`. On resume the same call replays "
                "with the same id, so `idempotent()` attaches the same key (`task_id:tool_call_id`), and the payments system "
                "recognises the key and returns the original result instead of paying again.\n\n"
                "**The rule:** any write that might be replayed needs an idempotency key, and the key must be derived "
                "from the step, not generated fresh on each attempt. A fresh UUID per attempt would defeat the purpose.")),
explain("al.explain.explain_crash_resume(n0, n1)"),
so_what("Checkpoints make crashes cheap; idempotency keys make replays safe. You need both."),

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
step("4.1 · Watch the state machine on its own",
     "See the three states before putting the breaker inside the agent. No model is involved, so this is fully deterministic.",
     [("CircuitBreaker(clock, failure_threshold=3, cooldown_ms=10_000)", "from `src/agentlab/reliability.py`: opens after 3 failures, probes after 10 s"),
      ("dependency_up_at = 15_000", "a made-up recovery time: the dependency fails until t = 15 s, then works"),
      ("al.SimClock()", "virtual time; each loop iteration is one call, one second apart")]),
run("""
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
breaker_log = pd.DataFrame(log, columns=["t_ms", "state_after", "result"]).T
breaker_log
"""),
reading(["`transitions`: (time, from, to) for each state change.",
         "Table columns are calls one second apart. `result` is ❌ error (the call reached the failing dependency), "
         "fast-fail (the breaker refused without calling), or ✅ ok."],
        expect=("Three errors, then open. Fast-fails until the 10 s cooldown ends, then a half-open probe. The first probe "
                "(t = 12 s) fails because the dependency is still down, so the breaker re-opens; the next probe (t = 22 s) succeeds and it closes. "
                "These times follow from the parameters, not the model."),
        deeper=("Why re-open on a single failed probe, rather than waiting for three failures again? In half-open you're "
                "testing whether the dependency has recovered. One failure answers that, and letting more traffic through "
                "would bring back the load the breaker exists to prevent.")),
explain("al.explain.explain_breaker(breaker.history, breaker_log)"),
predict("Now the orders database is down, and 5 customers write in. Every request retries up to 4 times. "
        "Roughly how many calls reach the dead database **without** a breaker, and **with** one shared breaker?",
        ["Both about 20", "About 20 without, about 3 with", "About 3 without, about 20 with"]),
step("4.2 · The same thing inside the agent: the orders database goes down",
     "Breakers matter **across requests**, so this uses one shared executor for five customers, as a real service would.",
     [("OUTAGE_TASKS", "five golden tasks that all need lookup_order"),
      ("FaultInjector(rates={'lookup_order': 1.0})", "every lookup fails with a 503: a full outage"),
      ("with_breakers(..., failure_threshold=3)", "one breaker per tool, shared by every request through this executor"),
      ("with_retry", "retries each failed lookup; the breaker sits inside it, so it counts each attempt")]),
run("""
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

outage = pd.DataFrame({"retry only": outage_run(False), "retry + breaker": outage_run(True)})
outage
"""),
reading(["`calls that hit the dead DB`: load on the failing dependency.",
         "`virtual seconds (all 5)`: total time the five customers spent, mostly backing off.",
         "`runs finished`: how many conversations ended with a reply (not necessarily a success)."],
        expect=("Far fewer calls reach the database with the breaker. The exact counts depend on how many times the live "
                "model asks for a lookup after seeing the error."),
        deeper=("The breaker doesn't make any request succeed, because the database is still down. It makes the failure "
                "*cheap*: no load on the struggling service and no minutes spent in backoff, and the fallback in §5 can "
                "answer immediately.")),
explain("al.explain.explain_outage(outage)"),
md("**🗣️ Discuss (2 min):** the breaker doesn't make this request succeed. What does it buy you, for this customer and for everyone else?"),

md("""
## 5 · Fallback and graceful degradation

When a dependency is down, do something useful that is **honestly labelled**:
* `lookup_order` → a read replica or cache, flagged `_stale_by: up to 15 minutes`
* `issue_refund` → **queue it for a human** and tell the customer so. Don't fail silently, and don't claim it's done.
"""),
step("5.1 · Two outages with fallbacks in place",
     "Show that the agent can stay useful during an outage without lying about what happened.",
     [("FALLBACKS", "`cached_order_lookup` (a read replica) and `refunds_degraded` (queue for a human), both from `src/agentlab/reliability.py`"),
      ("T['T04']", "Arjun asks where his headlamp is (needs lookup_order)"),
      ("T['T01']", "Priya's $129 refund (needs issue_refund)"),
      ("full stack", "with_fallback(with_retry(with_breakers(...))): retry first, fail fast once the breaker opens, then fall back")]),
run("""
FALLBACKS = {"lookup_order": cached_order_lookup, "issue_refund": refunds_degraded}

def degraded_run(task_id, down):
    w = al.World.fresh()
    fi = FaultInjector(rates={down: 1.0}, kinds=("server_error",))
    ex = with_fallback(with_retry(with_breakers(fi, w.clock)), FALLBACKS)
    tr = al.run_agent(T[task_id], w, executor=ex)
    print(f"━━ {task_id} with {down} DOWN → {tr.status}")
    print("   reply     :", tr.final)
    print("   ledger    :", w.refunds, "| human queue:", [e["reason"] for e in w.escalations])
    final = (tr.final or "").lower()
    return {"task": task_id, "down": down, "status": tr.status, "refunds": len(w.refunds),
            "queued": len(w.escalations),
            "claims_done": bool(re.search(r"processed|refunded|has been issued", final)) and "queue" not in final,
            "served_by_fallback": any(isinstance(st.output, dict) and "_degraded" in st.output for st in tr.tool_calls())}

degraded = [degraded_run("T04", "lookup_order"),     # status question answered from the replica
            degraded_run("T01", "issue_refund")]     # refund queued for a human, and the customer is told
"""),
reading(["`reply`: what the customer is told. For T04 it should give the status; for T01 it should say the refund is queued.",
         "`ledger`: empty for T01, because no money moved.",
         "`human queue`: the queued refund, with the reason it was queued."],
        expect=("T04 is answered from the replica. T01 has no refund in the ledger and one item in the human queue. "
                "The wording of the reply is up to the live model; check that it doesn't claim the refund is done."),
        deeper=("The oracle marks T01 ❌, because the expected outcome was a refund. **That's correct.** Graceful degradation "
                "isn't success. It's failing honestly, with a way forward for the customer.")),
explain("al.explain.explain_degraded(degraded)"),
step("5.2 · The model is a dependency too",
     "The model provider can have an outage like any other service, so we make the primary model fail once and watch the fallback take over.",
     [("primary", "gpt-4o-mini (`al.config.MODEL`)"),
      ("backup", "the fallback model `al.config.FALLBACK_MODEL` (set in `src/agentlab/config.py`)"),
      ("fail_primary_times=1", "`ModelWithFallback` simulates one 503 from the primary provider (`src/agentlab/reliability.py`)"),
      ("T['T04']", "a simple status question, so the only thing that differs is which model answered")]),
run("""
primary = al.OpenAIAgentModel()
backup = al.OpenAIAgentModel(al.config.FALLBACK_MODEL)
m = ModelWithFallback(primary, backup, fail_primary_times=1)
w = al.World.fresh()
fb_trace = al.run_agent(T["T04"], w, model=m)
print("models used per step:", m.used)
print("reply:", fb_trace.final)
"""),
reading(["`models used per step`: which model answered each model call. The first entry is the fallback, because the primary was made to fail.",
         "`reply`: the customer still gets an answer."],
        expect="The first step uses the fallback model and the later steps use the primary. The reply's wording depends on the live models.",
        deeper=("Fallback models behave differently: different tool-calling habits, different refusals. **Run the golden "
                "set against the fallback too.** A fallback nobody has evaluated is an untested code path that only runs "
                "during an incident.")),
explain("al.explain.explain_model_fallback(m.used, fb_trace)"),
so_what("A fallback should keep the customer served and be labelled for what it is, whether that's stale data, a queued action or a backup model."),

md("""
## 6 · Stop conditions: steps, budget, loops

A loop that never ends is a reliability failure *and* a cost failure. Here are three independent brakes, tried on the
same stuck run.
"""),
step("6.1 · Three ways to stop a stuck agent",
     "Compare what each brake leaves behind: a status, a bill, and whether the customer got an answer.",
     [("World.fresh(outages={'lookup_order'})", "lookup_order always answers 'timeout, please retry' (`src/agentlab/tools.py`)"),
      ("max_steps=4", "a run_agent parameter: at most 4 model calls"),
      ("budget_usd=0.0005", "a run_agent parameter: stop once the trace has cost this much"),
      ("LoopGuard(max_repeats=2)", "Notebook 04's guard: blocks a third identical call and tells the model to escalate")]),
run("""
def stuck(**kw):
    w = al.World.fresh(outages={"lookup_order"})          # returns 'please retry' forever
    tr = al.run_agent(T["T04"], w, **kw)
    return {"status": tr.status, "llm calls": len(tr.llm_calls()),
            "lookup calls": tr.tool_names().count("lookup_order"),
            "cost $": round(tr.cost_usd, 5), "reply": (tr.final or "")[:70]}

stops = pd.DataFrame({
    "max_steps=4": stuck(max_steps=4),
    "budget_usd=0.0005": stuck(budget_usd=0.0005),
    "LoopGuard(2)": stuck(guards=[LoopGuard(max_repeats=2)]),
}).T
stops
"""),
reading(["`status`: `max_steps` / `budget_exceeded` mean the harness cut the run off; `done` means the model wrote a reply.",
         "`lookup calls`: how many times the agent hit the broken tool.",
         "`reply`: empty when the run was cut off."],
        expect=("The live model may give up and escalate on its own before any brake fires. If it does, the status is "
                "`done` in every row. That's fine: the brakes are there for when it doesn't."),
        deeper=("Step and budget limits are blunt instruments: they bound cost but leave the customer without an answer. "
                "A LoopGuard is more targeted. It blocks the repeat and, through its error message, steers the model "
                "toward escalating.")),
explain("al.explain.explain_stops(stops)"),

md("""
## 7 · Chaos experiment: which patterns earn their keep?

Same tasks, fault rates from 0% to 40%, three configurations. Every injector is seeded, so the faults hit the same
calls in every configuration.
"""),
predict("At a **40%** fault rate, rank the three configurations (unprotected, retry reads, full stack) by success rate. "
        "Which one will be slowest?"),
step("7.1 · Run the chaos grid",
     "One experiment compares every configuration at every fault rate, so the trade-offs are visible side by side.",
     [("CHAOS_TASKS", "5 golden tasks covering refunds, a denial and status questions (kept small to limit live cost: 45 runs)"),
      ("CONFIGS", "unprotected; retry (reads); full stack = fallback(retry-writes(breakers(idempotent(faults))))"),
      ("fault rates 0.0 / 0.2 / 0.4", "the share of flaky-tool calls that fail"),
      ("seed = int(rate × 10)", "the same seed for every configuration at a given rate, so each faces identical faults")]),
run("""
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
reading(["Rows are fault rates and columns are configurations. Each cell is the success rate over the 5 tasks.",
         "The 0.0 row is the control: with no faults, all three configurations should match."],
        expect=("As faults rise the unprotected column should fall fastest and the full stack hold up best, though the "
                "live model can blur this at small sample sizes (5 tasks)."),
        deeper=("The full stack can still show ❌ where it degraded a refund into the human queue. The oracle counts that "
                "as a failure because no refund happened. That's honest, but whether it counts as success is a product "
                "decision.")),
step("7.2 · Plot success and compare latency",
     "A chart makes the slope visible. The latency table shows what resilience cost.",
     [("chaos_table", "from step 7.1"), ("chaos.virtual_s", "virtual seconds per task, mostly backoff and timeouts")]),
run("""
import matplotlib.pyplot as plt
ax = chaos_table.plot(marker="o", figsize=(7, 4), ylim=(-0.05, 1.05))
ax.set_xlabel("fraction of tool calls that fail"); ax.set_ylabel("task success rate")
ax.set_title("Chaos experiment: success rate vs. injected fault rate"); ax.grid(alpha=.3)
plt.show()
latency_table = chaos.pivot_table(index="fault_rate", columns="config", values="virtual_s", aggfunc="mean").round(1)
latency_table
"""),
reading(["Plot: one line per configuration. A flatter line is more resilient.",
         "Table: average virtual seconds per task. Configurations that retry spend more time."],
        expect="The configurations that succeed more usually take longer. Retries and backoff buy success with latency.",
        deeper=("At some point waiting longer is worse than degrading, and that point is a product decision, not an engineering constant. "
                "A customer on live chat won't wait 30 seconds; a batch refund job will.")),
explain("al.explain.explain_chaos(chaos_table, latency_table)"),
so_what("Each pattern has a measurable price in latency. The chaos grid tells you which ones are worth paying for."),

*exercise("a per-call time budget (3 min)",
          "Write `with_time_budget(inner, max_ms)`. If a single call takes longer than `max_ms` on the world clock, raise "
          "`TransientError` so the retry layer above can try again. Plug it in under `with_retry` and re-run §1's suite at 30%.",
          """
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
step("What this notebook spent",
     "Every OpenAI call in this notebook went through the spend meter, so the final bill can be checked.",
     [("al.METER", "the spend meter in `src/agentlab/llm.py`, capped at $1.00 per notebook")]),
run("al.METER"),
reading(["`calls`: model calls made, `tokens`: prompt + completion, `spent`: estimated USD using the prices in `src/agentlab/config.py`."],
        expect="The chaos grid (45 runs) accounts for most of the spend."),
]
