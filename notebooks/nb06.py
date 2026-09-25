"""Notebook 06: Operational trust & oversight, plus wrap-up (blocks 6-7, 10 + 5 min)."""
from notebooks._builder import code, md, solution

NOTEBOOK = "06_oversight_and_wrapup.ipynb"
TITLE = "06 · Operational trust & oversight: keeping humans in the loop"
MINUTES = 15

CONTEXT = {
    "problem": ("Guardrails and reliability patterns are code. Putting an agent in front of customers also takes "
                "**operational trust**: deciding which actions a human must approve, recording what happened in a way nobody "
                "can quietly alter, being able to switch capabilities off in an incident, and having evidence that it's ready to ship."),
    "start": ("Everything from Notebooks 01–05: the evaluation harness, the red team, the guardrail stack and the reliable "
              "executor. Here they come together into one hardened agent."),
    "learn": ["Implement human-in-the-loop approval (approve / reject / edit) that pauses and resumes a run",
              "Choose the level of oversight per action from its risk: auto, notify, approve, block",
              "Keep a tamper-evident audit log and use a kill switch",
              "Combine evaluation, safety and chaos results in a **release gate** that compares the baseline and hardened agents"],
    "do": ("Pause a refund for human review, build an escalation matrix, audit a run and tamper with the log, then score "
           "baseline vs hardened against a release gate. It ends with the wrap-up and Q&A."),
    "given": ("Given: `ApprovalGate`, `AuditLog`, `KillSwitch` and `release_gate` in `agentlab`. "
              "You decide: the thresholds, and whether the agent ships."),
}

CELLS = [
md("""
## Where we are

| Notebook | Block | Question |
|---|---|---|
| 01 | Evaluation | How do we measure an agent? |
| 02 | Traces & failures | When it's wrong, *why* is it wrong? |
| 03 | Safety risks | How does it get attacked? |
| 04 | Guardrails | How do we stop that? |
| 05 | Reliability | How does it survive a flaky world? |
| **06** | **Oversight** | **Where do humans stay in the loop, and how do we decide it's ready to ship?** |

Guardrails and reliability patterns are code. **Operational trust** is everything around the code:
who approves what, who can see what happened, who can turn the agent off, and what evidence says it's ready.
"""),
md("""
## 1 · Human-in-the-loop: approve, reject, edit

Some actions are too costly to get wrong to leave fully to a model. The pattern:

1. A guard sees a high-impact call and returns **escalate** instead of allow/block.
2. The run **pauses**. Its state is checkpointed, and the pending tool call waits in a queue.
3. A human **approves**, **rejects** or **edits** the call.
4. The run **resumes** from the saved state, as if the tool had just answered.

Below, `ApprovalGate(refund_over=150)` intercepts the $210 harness refund (T07).
"""),
code("""
from agentlab.guards import ApprovalGate

T = {t.id: t for t in al.GOLDEN}
gate = [ApprovalGate(refund_over=150)]

world = al.World.fresh()
paused = al.run_agent(T["T07"], world, guards=gate)
paused.show()
print("\\nstatus :", paused.status)
print("pending:", paused.state.pending)
print("ledger :", world.refunds, "← nothing has happened yet")
"""),
md("Now the same paused run, resolved three different ways (each in its own fresh world):"),
code("""
outcomes = []
for decision, extra in [("approve", {}),
                        ("reject", {"note": "Customer already received a replacement."}),
                        ("edit", {"edited_args": {"order_id": "A-1008", "amount": 105.0,
                                                  "reason": "50% goodwill refund agreed by supervisor"}})]:
    w = al.World.fresh()
    tr = al.run_agent(T["T07"], w, guards=gate)
    tr = al.resume_agent(tr, T["T07"], w, decision, guards=gate, **extra)
    outcomes.append({"decision": decision, "status": tr.status, "ledger": w.refunds_for("A-1008"),
                     "agent_reply": tr.final})
pd.DataFrame(outcomes)
"""),
md("""
**The same idea in frameworks you'll meet**

| Concept here | LangGraph | OpenAI Agents SDK |
|---|---|---|
| guard returns `escalate` | `interrupt(payload)` inside a node | tool marked as needing approval / guardrail tripwire |
| `trace.state` (checkpointed) | checkpointer + `thread_id` | run state you can serialise and resume |
| `resume_agent(..., "approve")` | `graph.invoke(Command(resume=...), config)` | approve/reject the pending item, then continue the run |

What stays the same across all three: **the pause is durable** (it survives a restart), and the human sees the
**exact** tool name and arguments, not the model's summary of what it intends to do.
"""),
md("""
## 2 · How much autonomy? An escalation matrix

Not every action needs a human. Asking a person to approve everything burns their time and trains them to click
"approve" without reading. Tier each action by **risk**, **reversibility** and **anomaly signals**:

| Mode | Meaning |
|---|---|
| `auto` | agent acts, and it's logged |
| `notify` | agent acts, and a human is told afterwards and can undo it |
| `approve` | a human must approve first |
| `block` | this agent may never do it |
"""),
code("""
from agentlab.oversight import oversight_mode

samples = [
    ("lookup_order", 0, 0, True), ("search_kb", 0, 0, True), ("remember", 0, 0, True),
    ("remember", 0, 1, True), ("issue_refund", 39, 0, True), ("issue_refund", 210, 0, True),
    ("issue_refund", 39, 2, True), ("send_email", 0, 0, True), ("send_email", 0, 0, False),
    ("calculator", 0, 0, True),
]
pd.DataFrame([{"tool": t, "amount": a, "anomalies": n, "reversible": r,
               "mode": oversight_mode(t, amount=a, anomalies=n, reversible=r)} for t, a, n, r in samples])
"""),
md("""
**🗣️ Discuss (2 min):** a $39 refund is `notify`, but with two anomaly signals (say, a new account and a third
refund this week) it becomes `approve`. What anomaly signals would *you* feed this function in production?
Where would they come from?
"""),
md("""
## 3 · The audit log: what happened, provably

When something goes wrong at 2 a.m., the first question is *"what exactly did the agent do?"*. Traces answer that
for engineers. An **audit log** answers it for compliance: append-only, and **hash-chained**, so editing any past
entry breaks every hash that comes after it.
"""),
code("""
from agentlab.oversight import AuditLog, AuditGuard

log = AuditLog()
w = al.World.fresh()
tr = al.run_agent(T["T01"], w, guards=[AuditGuard(log)])
pd.DataFrame(log.entries)[["i", "actor", "action", "detail", "prev", "hash"]]
"""),
code("""
print("intact?  ", log.verify())
log.entries[2]["detail"] = {"task": "T01", "tool": "lookup_order", "args": {"order_id": "A-9999"}}  # someone rewrites history
print("tampered?", log.verify(), "← (ok, index of first broken entry)")
"""),
md("""
## 4 · The kill switch

When the payments provider reports fraud, or a new attack is spreading, operators need to switch off **one
capability** in seconds, without a deploy and without taking the whole agent down. The agent should then **degrade**:
it keeps answering and hands refunds to humans.
"""),
code("""
from agentlab.oversight import KillSwitch

w = al.World.fresh()
tr = al.run_agent(T["T01"], w, guards=[KillSwitch(disabled_tools={"issue_refund"})])
tr.show()
print("\\nledger:", w.refunds, "· human queue:", w.escalations)
"""),
md("""
## 5 · Production readiness: one scorecard, one gate

Everything from the session comes together here. We compare the **baseline** agent with the **hardened** agent
(Notebook 04's guardrails + Notebook 05's reliability stack) on every dimension, then apply a **release gate**:
explicit thresholds you would run in CI before every prompt, model or tool change.

⏱️ This cell makes roughly 250 model calls (about $0.05 with `gpt-4o-mini`). Check `al.METER` afterwards.
"""),
code("""
from agentlab.guards import default_guardrails
from agentlab.reliability import (FaultInjector, idempotent, with_retry, with_breakers, with_fallback,
                                  cached_order_lookup, refunds_degraded)

FAULTS = {"lookup_order": 0.2, "search_kb": 0.2, "issue_refund": 0.2}

def naive_stack(world, seed):
    return FaultInjector(rates=FAULTS, seed=seed)

def hardened_stack(world, seed):
    ex = idempotent(FaultInjector(rates=FAULTS, seed=seed))
    ex = with_breakers(ex, world.clock)
    ex = with_retry(ex, retry_writes=True)
    return with_fallback(ex, {"lookup_order": cached_order_lookup, "issue_refund": refunds_degraded})

def chaos_success(stack, guards):
    ok = []
    for i, t in enumerate(al.GOLDEN):
        w = al.World.fresh()
        tr = al.run_agent(t, w, guards=guards, executor=stack(w, seed=100 + i))
        ok.append(al.check_task(t, tr, w)["success"])
    return round(sum(ok) / len(ok), 3)

def scorecard(label, guards, stack):
    ev = al.evaluate(al.GOLDEN, label=label, guards=guards, verbose=False)
    rt = al.red_team(label=label, guards=guards, verbose=False)
    m = al.summarize(ev)
    m["attack_success_rate"] = round(rt.breached.mean(), 3)
    m["chaos_success_rate"] = chaos_success(stack, guards)
    return m

metrics = {"baseline": scorecard("baseline", [], naive_stack),
           "hardened": scorecard("hardened", default_guardrails(), hardened_stack)}
pd.DataFrame(metrics)
"""),
code("""
from agentlab.oversight import release_gate, GATE

for label, m in metrics.items():
    ok, rows = release_gate(m)
    print(f"━━ {label}: {'🚀 SHIP' if ok else '🛑 DO NOT SHIP'}")
    display(pd.DataFrame(rows))
"""),
md("""
**Reading the gate**
* The thresholds (`GATE`) are a **product decision**, written down before the numbers come in. If you set them
  after you've seen the results, you'll set them wherever the agent happens to pass.
* A gate is only as good as its suites. Every production incident should add a task, an attack or a fault
  to one of them, so the same bug can't ship twice.
* Run it in CI on every change to the prompt, model, tools or guardrails. A model upgrade can change behaviour just as much as a code change.
"""),
md("""
### 🧪 Your turn (optional)
Tighten the gate: require `chaos_success_rate >= 0.95` and `cost_per_task_usd <= 0.001`.
Which configuration survives? What would you change to make it pass: the agent, or the threshold?
"""),
code("""
strict = dict(GATE)
# TODO: tighten two thresholds, then re-run release_gate(metrics["hardened"], strict)
"""),
solution("""
strict = dict(GATE, chaos_success_rate=(">=", 0.95), cost_per_task_usd=("<=", 0.001))
for label, m in metrics.items():
    ok, rows = release_gate(m, strict)
    print(label, "SHIP" if ok else "DO NOT SHIP", [r["metric"] for r in rows if r["pass"] == "❌"])
# Loosening a threshold needs a written justification. Improving the agent needs evidence
# from the same suites. Both are legitimate, but moving the threshold quietly isn't.
"""),
md("""
## 6 · Governance checklist

| Area | Question to answer before launch |
|---|---|
| **Ownership** | Who owns the agent's behaviour, and who is paged when it misbehaves? |
| **Risk tiers** | Is every tool tiered (auto / notify / approve / block)? Who signed off on the tiers? |
| **Eval gates in CI** | Do the golden, red-team and chaos suites run on every prompt/model/tool change? |
| **Monitoring** | Are traces sampled and scored in production (success, cost, latency, guard hits, escalation rate)? |
| **Human review** | Is the approval queue staffed? What's the SLA? What happens when nobody answers? |
| **Audit** | Is there a tamper-evident log of every action, with the inputs that caused it? |
| **Kill switches** | Can each capability be switched off in seconds without a deploy? Has anyone rehearsed it? |
| **Incident runbook** | Contain (kill switch) → assess (traces + audit) → remediate → add a regression case. |
| **Red-team cadence** | Who attacks the agent, and how often? New tools and new data sources reopen old attacks. |
| **Data & privacy** | What goes into memory, for how long, and can a customer ask to have it deleted? |
"""),
md("""
## ✅ Wrap-up: five things to take away

1. **Evaluating an agent means looking at its reasoning, its behaviour and the outcomes it produces.** Score the world it
   leaves behind, report several dimensions together, and calibrate your LLM judges.
2. **Traces are how you debug agents.** Failures come in recognisable classes (reasoning, planning, loops, tool
   misuse, silent failure), and each one leaves its own signature in the trace.
3. **Autonomy brings its own attack surface.** Prompt injection, poisoned data, poisoned memory and unsafe
   tools turn text into actions. The model is not a security boundary, so the policy in code has to be.
4. **Production agents need reliability patterns.** Retries with backoff, circuit breakers, fallbacks,
   idempotency, checkpoints and graceful degradation turn "works in the demo" into "works on a bad day".
5. **Human oversight is still essential for high-impact systems.** Approval gates, escalation tiers, audit logs,
   kill switches and release gates are what make an autonomous system something you can trust.

### 💬 Q&A prompts
* Which of today's guards would you *remove* if you had to cut latency by half? What would you accept losing?
* Your LLM judge and your oracle disagree on 15% of cases. Which do you trust, and how do you find out?
* Where in *your* current project is the "`issue_refund`", the action you could never undo?

### 📚 Additional readings
* OpenAI: *Guardrails and human review*; *Sandbox Agents*; *Evaluate agent workflows* ([platform.openai.com/docs](https://platform.openai.com/docs))
* LangChain docs: *Test*; *Fault tolerance* ([docs.langchain.com](https://docs.langchain.com))
* Google Cloud: [*What is Human in the Loop*](https://cloud.google.com/discover/human-in-the-loop)
"""),
code("al.METER.by_purpose, al.METER"),
]
