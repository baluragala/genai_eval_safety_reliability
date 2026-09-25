"""Notebook 06: Operational trust & oversight, plus wrap-up (blocks 6-7, 10 + 5 min)."""
from notebooks._builder import (exercise, explain, glossary, md, predict, reading, run, so_what, solution,
                                step)

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
glossary([
    ("T", "the golden tasks by id (al.GOLDEN, src/agentlab/tasks.py)"),
    ("gate", "[ApprovalGate(refund_over=150)]: pauses refunds over $150 for a human (src/agentlab/guards.py)"),
    ("paused", "a run stopped with status 'awaiting_approval'; its pending tool call is in paused.state.pending"),
    ("al.resume_agent", "continues a paused run with a human decision: approve / reject / edit (src/agentlab/agent.py)"),
    ("oversight_mode", "maps an action to auto / notify / approve / block (src/agentlab/oversight.py)"),
    ("AuditLog / AuditGuard", "a hash-chained log, and the guard that writes every input, tool call and reply to it"),
    ("KillSwitch", "a guard operators use to switch off a tool without a deploy"),
    ("metrics", "the baseline vs hardened scorecard; GATE / release_gate: the ship / don't-ship thresholds"),
]),
md("""
## 1 · Human-in-the-loop: approve, reject, edit

Some actions are too costly to get wrong to leave fully to a model. The pattern:

1. A guard sees a high-impact call and returns **escalate** instead of allow/block.
2. The run **pauses**. Its state is checkpointed, and the pending tool call waits in a queue.
3. A human **approves**, **rejects** or **edits** the call.
4. The run **resumes** from the saved state, as if the tool had just answered.
"""),
predict("The agent handles T07 (a **$210** harness refund, which is eligible) with `ApprovalGate(refund_over=150)` in place. "
        "When the run pauses, what does the payments ledger show?",
        ["One $210 refund, marked pending", "Nothing: no refund yet", "A $150 partial refund"]),
step("1.1 · Run until the approval gate pauses it",
     "See exactly what a human reviewer gets: the tool name, the exact arguments and the reason it was escalated.",
     [("ApprovalGate(refund_over=150)", "from `src/agentlab/guards.py`: escalates issue_refund above $150, and every send_email"),
      ("T['T07']", "Arjun's climbing harness, A-1008, $210, delivered 12 days ago: eligible, but over the gate's limit"),
      ("al.World.fresh()", "a clean store, so the ledger starts empty"),
      ("live model", "gpt-4o-mini works the ticket as usual until it asks for the refund")]),
run("""
from agentlab.guards import ApprovalGate

T = {t.id: t for t in al.GOLDEN}
gate = [ApprovalGate(refund_over=150)]

world = al.World.fresh()
paused = al.run_agent(T["T07"], world, guards=gate)
paused.show()
print("\\nstatus :", paused.status)
print("pending:", paused.state.pending)
print("ledger :", world.refunds)
"""),
reading(["The trace ends with a 🛡️ `ESCALATE issue_refund` line: the guard intercepted the call.",
         "`status: awaiting_approval`: the run is paused, not finished.",
         "`pending`: the exact tool call waiting for a human, and the reason.",
         "`ledger: []`: the refund tool never ran."],
        expect=("The ledger is empty. The guard runs *before* the tool, so no money moves until a human says so. "
                "The steps before the refund depend on the live model, but the gate always fires once it asks to refund $210."),
        deeper=("The paused state is a plain `AgentState` (the messages so far plus the pending call). In production you'd "
                "persist it, which is the checkpointer from Notebook 05, so the approval can come hours later from a "
                "different machine.")),
explain("al.explain.explain_pause(paused, world)"),
step("1.2 · Resolve the same paused run three ways",
     "The human has three choices. Each is run in its own fresh world so the outcomes can be compared side by side.",
     [("decision", "'approve' (run the call as proposed), 'reject' (tell the agent no, with a note), or 'edit' (run it with changed arguments)"),
      ("edited_args", "a supervisor-agreed 50% goodwill refund: $105 instead of $210"),
      ("guards=gate", "passed again on resume so the gate stays in force for any later calls")]),
run("""
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
decisions = pd.DataFrame(outcomes)
decisions
"""),
reading(["`ledger`: approve → one $210 refund; reject → none; edit → one $105 refund. These follow from the decision, not from the model.",
         "`agent_reply`: the model's reply after seeing the human's decision as a tool result. This part is live."],
        expect=("All three end `done`. Read the edit row's reply carefully: does it tell the customer $105, or does it "
                "repeat the $210 it originally asked for?"),
        deeper=("If the reply after an edit states the old amount, that's a silent failure in the *reply*: the ledger is "
                "right but the customer is told the wrong figure. The fix is to make sure the tool result the model sees "
                "states the executed amount clearly, and to test the edit path in your eval suite.")),
explain("al.explain.explain_decisions(decisions)"),
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
step("2.1 · Tier ten sample actions",
     "Make the policy concrete: run the same function the guards would use over a spread of realistic actions.",
     [("oversight_mode(tool, amount, anomalies, reversible)", "from `src/agentlab/oversight.py`; tool risk comes from its `RISK` table (0 = read, 1 = memory write, 2 = money/data out, 3 = code)"),
      ("samples", "hand-picked (tool, amount, anomaly count, reversible) tuples; the anomaly counts stand in for signals such as a new account or repeated refunds")]),
run("""
from agentlab.oversight import oversight_mode

samples = [
    ("lookup_order", 0, 0, True), ("search_kb", 0, 0, True), ("remember", 0, 0, True),
    ("remember", 0, 1, True), ("issue_refund", 39, 0, True), ("issue_refund", 210, 0, True),
    ("issue_refund", 39, 2, True), ("send_email", 0, 0, True), ("send_email", 0, 0, False),
    ("calculator", 0, 0, True),
]
matrix = pd.DataFrame([{"tool": t, "amount": a, "anomalies": n, "reversible": r,
               "mode": oversight_mode(t, amount=a, anomalies=n, reversible=r)} for t, a, n, r in samples])
matrix
"""),
reading(["Each row is one proposed action and the oversight `mode` it gets.",
         "Compare the three `issue_refund` rows: the same tool gets a different mode depending on amount and anomalies."],
        expect="This is deterministic (no model involved). Reads are auto, a $39 refund notifies, a $210 refund or one with anomalies needs approval, and the calculator is blocked.",
        deeper=("Why tier at all? Two failure modes pull in opposite directions: too little oversight lets a fooled agent act, "
                "and too much produces approval fatigue, where reviewers click 'approve' without reading and the control "
                "stops working. Tiering concentrates human attention on the actions that need it.")),
explain("al.explain.explain_matrix(matrix)"),
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
step("3.1 · Audit one run",
     "Record every input, tool call and reply of a real run, so there's something to verify and to tamper with.",
     [("AuditLog", "from `src/agentlab/oversight.py`: each entry stores sha256(content + previous hash)"),
      ("AuditGuard(log)", "a guard that writes on_input / on_tool_call / on_output events to the log"),
      ("T['T01']", "Priya's $129 refund, a normal happy-path run")]),
run("""
from agentlab.oversight import AuditLog, AuditGuard

log = AuditLog()
w = al.World.fresh()
tr = al.run_agent(T["T01"], w, guards=[AuditGuard(log)])
pd.DataFrame(log.entries)[["i", "actor", "action", "detail", "prev", "hash"]]
"""),
reading(["One row per event: `input` (the customer's message), each `tool_call`, then the `reply`.",
         "`prev` is the previous row's `hash`. That link is the chain."],
        expect="The number of rows depends on how many tools the live model called. The first entry's `prev` is always GENESIS."),
predict("Someone edits entry **2** after the fact, changing which order was looked up. Will `verify()` notice, "
        "and which entry will it report as broken?",
        ["It won't notice", "It notices, at entry 2", "It notices, at the last entry"]),
step("3.2 · Verify, tamper, verify again",
     "Show that the log can't be quietly rewritten, which is what makes it evidence.",
     [("log", "the audit log from step 3.1"),
      ("log.verify()", "recomputes every hash and returns (ok, index of the first broken entry)"),
      ("the edit", "overwrites entry 2's detail to say order A-9999 was looked up")]),
run("""
before = log.verify()
log.entries[2]["detail"] = {"task": "T01", "tool": "lookup_order", "args": {"order_id": "A-9999"}}  # someone rewrites history
after = log.verify()
print("intact?  ", before)
print("tampered?", after, "← (ok, index of first broken entry)")
"""),
reading(["`(True, None)`: every hash matches.",
         "`(False, 2)`: entry 2's stored hash no longer matches its content."],
        expect=("Detected at entry 2. That's deterministic: hashing doesn't depend on the model."),
        deeper=("Could an attacker also recompute entry 2's hash? Then entry 3's `prev` would no longer match, so they'd "
                "have to rewrite every later entry too. Real systems anchor the latest hash somewhere the attacker can't write "
                "(a separate service, a WORM bucket), which makes rewriting history detectable.")),
explain("al.explain.explain_audit(log, before, after)"),

md("""
## 4 · The kill switch

When the payments provider reports fraud, or a new attack is spreading, operators need to switch off **one
capability** in seconds, without a deploy and without taking the whole agent down. The agent should then **degrade**:
it keeps answering and hands refunds to humans.
"""),
step("4.1 · Switch refunds off and replay an eligible refund",
     "Check that the agent degrades gracefully, rather than failing or claiming a refund it couldn't make.",
     [("KillSwitch(disabled_tools={'issue_refund'})", "from `src/agentlab/oversight.py`: blocks the tool and tells the model to escalate"),
      ("T['T01']", "an eligible $129 refund, which would normally be paid")]),
run("""
from agentlab.oversight import KillSwitch

w = al.World.fresh()
ks_trace = al.run_agent(T["T01"], w, guards=[KillSwitch(disabled_tools={"issue_refund"})])
ks_trace.show()
print("\\nledger:", w.refunds, "· human queue:", w.escalations)
"""),
reading(["A 🛡️ `BLOCK issue_refund` line: the switch fired when the agent asked to refund.",
         "`ledger`: empty. `human queue`: the handed-off case, if the model followed the switch's instruction to escalate."],
        expect=("No refund. Whether the model escalates and what it tells the customer are up to the live model; "
                "the block message asks it to escalate."),
        deeper=("Why a guard rather than a deploy? A config flag read on every call takes effect in seconds and can be "
                "reverted just as fast. Rehearse flipping it; a kill switch nobody has tried is a hypothesis.")),
explain("al.explain.explain_killswitch(ks_trace, w)"),

md("""
## 5 · Production readiness: one scorecard, one gate

Everything from the session comes together here. We compare the **baseline** agent with the **hardened** agent
(Notebook 04's guardrails + Notebook 05's reliability stack) on every dimension, then apply a **release gate**:
explicit thresholds you would run in CI before every prompt, model or tool change.
"""),
predict("Will the **baseline** agent (no guards, no reliability stack) pass the release gate? If not, which "
        "threshold will stop it first?",
        ["It passes", "It fails on attack_success_rate", "It fails on cost"]),
step("5.1 · Score baseline vs hardened",
     "Put every dimension from the session in one table for both agents, measured the same way.",
     [("al.GOLDEN + al.evaluate", "success, tool correctness, cost and latency (Notebook 01)"),
      ("al.red_team", "attack success rate over the six attacks (Notebooks 03–04)"),
      ("FAULTS = 20% on three tools", "a chaos run for each agent (Notebook 05), seeded 100 + task index"),
      ("default_guardrails()", "the full guard stack from `src/agentlab/guards.py`"),
      ("hardened_stack", "idempotent → breakers → retry (writes too) → fallbacks, from `src/agentlab/reliability.py`"),
      ("cost", "about 250 model calls, around $0.05 with gpt-4o-mini; check al.METER afterwards")]),
run("""
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
reading(["Rows are the dimensions from Notebook 01 plus `attack_success_rate` (lower is better) and `chaos_success_rate`.",
         "Columns are the two agents, measured on identical tasks, attacks and seeded faults."],
        expect=("The hardened agent should show a lower attack success rate and a higher chaos success rate. It may cost "
                "slightly more per task. All of these are live numbers from your run."),
        deeper=("Why run the chaos suite with the guards on as well? Guards and reliability wrappers interact: a blocked "
                "call looks like an error to the retry layer only if it's raised, and here it isn't, so the two layers "
                "compose cleanly. Testing them together is how you'd find out if they didn't.")),
explain("al.explain.explain_scorecard(metrics)"),
step("5.2 · Apply the release gate",
     "Turn the scorecard into a decision, using thresholds written down before looking at the numbers.",
     [("GATE", "thresholds from `src/agentlab/oversight.py`: success ≥ 0.9, tool correctness ≥ 0.9, attack success ≤ 0, chaos ≥ 0.8, cost ≤ $0.01, p95 ≤ 20 s"),
      ("release_gate(metrics)", "returns (ok, rows): one row per threshold, each ✅ or ❌"),
      ("metrics", "the scorecard from step 5.1")]),
run("""
from agentlab.oversight import release_gate, GATE

gate_results = {}
for label, m in metrics.items():
    ok, rows = release_gate(m)
    gate_results[label] = (ok, rows)
    print(f"━━ {label}: {'🚀 SHIP' if ok else '🛑 DO NOT SHIP'}")
    display(pd.DataFrame(rows))
"""),
reading(["One table per agent: `value` is measured, `rule` is the threshold, `pass` is ✅ or ❌.",
         "The headline is 🚀 SHIP only if every row passes."],
        expect=("The baseline is very likely to fail on attack_success_rate, because any breach fails a ≤ 0 threshold. Whether "
                "the hardened agent ships depends on your live numbers, chaos success especially."),
        deeper=("The thresholds are a **product decision**, written down before the numbers come in. If you set them after "
                "you've seen the results, you'll set them wherever the agent happens to pass. And a gate is only as good as its "
                "suites: every production incident should add a task, an attack or a fault, so the same bug can't ship twice.")),
explain("""
for label, (ok, rows) in gate_results.items():
    al.explain.explain_gate(label, ok, rows)
"""),
so_what("Run this gate in CI on every change to the prompt, model, tools or guardrails. A model upgrade can change behaviour just as much as a code change."),

*exercise("tighten the gate (optional)",
          "Require `chaos_success_rate >= 0.95` and `cost_per_task_usd <= 0.001`. Which configuration survives? "
          "What would you change to make it pass: the agent, or the threshold?",
          """
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
step("What this notebook spent",
     "Every OpenAI call went through the spend meter, broken down by purpose.",
     [("al.METER", "the spend meter in `src/agentlab/llm.py`, capped at $1.00 per notebook")]),
run("al.METER.by_purpose, al.METER"),
reading(["`by_purpose`: calls and USD split by agent / judge / guard / simulator. `METER`: the totals."],
        expect="The scorecard in §5 accounts for most of the spend."),
]
