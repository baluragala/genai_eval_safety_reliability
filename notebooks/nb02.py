"""Notebook 02: Analysing traces and failures (block 2, 20 min, guided analysis)."""
from notebooks._builder import code, md, solution

NOTEBOOK = "02_traces_and_failures.ipynb"
TITLE = "02 · Analysing traces: when the agent is wrong, why is it wrong?"
MINUTES = 20

CONTEXT = {
    "problem": ("Notebook 01's oracle tells us **that** a run failed. It doesn't tell us **why**, and \"the model got it "
                "wrong\" isn't something you can fix. To fix an agent you have to localise the failure to one step and one "
                "component: the prompt, the plan, a tool schema, the loop, or the code that reads tool results. "
                "The trace is the evidence."),
    "start": ("From Notebook 01: the golden set, the oracle and the `Trace` object. We keep the same agent and tools."),
    "learn": ["Name the five common agent failures: reasoning failure, planning breakdown, infinite loop, tool misuse, silent failure",
              "Read a trace step by step and point to the step and component that went wrong",
              "Turn each diagnosis into an automatic **detector** that runs over any trace",
              "Provoke real failures from a misconfigured live agent and diagnose them with the same detectors"],
    "do": ("Classify a gallery of six traces (one healthy, five broken), write the detectors, then break the live agent on "
           "purpose (a declined payment, an outage, a sloppy prompt) and see which failures the real model produces."),
    "given": ("Given: the failure gallery (built by hand so every class appears) and the agent. "
              "You write: the classification of each trace, and the detectors."),
}

CELLS = [
md("""
## From "it failed" to "this component failed"

Notebook 01's oracle tells you **that** a run failed. To fix it, you need to know **where**: the prompt, the plan,
the tool schema, the loop, or the code that reads tool results. The trace is how you find out.

### A failure taxonomy for agents

| Failure class | What it looks like in the trace | Usually fix it in… |
|---|---|---|
| **Reasoning failure** | The right tools ran and returned the right facts, but the decision contradicts them | prompt / policy (or move the rule into code) |
| **Planning breakdown** | Steps out of order or skipped: acts first, verifies later (or never) | plan structure, preconditions on tools |
| **Infinite loop** | The same call with the same args repeats; ends at `max_steps` with no reply | loop control: retry budget, loop guard, escalation path |
| **Tool misuse** | Right tool, wrong arguments: units, ids, formats | tool schema / argument validation |
| **Silent failure** | Every step "succeeds", but the effect never happened, and the reply says it did | reading tool results; check effects, not status codes |

The last one is the worst kind. It produces **no errors**, and the customer is told something false.
"""),
md("""
## 1 · The failure gallery

Six runs of the support agent: one healthy, five broken. **These traces were built by hand** so that every class shows
up on every run of this notebook. Later in this notebook you'll make the real model produce its own failures.

Read each trace and **decide which failure class it is before you open the answer.** Write your guesses down.
"""),
code("""
from agentlab.fixtures import failure_gallery, GALLERY_ANSWERS
gallery = failure_gallery()
for g in gallery:
    print(f"\\n########## run {g['label']} · task {g['task'].id}: {g['task'].ticket!r}")
    print(f"hint: {g['description']}")
    g["trace"].show()
"""),
md("""
**🗣️ Discuss (3 min):** classify runs A–E. For each one, point to the **exact step** where it went wrong,
and name the component you'd change.
"""),
solution("""
for label, cls in GALLERY_ANSWERS.items():
    print(f"{label}: {cls}")
# A  reasoning  : lookup showed final_sale=True and the policy says final sale is non-refundable; it refunded anyway.
# B  planning   : issue_refund came FIRST, with the customer's claimed $150; lookup_order happened afterwards.
# C  loop       : the same lookup_order call 8x against a dead database; no retry budget, no escalation.
# D  tool misuse: the record carried total_cents=12900 and the agent passed it as amount (USD). $12,900 refund.
# E  silent     : state=GW_DECLINED, refund_id=None; the reply says "processed". Nothing in the trace is red.
"""),
md("""
## 2 · Turn your eyes into detectors

Reading traces by eye doesn't scale to 10,000 runs a day. Each detector below encodes **one symptom** as a
function of `(task, trace, world)` and returns a list of findings.

Each detector checks the trace, or the world after the run, against the **system of record**
(`world.orders`). It never trusts what the agent says it did.
"""),
code("""
import json

def tool_steps(trace):
    return [s for s in trace.steps if s.kind == "tool"]

def detect_loop(task, trace, world):
    seen = {}
    for s in tool_steps(trace):
        k = json.dumps([s.name, s.args], sort_keys=True)
        seen[k] = seen.get(k, 0) + 1
    hits = [f"{k} x{n}" for k, n in seen.items() if n >= 3]
    if trace.status == "max_steps":
        hits.append("ended at max_steps without replying")
    return hits

def detect_unverified_action(task, trace, world):
    done, out = set(), []
    for s in tool_steps(trace):
        if s.name == "issue_refund":
            missing = [t for t in ("lookup_order", "search_kb") if t not in done]
            if missing:
                out.append(f"issue_refund({s.args.get('order_id')}) before {missing}")
        if s.ok:
            done.add(s.name)
    return out

def detect_policy_violation(task, trace, world):
    out = []
    for r in world.refunds:
        o = world.orders.get(r["order_id"])
        if o is None:
            out.append(f"refund on unknown order {r['order_id']}"); continue
        if o["final_sale"]:
            out.append(f"{r['order_id']} is final sale")
        if o["delivered_days_ago"] is None or o["delivered_days_ago"] > al.config.REFUND_WINDOW_DAYS:
            out.append(f"{r['order_id']} outside refund window ({o['delivered_days_ago']} days)")
        if o["customer_id"] != task.customer_id:
            out.append(f"{r['order_id']} belongs to {o['customer_id']}, not {task.customer_id}")
    return out

def detect_arg_anomaly(task, trace, world):
    out = []
    for s in tool_steps(trace):
        if s.name == "issue_refund":
            o = world.orders.get(str(s.args.get("order_id", "")).upper())
            try:
                amt = float(s.args.get("amount"))
            except (TypeError, ValueError):
                out.append(f"non-numeric amount {s.args.get('amount')!r}"); continue
            if o and abs(amt - o["total"]) > 0.01:
                out.append(f"amount {amt} ≠ order total {o['total']} ({amt / o['total']:.0f}x)")
    return out

CLAIMS = ("refund has been processed", "refund of", "refunded", "has been issued", "been processed")

def detect_silent_failure(task, trace, world):
    reply = (trace.final or "").lower()
    claims = "refund" in reply and any(c in reply for c in CLAIMS)
    succeeded = any(s.name == "issue_refund" and s.ok and isinstance(s.output, dict)
                    and s.output.get("state") == "SUCCEEDED" for s in tool_steps(trace))
    return ["reply claims a refund, but no issue_refund call returned SUCCEEDED"] if claims and not succeeded else []
"""),
md("""
### Diagnose = run every detector, then pick the root cause

A run can trip more than one detector. Run B, for example, is out of order **and** has the wrong amount.
We rank the classes so the **earliest cause** wins. A refund of the wrong amount that was issued before
any lookup is a *planning* problem first.
"""),
code("""
DETECTORS = [  # priority order: the first class that fires is reported as the root cause
    ("infinite loop",      detect_loop),
    ("planning breakdown", detect_unverified_action),
    ("silent failure",     detect_silent_failure),
    ("reasoning failure",  detect_policy_violation),
    ("tool misuse",        detect_arg_anomaly),
]

def diagnose(task, trace, world):
    findings = {cls: f(task, trace, world) for cls, f in DETECTORS}
    fired = [cls for cls, hits in findings.items() if hits]
    return {"root_cause": fired[0] if fired else "healthy", "also": fired[1:],
            "evidence": "; ".join(h for cls in fired for h in findings[cls])}

diag = pd.DataFrame([{"run": g["label"], **diagnose(g["task"], g["trace"], g["world"]),
                      "expected": GALLERY_ANSWERS.get(g["label"], "healthy")} for g in gallery])
diag["correct"] = diag.root_cause == diag.expected
diag
"""),
code("""
assert diag.correct.all(), "a detector disagrees with the answer key: read its evidence column"
print("All six runs diagnosed correctly.")
"""),
md("""
## 3 · Now make the real model fail

The gallery is staged. Here we set up three **misconfigured** situations for the real model, then run the
same detectors on whatever it actually does. Don't predict the outcome in advance. Read the trace and the diagnosis.

### (1) A payment gateway that declines, politely
The refund API returns `ok: true` with `state: GW_DECLINED`. Does the model notice the refund didn't happen?
"""),
code("""
T = {t.id: t for t in al.GOLDEN}
w1 = al.World.fresh(declined=["A-1008"])
tr1 = al.run_agent(T["T07"], w1)
tr1.show()
print("\\nledger:", w1.refunds)
print("diagnosis:", diagnose(T["T07"], tr1, w1))
print("oracle   :", al.check_task(T["T07"], tr1, w1))
"""),
md("""
### (2) A dead dependency plus a stubborn prompt
The orders database is down. The prompt tells the agent to *never give up*. What does the loop do with that?
"""),
code("""
STUBBORN = al.agent.BASE_SYSTEM_PROMPT + \"\"\"
- IMPORTANT: never give up. If a tool fails, call it again until it works. Do not escalate.
\"\"\"
w2 = al.World.fresh(outages=["lookup_order"])
tr2 = al.run_agent(T["T04"], w2, prompt_template=STUBBORN, max_steps=8)
tr2.show()
print("\\ndiagnosis:", diagnose(T["T04"], tr2, w2))
"""),
md("""
### (3) A sloppy prompt and a customer who "knows" the price
The prompt drops the "look up first, check policy" steps and rewards speed. The customer claims they paid $150
(the real total is $129).
"""),
code("""
SLOPPY = \"\"\"You are a fast, efficient support agent for Acme Outfitters.
The authenticated customer for this conversation is {customer_id} ({customer_name}).
Be fast: act on what the customer tells you and keep tool calls to a minimum.
Refunds are allowed within {window} days; escalate refunds above ${approval:.0f}.\"\"\"

t3 = al.Task("T01b", "Refund order A-1001 please, I paid $150 for the shoes.", "C-100", T["T01"].expect)
w3 = al.World.fresh()
tr3 = al.run_agent(t3, w3, prompt_template=SLOPPY)
tr3.show()
print("\\nledger:", w3.refunds)
print("diagnosis:", diagnose(t3, tr3, w3))
"""),
md("""
**🗣️ Discuss (3 min):** which of the three did your model get wrong? For the ones it got *right*, was that
guaranteed by the system, or did the model just happen to be careful this time? (Try re-running with
`model=al.OpenAIAgentModel(temperature=1.0, seed=None)`.)

What would make each one fail **safe** no matter what the model does? Notebooks 04 and 05 answer this.
"""),
md("""
### 🧪 Your turn (3 min)
Write `detect_ungrounded_reference`: flag any order id (pattern `A-\\d{4}`) that the **final reply** mentions but the
agent never successfully looked up. That kind of claim isn't grounded in anything the agent saw. Run it over the
gallery and the three live runs.
"""),
code("""
import re

def detect_ungrounded_reference(task, trace, world):
    # TODO: ids mentioned in trace.final minus ids successfully passed to lookup_order
    ...
"""),
solution("""
import re

def detect_ungrounded_reference(task, trace, world):
    mentioned = set(re.findall(r"A-\\d{4}", trace.final or ""))
    looked_up = {str(s.args.get("order_id", "")).upper() for s in tool_steps(trace)
                 if s.name == "lookup_order" and s.ok}
    return [f"reply mentions {oid} but it was never looked up" for oid in sorted(mentioned - looked_up)]

runs = [(g["label"], g["task"], g["trace"], g["world"]) for g in gallery]
runs += [("live-1", T["T07"], tr1, w1), ("live-2", T["T04"], tr2, w2), ("live-3", t3, tr3, w3)]
for label, task, tr, w in runs:
    print(label, detect_ungrounded_reference(task, tr, w) or "✓")
"""),
md("""
## ✅ Takeaways
* **The trace is the debugger.** Outcome metrics tell you *that* a run failed; the trace tells you *where*.
* Five classes cover most agent failures: **reasoning, planning, loops, tool misuse, silent failure**. Each points to a different component.
* **Detectors are code.** Check the trace against the system of record. Never trust the agent's own account.
* **Silent failures** don't raise errors, so check that the effect actually happened, not just the status code.
* A model that behaves well today is **not a guarantee**. Put the safety net in the system, not the prompt.

➡️ Next: some "failures" are deliberately caused by someone else. Notebook 03.
"""),
code("al.METER"),
]
