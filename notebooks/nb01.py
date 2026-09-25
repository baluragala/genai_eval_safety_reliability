"""Notebook 01: Evaluating agentic systems (block 1, 20 min, conceptual + demonstration)."""
from notebooks._builder import code, md, solution

NOTEBOOK = "01_evaluating_agents.ipynb"
TITLE = "01 · Evaluating agentic systems: what does \"correct\" even mean?"
MINUTES = 20

CELLS = [
md("""
## Where we are

You've built agents with loops, plans, memory, tools and more than one agent. This session asks what comes next:
**is the agent correct, is it safe, and will it keep working?**

Every notebook today uses the same system:

> **Acme Outfitters support agent.** It's an OpenAI tool-calling agent that looks up orders, reads the help centre,
> issues refunds, emails customers, keeps long-term notes, and hands off to humans.

| Notebook | Block | Question |
|---|---|---|
| **01** | Evaluation | How do we measure an agent? |
| 02 | Traces & failures | When it's wrong, *why* is it wrong? |
| 03 | Safety risks | How does it get attacked? |
| 04 | Guardrails | How do we stop that? |
| 05 | Reliability | How does it survive a flaky world? |
| 06 | Oversight | Where do humans stay in the loop? |
"""),
md("""
## 1 · Meet the agent (and its trace)

`run_agent` is a plain ReAct-style loop: call the model, run whatever tools it asks for, repeat until it replies.
It returns a **trace**, the step-by-step record of every model call and tool call, with tokens, cost and latency.
"""),
code("""
T = {t.id: t for t in al.GOLDEN}
world = al.World.fresh()
trace = al.run_agent(T["T01"], world)
trace.show()
"""),
code("""
# The reply is the agent's own account of what it did. The world is what actually happened.
print("Agent said :", trace.final)
print("Ledger says:", world.refunds)
"""),
md("""
## 2 · Evaluation dimensions

A chatbot reply is judged mostly on its text. An agent **acts**, so we judge it along several axes at once:

| Dimension | Question | How we measure it here |
|---|---|---|
| **Task success** | Did the right thing happen in the world? | oracle reads the ledger, outbox and human queue |
| **Tool correctness** | Right tools, right args, nothing forbidden? | `must_call` / `must_not_call` per task |
| **Trajectory quality** | Sensible order? No wasted steps? | trace assertions (§4) |
| **Cost** | Tokens × price per task | trace token counts |
| **Latency** | p50 / p95 time-to-resolution | trace timings |
| **Robustness** | Does it survive rephrasing, typos, pressure? | simulation (§6) |
| **Safety** | Does it refuse to do harm? | red team (Notebook 03) |

> **Rule of thumb:** where you can, score outcomes against the **state of the world**, not the agent's words.
> Use an LLM judge for the things only language can capture: tone, clarity, helpfulness.
"""),
md("### The golden set\nTen tickets with known right answers. Each one names an observable outcome."),
code("""
pd.DataFrame([{"id": t.id, "customer": t.customer_id, "ticket": t.ticket,
               "expected": t.expect["outcome"], "must_call": t.expect.get("must_call"),
               "must_not_call": t.expect.get("must_not_call")} for t in al.GOLDEN])
"""),
code("""
baseline = al.evaluate(al.GOLDEN, label="baseline")
baseline[["task", "success", "tool_score", "steps", "tool_calls", "tokens", "cost_usd", "latency_s", "why"]]
"""),
code("""
al.summarize(baseline)
"""),
md("""
**🗣️ Discuss (2 min):** if the success rate is 100%, is this agent ready to ship? What haven't we measured yet?

*(Hint: look at the table of dimensions again. We have scored exactly two of them.)*
"""),
md("""
## 3 · Reading one trace like an engineer

Pick any task and read its trace. For each step, ask three questions:
1. **Why** did the model take this step? (What did it know at that point?)
2. Was it the **right** step? (Right tool, right arguments?)
3. What did it **cost**? (Tokens, time.)
"""),
code("""
tr, w = baseline.attrs["traces"]["T03"]   # the $649 tent: should be escalated, not refunded
tr.show()
print("\\nhuman queue:", w.escalations)
"""),
md("""
## 4 · Trace-based evaluation: assert on the trajectory

Outcome checks tell you **whether** something went wrong. Trajectory checks tell you **where** it went wrong, and
they catch "right answer, wrong reasons" runs that will fail next time.

Here are three trajectory rules every refund run should obey.
"""),
code("""
def trajectory_checks(trace):
    names = trace.tool_names()
    issues = []
    if "issue_refund" in names:
        i = names.index("issue_refund")
        if "lookup_order" not in names[:i]:
            issues.append("refund issued before the order was looked up")
        if "search_kb" not in names[:i]:
            issues.append("refund issued without checking policy")
    repeats = max((names.count(n) for n in set(names)), default=0)
    if repeats >= 3:
        issues.append(f"a tool was called {repeats}x (possible loop)")
    if trace.status != "done":
        issues.append(f"ended with status {trace.status}")
    return issues

for tid, (tr, w) in baseline.attrs["traces"].items():
    print(tid, trajectory_checks(tr) or "✓")
"""),
md("""
### 🧪 Your turn (3 min)
Add a rule: **"the agent must call `recall` before anything else"** (our system prompt asks for it).
How many golden runs break it? Is this rule worth failing a build over, or only worth a warning?
"""),
code("""
def recall_first(trace):
    # TODO: return a list of issues (empty if fine)
    ...
"""),
solution("""
def recall_first(trace):
    names = trace.tool_names()
    return [] if names and names[0] == "recall" else [f"first tool was {names[:1] or 'none'}, not recall"]

for tid, (tr, w) in baseline.attrs["traces"].items():
    print(tid, recall_first(tr) or "✓")
# Treat it as a warning, not a gate: the outcome is what matters, and this rule is about process.
"""),
md("""
## 5 · LLM-as-judge, and why the judge needs evaluating too

Some things only language can judge: was the reply clear, polite, honest about what happened?
We use a second model with a **rubric** (policy compliance, groundedness, helpfulness, tone) that returns JSON.

The judge is a model too, so it can be wrong. We **calibrate** it against labels we trust. Here the trusted labels come from
the world-state oracle, which stands in for a human expert.

We'll grade the ten golden runs plus three broken runs from Notebook 02's gallery (a wrong refund, a refund of
the wrong amount, and a refund that silently failed). Each run gets graded twice: **reply only** and **reply + trace**.
"""),
code("""
from agentlab import judge as J
from agentlab.fixtures import failure_gallery

cases = [(t, *baseline.attrs["traces"][t.id]) for t in al.GOLDEN]
cases += [(g["task"], g["trace"], g["world"]) for g in failure_gallery() if g["label"] in ("A", "D", "E")]

rows = []
for task, tr, w in cases:
    oracle = al.check_task(task, tr, w)["success"]
    blind = J.judge(task, tr, with_trace=False)
    seeing = J.judge(task, tr, with_trace=True)
    rows.append({"task": task.id, "oracle": oracle, "judge_reply_only": blind["verdict"] == "pass",
                 "judge_with_trace": seeing["verdict"] == "pass", "why (with trace)": seeing.get("rationale", "")})
judged = pd.DataFrame(rows)
judged
"""),
code("""
pd.DataFrame({
    "reply only": J.calibrate(list(judged.judge_reply_only), list(judged.oracle)),
    "reply + trace": J.calibrate(list(judged.judge_with_trace), list(judged.oracle)),
})
"""),
md("""
**What to look for**
* A **false positive** (judge says pass, oracle says fail) is the dangerous kind: a bad run gets approved.
  "Your refund has been processed!" *reads* perfectly well, and only the trace shows the gateway declined it.
* **Cohen's κ** corrects plain agreement for chance. Agreement of 0.8 means little if 80% of cases pass anyway.
* The judge costs money too. It's in `al.METER.by_purpose["judge"]`.

**Judge hygiene in practice:** give it the evidence (the trace), use a fixed rubric with anchored scores, ask for JSON,
calibrate it on labelled cases before you trust it, and re-calibrate whenever you change the judge model or prompt.
"""),
code("al.METER.by_purpose"),
md("""
## 6 · Simulation-based testing

Ten hand-written tickets can't cover the ways real customers write. Two cheap ways to get more variety:

1. **Perturbations:** rewrite each ticket deterministically (typos, SHOUTING, terse, padded with chit-chat, Hinglish).
   The expected outcome stays the same, so the oracle still applies.
2. **Simulated customers:** a second LLM plays a customer with a goal and a persona across several turns.
   It only reveals its order ID when asked.
"""),
code("""
from agentlab.simulation import perturb, STYLES
import dataclasses

print(perturb(T["T01"].ticket, "typos"))
print(perturb(T["T01"].ticket, "hinglish"))
"""),
code("""
variants = []
for tid in ["T01", "T02", "T05"]:
    for style in STYLES:
        base = T[tid]
        variants.append(dataclasses.replace(base, id=f"{tid}/{style}", ticket=perturb(base.ticket, style)))

robust = al.evaluate(variants, label="perturbed", verbose=False)
robust["base"] = robust.task.str.split("/").str[0]
robust["style"] = robust.task.str.split("/").str[1]
robust.pivot(index="base", columns="style", values="success")
"""),
code("""
# Any failures? Read the trace before you blame the model.
for _, r in robust[~robust.success].iterrows():
    print(r.task, "→", r.why)
"""),
md("### A multi-turn simulated customer"),
code("""
from agentlab.simulation import SCENARIOS, simulate
from agentlab.agent import Task

for sc in SCENARIOS[:2]:
    print(f"━━ {sc.id}: {sc.persona} | goal: {sc.goal}")
    tr, w, transcript = simulate(sc)
    print("   oracle:", al.check_task(Task(sc.id, "", sc.customer_id, sc.expect), tr, w))
"""),
md("""
## 7 · Consistency: pass@k versus pass^k

At temperature 0 the agent is *nearly* deterministic. Production agents often run hotter, and customers retry.
Run the same task *n* times:

* **pass@k:** chance that *at least one* of k tries succeeds. It fits code generation, where you can pick the best attempt.
* **pass^k:** chance that *all* k tries succeed. It fits agents that take real actions for real customers, who need it right every time.
"""),
code("""
hot = al.OpenAIAgentModel(temperature=1.0, seed=None)
trials = {}
for tid in ["T02", "T03"]:
    ok = [al.run_task(T[tid], model=hot)[2]["success"] for _ in range(4)]
    trials[tid] = ok
    n, c = len(ok), sum(ok)
    print(f"{tid}: {c}/{n} succeeded · pass@2={al.pass_at_k(n, c, 2):.2f} · pass^2={al.pass_hat_k(n, c, 2):.2f} · "
          f"pass^4={al.pass_hat_k(n, c, 4):.2f}")
"""),
md("""
## ✅ Takeaways
* An agent **acts**, so score it on the world it leaves behind, not on its account of what it did.
* Report **several dimensions** together: success, tool correctness, cost, latency, robustness. A single number hides trade-offs.
* **Traces** let you assert on *how* the agent got there, which catches "right answer, wrong reasons".
* **LLM-as-judge** is useful but fallible. Give it the evidence, and calibrate it (agreement, κ, false positives) against trusted labels.
* **Simulation** gives you variety that ten hand-written tickets can't. **pass^k** measures the consistency customers actually experience.

➡️ Next: when the oracle says ❌, how do we find out *why*? Notebook 02.
"""),
code("al.METER"),
]
