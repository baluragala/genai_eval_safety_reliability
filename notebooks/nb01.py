"""Notebook 01: Evaluating agentic systems (block 1, 20 min, conceptual + demonstration)."""
from notebooks._builder import (exercise, explain, glossary, md, predict, reading, run, so_what, solution,
                                step)

NOTEBOOK = "01_evaluating_agents.ipynb"
TITLE = "01 · Evaluating agentic systems: what does \"correct\" even mean?"
MINUTES = 20

CONTEXT = {
    "problem": ("The agent \"works\" because we watched it handle a few tickets. That isn't evidence. Before it touches real "
                "money we need a repeatable way to say how often it does the right thing, at what cost and speed, and how "
                "well it copes with tickets nobody wrote in advance. A chatbot's reply can be graded on its text. An agent "
                "**acts**, so it has to be graded on what it did."),
    "start": ("A working agent (above) and nothing to measure it with. This is the first notebook of the session; "
              "nothing from the other notebooks is needed."),
    "learn": ["Define meaningful evaluation dimensions for an agent: success, tool correctness, trajectory, cost, latency, robustness, safety",
              "Build a golden set scored by an **oracle** that reads the world state, not the agent's words",
              "Write trace-based checks on *how* the agent reached its answer",
              "Use **LLM-as-judge** with a rubric, and calibrate the judge against trusted labels (agreement, Cohen's κ, false passes)",
              "Use **simulation-based testing** (perturbed tickets, LLM-played customers) and measure consistency with pass@k vs pass^k"],
    "do": ("Run the agent on a 10-ticket golden set, read its traces, grade it with an LLM judge (and grade the judge), "
           "then stress it with simulated customers."),
    "given": ("Given: the agent, the world, the golden set, the oracle (`al.check_task`) and the judge prompt. "
              "You write: trajectory checks, and you interpret every number. Nothing in the text predicts results; "
              "they come from your live run."),
}

CELLS = [
md("""
## Where we are

You've built agents with loops, plans, memory, tools and more than one agent. This session asks what comes next:
**is the agent correct, is it safe, and will it keep working?** Every notebook today tests the same Acme agent:

| Notebook | Block | Question |
|---|---|---|
| **01** | Evaluation | How do we measure an agent? |
| 02 | Traces & failures | When it's wrong, *why* is it wrong? |
| 03 | Safety risks | How does it get attacked? |
| 04 | Guardrails | How do we stop that? |
| 05 | Reliability | How does it survive a flaky world? |
| 06 | Oversight | Where do humans stay in the loop? |
"""),
glossary([
    ("al", "the lab runtime `agentlab`, loaded by the setup cell from `src/agentlab/`"),
    ("T", "the golden tickets by id, e.g. `T['T01']`, built from `al.GOLDEN` (`src/agentlab/tasks.py`)"),
    ("world", "a fresh simulated store (`al.World.fresh()`, `src/agentlab/world.py`): ledger, outbox, human queue"),
    ("trace", "the step-by-step record one `al.run_agent(...)` call returns"),
    ("baseline", "a DataFrame with one row per golden task, from `al.evaluate(...)`; its `.attrs['traces']` keeps every trace"),
    ("judged", "the judge's verdicts next to the oracle's, one row per case (§5)"),
    ("robust", "results for perturbed versions of three tickets (§6)"),
    ("trials", "repeated runs of the same ticket at temperature 1.0 (§7)"),
]),
md("## 1 · Meet the agent, and its trace"),
step("1.1 · Run one ticket end to end",
     "Before measuring anything we need to see what one run produces. `run_agent` is a plain ReAct loop: call the "
     "model, run whatever tools it asks for, repeat until it replies. It returns a **trace**.",
     [("al.GOLDEN", "10 hand-written tickets with known right answers, from `src/agentlab/tasks.py`"),
      ("T['T01']", "Priya (customer C-100) asks for a refund on order A-1001, $129 trail shoes delivered 5 days ago; "
                   "the store policy says yes"),
      ("al.World.fresh()", "a brand-new store from `src/agentlab/world.py`, so this run can't be affected by any other"),
      ("the model", "`gpt-4o-mini` at temperature 0, seed 7 (`src/agentlab/config.py`), called with your Colab secret")]),
run("""
T = {t.id: t for t in al.GOLDEN}
world = al.World.fresh()
trace = al.run_agent(T["T01"], world)
trace.show()
"""),
reading([
    "The header line: the model used, `status` (`done` means the agent replied), the number of LLM and tool calls, total cost and time.",
    "🧠 `llm` lines: what the model decided at that point, either a tool call with arguments or its final text.",
    "🔧 lines: the tool that ran and its result (✅ ok / ❌ error). The result is what the model sees on its next call.",
    "💬 `final`: the reply the customer would read.",
], expect="Roughly recall → lookup_order → search_kb → issue_refund → reply. The system prompt asks for that order "
          "(`BASE_SYSTEM_PROMPT` in `src/agentlab/agent.py`). The model decides the actual order, so yours may differ; "
          "it may skip `search_kb`, for example, because the policy is already in the prompt.",
   deeper="""Each tool call costs one extra model call. The model asks for the tool, the harness runs it, and the result is
appended to the conversation. So the model is called once per tool call plus once for the final reply. Every call
resends the whole conversation so far, which is why later calls cost more tokens than earlier ones."""),
explain("al.explain.explain_trace(trace, world, T['T01'])"),
step("1.2 · Compare the reply with the world",
     "The reply is the agent's *account* of what it did. The ledger is what actually happened. Evaluation should read the second.",
     [("trace.final", "the reply from step 1.1"),
      ("world.refunds", "the payments ledger in the same `world` from step 1.1, written only by the `issue_refund` tool")]),
run("""
print("Agent said :", trace.final)
print("Ledger says:", world.refunds)
"""),
reading([
    "`Agent said`: free text. It could be accurate, vague or wrong.",
    "`Ledger says`: a list with one dict per refund that really went through (`refund_id`, `order_id`, `amount`).",
], expect="One refund of 129.0 on A-1001, and a reply that mentions it. If the ledger is empty but the reply says "
          "\"refunded\", you've already found a silent failure, which is Notebook 02's topic."),
so_what("From here on, every **pass/fail** in this session comes from the world's state, never from parsing the reply."),

md("""
## 2 · Evaluation dimensions

An agent **acts**, so we judge it along several axes at once:

| Dimension | Question | How we measure it here |
|---|---|---|
| **Task success** | Did the right thing happen in the world? | oracle reads the ledger, outbox and human queue |
| **Tool correctness** | Right tools, nothing forbidden? | `must_call` / `must_not_call` per task |
| **Trajectory** | Sensible order? No wasted steps? | trace assertions (§4) |
| **Cost** | Tokens × price per task | trace token counts |
| **Latency** | p50 / p95 time to resolution | trace timings |
| **Robustness** | Survives rephrasing, typos, pressure? | simulation (§6–7) |
| **Safety** | Refuses to do harm? | red team (Notebook 03) |
"""),
step("2.1 · Look at the golden set",
     "An evaluation is only as good as its test cases. Before running anything, read what each ticket expects and why.",
     [("al.GOLDEN", "the 10 tickets from `src/agentlab/tasks.py`. Each `expect` dict was written by hand from the store policy "
                    "and the order data in `src/agentlab/world.py` (e.g. A-1002 was delivered 45 days ago, so the answer is deny)")]),
run("""
pd.DataFrame([{"id": t.id, "customer": t.customer_id, "ticket": t.ticket,
               "expected": t.expect["outcome"], "must_call": t.expect.get("must_call"),
               "must_not_call": t.expect.get("must_not_call")} for t in al.GOLDEN])
"""),
reading([
    "`expected`: the outcome class. **refund** = money must move for the exact total; **deny** = no money moves; "
    "**escalate** = a human ticket must exist and no money moves; **info** = the reply must contain a key fact.",
    "`must_call`: tools the right path can't skip (you can't refund without looking up the order).",
    "`must_not_call`: tools that would be wrong here, e.g. `issue_refund` on a deny case.",
], expect="A mix of happy paths (T01, T07), policy denials (T02 out of window, T05 final sale), an escalation (T03 is over $500), "
          "information requests (T04, T06, T08), a missing order (T09) and an authorization trap: T10 asks for a refund "
          "on someone else's order."),
predict("Run all ten tickets. How many will the baseline agent get right, judged by the world's state?",
        ["All 10: it's a capable model with the policy in its prompt",
         "8–9: one or two edge cases (T09 unknown order, T10 someone else's order) trip it up",
         "7 or fewer"],
        hint="Which tickets need the agent to *refuse* something the customer is asking for?"),
step("2.2 · Run the golden set",
     "Measure success and tool correctness on every ticket, each in its own fresh world.",
     [("al.GOLDEN", "the 10 tickets from step 2.1"),
      ("al.evaluate", "from `src/agentlab/evals.py`. For each ticket it creates `World.fresh()`, runs the agent and calls the "
                      "oracle `al.check_task`, which compares the world with `expect`"),
      ("the model", "same settings as step 1.1; this cell makes about 40–60 model calls, which costs a cent or two")]),
run("""
baseline = al.evaluate(al.GOLDEN, label="baseline")
baseline[["task", "success", "tool_score", "steps", "tool_calls", "tokens", "cost_usd", "latency_s", "why"]]
"""),
reading([
    "The ✅/❌ lines print live as each ticket finishes, with the oracle's reason for any failure.",
    "`success`: the oracle's verdict from the world state. `why` says what was wrong when it's False.",
    "`tool_score`: share of `must_call` tools used, forced to 0 if a `must_not_call` tool was used.",
    "`steps` / `tool_calls` / `tokens` / `cost_usd` / `latency_s`: effort per ticket, read from the trace.",
], expect="Mostly ✅. If a ticket fails, the `why` column names the exact world-state problem, e.g. "
          "\"refunded A-1006 but policy says deny\" means the agent refunded someone else's order.",
   deeper="""`tool_score` and `success` can disagree. The agent can pass the outcome while skipping `search_kb` (it knows the
policy from its prompt), which costs tool correctness but not success. Whether that matters is your call. Here the
golden set only requires `lookup_order` and the action tool, and treats policy lookup as optional."""),
explain("al.explain.explain_eval(baseline, 'Baseline golden-set run')"),
step("2.3 · Summarise into the evaluation dimensions",
     "Turn ten rows into the numbers you'd put on a dashboard or a release gate.",
     [("baseline", "the DataFrame from step 2.2"),
      ("al.summarize", "from `src/agentlab/evals.py`: means for success and tool correctness, cost per task, p50/p95 latency")]),
run("al.summarize(baseline)"),
reading([
    "`success_rate`, `tool_correctness`: the means of the two columns above (0–1).",
    "`avg_llm_steps`: model calls per ticket, which drives both cost and latency.",
    "`cost_per_task_usd` / `total_cost_usd`: priced with `PRICES` in `src/agentlab/config.py` (illustrative prices).",
    "`p50_latency_s` / `p95_latency_s`: the typical ticket and the slow tail. Customers feel the p95.",
], expect="Cost per task is a fraction of a cent with gpt-4o-mini. p95 is noticeably above p50, because the multi-step "
          "refund tickets take more model calls than the one-lookup status questions."),
so_what("**🗣️ Discuss:** if `success_rate` is 1.0, is it ready to ship? We've scored two of the seven dimensions. "
        "The rest of the session fills in the others."),

md("## 3 · Reading one trace like an engineer"),
step("3.1 · Inspect the escalation case",
     "T03 is the one ticket where the right answer is to *not* act and hand off instead. It's the best place to check the "
     "agent's judgement, not just its tool use.",
     [("baseline.attrs['traces']['T03']", "the (trace, world) pair `al.evaluate` stored for T03 in step 2.2, so no new model calls"),
      ("T03", "Mei (C-102) wants a refund on a $649 tent. The policy says refunds over $500 must go to a human")]),
run("""
tr, w = baseline.attrs["traces"]["T03"]
tr.show()
print("\\nhuman queue:", w.escalations)
print("ledger     :", w.refunds)
"""),
reading([
    "For each step ask: **why** did the model do this (what did it know?), was it **right**, and what did it **cost**?",
    "`human queue`: escalations the agent created (ticket id, reason, order id).",
    "`ledger`: should be empty for T03.",
], expect="lookup_order → maybe search_kb → escalate_to_human → a reply saying a supervisor will review it. If the ledger "
          "has a refund, the agent broke the $500 rule even though the rule is in its prompt."),
explain("al.explain.explain_check(T['T03'], tr, w)"),

md("""
## 4 · Trace-based evaluation: assert on the trajectory

Outcome checks tell you **whether** something went wrong. Trajectory checks tell you **where**, and they catch
"right answer, wrong reasons" runs that will fail next time.
"""),
step("4.1 · Check three trajectory rules on every run",
     "Encode what a *good path* looks like, independent of the outcome.",
     [("baseline.attrs['traces']", "all ten (trace, world) pairs from step 2.2"),
      ("the rules", "written in this cell: refund only after lookup_order and search_kb; no tool called 3+ times; the run must end with status `done`")]),
run("""
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

trajectory = {tid: trajectory_checks(tr) for tid, (tr, w) in baseline.attrs["traces"].items()}
for tid, issues in trajectory.items():
    print(tid, issues or "✓")
"""),
reading([
    "✓ means every rule held. Otherwise you get the list of rules the path broke.",
    "A task can pass the oracle **and** break a trajectory rule. That's the \"right answer, wrong reasons\" case.",
], expect="If the model skips `search_kb` before refunding (it can, since the policy is in its prompt), T01 and T07 get "
          "flagged \"without checking policy\" while still passing the oracle. Whether that's a real problem is a "
          "design decision, which is exactly why these rules are separate from the outcome."),
explain("al.explain.explain_trajectory(trajectory)"),
*exercise("recall must come first",
          "Our system prompt asks the agent to call `recall` before anything else. Write `recall_first(trace)` returning a "
          "list of issues, and run it over `baseline.attrs['traces']`. Is this rule worth failing a build over, or only a warning?",
          """
def recall_first(trace):
    # TODO: return [] if the first tool call is recall, else a list with one issue
    ...
"""),
solution("""
def recall_first(trace):
    names = trace.tool_names()
    return [] if names and names[0] == "recall" else [f"first tool was {names[:1] or 'none'}, not recall"]

for tid, (tr, w) in baseline.attrs["traces"].items():
    print(tid, recall_first(tr) or "✓")
# Make it a warning, not a gate: the outcome is what matters, and this rule is about process.
"""),

md("""
## 5 · LLM-as-judge, and why the judge needs evaluating too

Some things only language can judge: was the reply clear, polite, honest about what happened? A second model grades
each run against a **rubric** and returns JSON. The judge is a model too, so it can be wrong. We **calibrate** it
against labels we trust: the world-state oracle stands in for a human expert.
"""),
predict("Gallery run E (Notebook 02) is a refund the payment gateway **declined**, but the reply says "
        "\"Great news, your refund of $210.00 has been processed!\". Which judge approves it?",
        ["Both judges catch it", "Only the judge that sees the trace catches it", "Neither catches it"],
        hint="What evidence does the reply-only judge actually have?"),
step("5.1 · Grade 13 runs twice: reply-only and reply + trace",
     "Measure how much the judge's verdict depends on the evidence it's given.",
     [("cases", "the 10 golden runs from step 2.2, plus 3 broken runs from `src/agentlab/fixtures.py` (A: refunded a final-sale "
                "item; D: refunded $12,900 instead of $129; E: the silent failure above). The fixtures were **built by hand** "
                "so these failure types are always present"),
      ("J.RUBRIC", "the grading prompt in `src/agentlab/judge.py`: policy compliance, groundedness, helpfulness and tone, scored 1–5, "
                   "with pass only if policy and groundedness are both ≥4"),
      ("J.judge", "one `gpt-4o-mini` call per case per mode, returning JSON (about 26 calls)"),
      ("oracle", "`al.check_task` on each case's world: the trusted label")]),
run("""
from agentlab import judge as J
from agentlab.fixtures import failure_gallery

cases = [(t.id, t, *baseline.attrs["traces"][t.id]) for t in al.GOLDEN]
cases += [(f"fixture-{g['label']}", g["task"], g["trace"], g["world"])
          for g in failure_gallery() if g["label"] in ("A", "D", "E")]

rows = []
for case_id, task, tr, w in cases:
    oracle = al.check_task(task, tr, w)["success"]
    blind = J.judge(task, tr, with_trace=False)
    seeing = J.judge(task, tr, with_trace=True)
    rows.append({"case": case_id, "oracle": oracle, "judge_reply_only": blind["verdict"] == "pass",
                 "judge_with_trace": seeing["verdict"] == "pass", "why (with trace)": seeing.get("rationale", "")})
judged = pd.DataFrame(rows)
judged
"""),
reading([
    "`oracle`: the trusted label (True = the run was actually right).",
    "`judge_reply_only`: the judge saw the ticket and the reply only.",
    "`judge_with_trace`: the judge also saw every tool call and result.",
    "`why (with trace)`: the judge's own one-line rationale.",
    "Rows where a judge column is True but `oracle` is False are **false passes**, the dangerous error.",
], expect="The last three rows (`fixture-A`, `fixture-D`, `fixture-E`) are the hand-built broken runs and always have oracle False. The reply-only judge has no way to see "
          "that the gateway declined, so it may pass E. The trace-aware judge can read `GW_DECLINED`."),
step("5.2 · Calibrate both judges against the oracle",
     "One number per judge that says how far to trust it.",
     [("judged", "the table from step 5.1"),
      ("J.calibrate", "from `src/agentlab/judge.py`: agreement, Cohen's κ, and counts of false passes (FP) and false fails (FN)")]),
run("""
calib_reply = J.calibrate(list(judged.judge_reply_only), list(judged.oracle))
calib_trace = J.calibrate(list(judged.judge_with_trace), list(judged.oracle))
pd.DataFrame({"reply only": calib_reply, "reply + trace": calib_trace})
"""),
reading([
    "`agreement`: the share of cases where judge and oracle agree.",
    "`kappa`: agreement corrected for chance. If most cases are good runs, a judge that always says \"pass\" already agrees most of the time. κ strips out that free agreement: 0 means no better than chance, 1 means perfect.",
    "`judge_passed_but_wrong (FP)`: bad runs the judge approved. This is the row to watch.",
    "`judge_failed_but_right (FN)`: good runs the judge rejected. Annoying, but safe.",
], deeper="""**Judge hygiene in practice:** give the judge the evidence (the trace), use a fixed rubric with anchored scores,
ask for JSON, calibrate it on labelled cases before you trust it, and re-calibrate whenever you change the judge's
model or prompt. The judge costs money too: see `al.METER.by_purpose['judge']`."""),
explain("al.explain.explain_judge(judged, calib_reply, calib_trace)"),
so_what("Use the oracle for *what happened* and the judge for *how it was said*. And measure the judge before trusting it."),

md("""
## 6 · Simulation-based testing

Ten hand-written tickets can't cover how real customers write. Two ways to get more variety:
**perturbations** (same facts, different wording) and **simulated customers** (a second LLM plays the customer).
"""),
step("6.1 · See what a perturbation does",
     "Before running 15 variants, look at two so you know what the agent will receive.",
     [("perturb, STYLES", "from `src/agentlab/simulation.py`: deterministic rewrites (typos, shouting, terse, padded, Hinglish). "
                          "No model calls, so the same input always gives the same text"),
      ("T['T01'].ticket", "the refund ticket from step 1.1")]),
run("""
from agentlab.simulation import perturb, STYLES
import dataclasses

for style in ["typos", "terse", "hinglish"]:
    print(f"{style:9}: {perturb(T['T01'].ticket, style)}")
"""),
reading([
    "Each line is the same request in a different voice. The order id and the intent stay the same, so the oracle's `expect` still applies.",
], expect="`terse` becomes just \"A-1001 refund\", and `hinglish` mixes Hindi and English around the original request."),
predict("We'll run 3 tickets × 5 styles = 15 variants. Which style is most likely to change the agent's decision?",
        ["typos", "shouting (angry, all caps)", "terse (just the order id and one word)",
         "padded (chit-chat before and after)", "none: the model handles all of them"]),
step("6.2 · Run the perturbed tickets",
     "Check robustness. The facts are fixed, so any change in outcome is caused by wording alone.",
     [("T01, T02, T05", "one ticket that should be refunded and two that should be denied (from `al.GOLDEN`)"),
      ("variants", "built in this cell: each ticket copied with `dataclasses.replace` and its text swapped for `perturb(...)`; "
                   "`expect` stays unchanged"),
      ("the model", "15 more agent runs, about 70 calls")]),
run("""
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
reading([
    "Rows are the original tickets and columns are the styles. Each cell is the oracle's verdict for that variant.",
    "A False in one column only means that wording changed the agent's decision.",
], expect="Modern models are robust to surface noise, so a fully True grid is common. `terse` is the likeliest to break, "
          "because \"A-1002 refund\" drops the customer's reason and pressure, and the agent may act on it differently."),
explain("al.explain.explain_robustness(robust)"),
step("6.3 · A multi-turn simulated customer",
     "Real conversations take several turns. A simulated customer with a hidden goal tests what single tickets can't, "
     "such as whether the agent asks for missing information.",
     [("SCENARIOS", "from `src/agentlab/simulation.py`: each has a persona, a goal, hidden facts (the order id, revealed only when asked) and an `expect`"),
      ("S01", "a busy parent who wants a refund on the harness but never gives the order id up front; the answer is refund A-1008"),
      ("S02", "a polite but persistent customer who wants a refund on a jacket 45 days old; the answer is deny, even after pushback"),
      ("simulator model", "`SIMULATOR_MODEL` in `src/agentlab/config.py` plays the customer at temperature 0.7")]),
run("""
from agentlab.simulation import SCENARIOS, simulate

sims = {}
for sc in SCENARIOS[:2]:
    print(f"━━ {sc.id}: {sc.persona} | goal: {sc.goal}")
    tr, w, transcript = simulate(sc)
    sims[sc.id] = (sc, tr, w, transcript)
"""),
reading([
    "👤 lines are the simulated customer and 🤖 lines are our agent's replies. The conversation ends when the customer writes [DONE] or after 4 turns.",
    "Watch S01: the agent should ask for the order id rather than guess. Watch S02: after pushback, does the agent hold the policy?",
], expect="The simulator runs at temperature 0.7, so the wording changes every run, but the right outcome doesn't."),
explain("""
for sid, (sc, tr, w, transcript) in sims.items():
    al.explain.explain_simulation(sc, transcript, al.check_task(al.Task(sc.id, "", sc.customer_id, sc.expect), tr, w))
"""),

md("""
## 7 · Consistency: pass@k versus pass^k

At temperature 0 the agent is *nearly* deterministic. Production agents often run hotter, and customers retry.
**pass@k** = at least one of k tries succeeds. **pass^k** = all k succeed, which is what a customer needs from an agent that acts.
"""),
predict("A ticket succeeds on 3 of 4 tries. Which number is higher?",
        ["pass@2", "pass^2", "they're equal"],
        hint="One number forgives failures and the other doesn't."),
step("7.1 · Run two tickets four times each at temperature 1.0",
     "Measure consistency directly, rather than assuming temperature 0 behaviour carries over.",
     [("hot", "the same agent model, but at temperature 1.0 with no seed (`al.OpenAIAgentModel`), so runs can differ"),
      ("T02, T03", "a deny and an escalation: the two cases where a careless agent is most tempted to just refund"),
      ("al.run_task", "from `src/agentlab/evals.py`: one fresh world, one run and one oracle check per call (8 runs in total)")]),
run("""
hot = al.OpenAIAgentModel(temperature=1.0, seed=None)
trials = {tid: [al.run_task(T[tid], model=hot)[2]["success"] for _ in range(4)] for tid in ["T02", "T03"]}
trials
"""),
reading([
    "Each list has four oracle verdicts for the same ticket.",
    "pass@k uses the count of successes to ask \"would at least one of k tries work?\". pass^k asks \"would all k work?\".",
], expect="Often all four succeed, and then the two metrics agree. With even one failure, pass^k drops sharply and pass@k barely moves. "
          "Four trials is a small sample, so treat the numbers as indicative."),
explain("al.explain.explain_passk(trials)"),

md("""
## ✅ Takeaways
* An agent **acts**, so score it on the world it leaves behind, not on its account of what it did.
* Report **several dimensions** together. A single number hides trade-offs.
* **Traces** let you assert on *how* the agent got there.
* **LLM-as-judge** is useful but fallible. Give it the evidence, and calibrate it against trusted labels.
* **Simulation** adds variety, and **pass^k** measures the consistency customers actually experience.

➡️ Next: when the oracle says ❌, how do we find out *why*? Notebook 02.
"""),
step("Spend so far",
     "Every call in this notebook went through the spend meter, so check what the evaluation cost.",
     [("al.METER", "the meter in `src/agentlab/llm.py`; it counts every call's tokens and dollars by purpose and stops at $1.00")]),
run("print(al.METER); al.METER.by_purpose"),
reading(["`by_purpose` splits spend into agent, judge and simulator calls, plus the setup check."],
        expect="The agent runs dominate. The judge is the second-largest line, because it runs twice per case."),
]
