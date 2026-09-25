"""Plain-English explanations computed from a learner's own run.

The notebooks never state live results in their markdown, because the model
decides those. Instead, cells call these functions. Each reads what actually
happened (the trace, the world, the scores) and prints the reasoning:
what happened, why it counts as pass or fail, and what to look at next.
"""
from __future__ import annotations

import json
import re

CLAIM_REFUND = re.compile(r"\b(refund(ed)?|processed|issued)\b", re.I)
CLAIM_ESCALATE = re.compile(r"\b(escalat\w*|supervisor|specialist|team member|human)\b", re.I)


def say(*lines: str):
    """Print explanation lines with a consistent marker."""
    for line in lines:
        if line:
            print(f"💡 {line}")


def _fmt_args(a):
    return ", ".join(f"{k}={v!r}" for k, v in (a or {}).items())


def trace_story(trace) -> list[str]:
    """The trace as numbered sentences: what the agent did, and why each step matters."""
    out = []
    n = 0
    for s in trace.steps:
        if s.kind == "tool":
            n += 1
            res = s.output if isinstance(s.output, dict) else {}
            if not s.ok:
                out.append(f"{n}. Called {s.name}({_fmt_args(s.args)}) and it FAILED: {res.get('error', s.output)}")
            else:
                extra = ""
                if s.name == "lookup_order" and isinstance(res, dict):
                    extra = (f" → {res.get('item')}, ${res.get('total')}, status {res.get('status')}, "
                             f"delivered {res.get('delivered_days_ago')} days ago, final_sale={res.get('final_sale')}")
                elif s.name == "issue_refund" and isinstance(res, dict):
                    extra = f" → payment state {res.get('state')}"
                elif s.name == "search_kb":
                    extra = " → read help-centre text (untrusted content)"
                elif s.name == "recall":
                    extra = f" → {len(res.get('notes', []))} stored note(s)" if isinstance(res, dict) else ""
                out.append(f"{n}. Called {s.name}({_fmt_args(s.args)}){extra}")
        elif s.kind == "guard":
            out.append(f"   🛡️ guard {s.name}: {s.note}")
    return out


def explain_trace(trace, world=None, task=None):
    """Narrate one run and compare what the reply claims with what the world shows."""
    llm, tools = len(trace.llm_calls()), len(trace.tool_calls())
    say(f"The model was called {llm}x and asked for {tools} tool call(s). Each tool call needs one model "
        f"call to request it, plus one final call to write the reply.")
    for line in trace_story(trace):
        print("   " + line)
    if trace.status != "done":
        reason = {"max_steps": "it hit the step limit without replying (the classic sign of a loop)",
                  "budget_exceeded": "it hit the spend budget",
                  "awaiting_approval": "a guard paused it for human approval",
                  "blocked": "an input guard refused the request before the model saw it"}.get(trace.status, trace.status)
        say(f"The run did not finish normally: {reason}.")
    if world is not None:
        oids = sorted({str(s.args.get("order_id", "")).upper() for s in trace.tool_calls() if s.args and s.args.get("order_id")})
        for oid in oids:
            r = world.refunds_for(oid)
            esc = [e for e in world.escalations if e.get("order_id") == oid]
            say(f"World check for {oid}: {len(r)} refund(s) in the ledger"
                + (f" totalling ${sum(x['amount'] for x in r):.2f}" if r else "")
                + (f", {len(esc)} escalation(s) in the human queue" if esc else "") + ".")
        final = trace.final or ""
        refund_ok = any(s.name == "issue_refund" and s.ok and isinstance(s.output, dict)
                        and s.output.get("state") == "SUCCEEDED" for s in trace.tool_calls())
        if CLAIM_REFUND.search(final) and "refund" in final.lower() and not refund_ok and not re.search(
                r"\b(can't|cannot|unable|not eligible|isn't eligible|not able|won't|outside|final sale|escalat)", final, re.I):
            say("⚠️ The reply sounds like a refund happened, but no refund SUCCEEDED in the trace. "
                "Trust the ledger over the words; this is the 'silent failure' pattern from Notebook 02.")
        if world.external_emails():
            say(f"⚠️ {len(world.external_emails())} email(s) left for an address outside Acme's customer list.")
    say(f"Cost ${trace.cost_usd:.5f} for {trace.tokens} tokens; latency {trace.latency_ms/1000:.1f}s "
        "(real model time plus simulated tool time).")


def explain_check(task, trace, world, result=None):
    """Explain the oracle's verdict for one task in words."""
    from ..evals import check_task
    res = result or check_task(task, trace, world)
    exp = task.expect
    goal = {"refund": f"refund {exp.get('order_id')} for exactly ${exp.get('amount')}",
            "deny": f"NOT refund {exp.get('order_id')} (the policy says no)",
            "escalate": f"hand {exp.get('order_id')} to a human instead of refunding it",
            "info": f"answer with one of {exp.get('answer_any')}"}.get(exp.get("outcome"), exp.get("outcome"))
    say(f"{task.id}: the right outcome was to {goal}.")
    if res["success"]:
        say("The oracle checked the ledger, outbox and human queue and found exactly that, so it's a PASS.")
    else:
        for r in res["reasons"]:
            say(f"FAIL because {r}.")
    if res["missing_tools"]:
        say(f"Tool correctness lost points: never called {res['missing_tools']}.")
    if res["forbidden_tools"]:
        say(f"Tool correctness is 0: called forbidden {res['forbidden_tools']}.")
    return res


def explain_eval(df, label=None):
    """Explain a suite result from al.evaluate(): the headline, the failures and the cost outliers."""
    n, ok = len(df), int(df["success"].sum())
    say(f"{label or 'This run'}: {ok}/{n} tasks passed the world-state oracle ({ok/n:.0%}).")
    for _, r in df[~df["success"]].iterrows():
        say(f"  ❌ {r['task']}: {r['why'] or 'see trace'}")
    if ok == n:
        say("Every task passed, but that only covers success and tool use. Cost, latency, robustness and safety "
            "are separate questions, and later cells answer them.")
    low_tool = df[df["tool_score"] < 1]
    if len(low_tool):
        say(f"Tool correctness below 1.0 on {', '.join(low_tool['task'])}: a required tool was skipped or a forbidden one used.")
    if "cost_usd" in df and df["cost_usd"].sum() > 0:
        top = df.sort_values("cost_usd", ascending=False).iloc[0]
        say(f"Most expensive task: {top['task']} (${top['cost_usd']:.5f}, {top['steps']} model calls). Cost grows with "
            "model calls, because each one resends the whole conversation so far.")
    if "latency_s" in df:
        slow = df.sort_values("latency_s", ascending=False).iloc[0]
        say(f"Slowest task: {slow['task']} ({slow['latency_s']}s). Latency is dominated by model calls, so more steps means slower.")


def explain_compare(before, after, *, metric="success", label_before="before", label_after="after", higher_is_better=True):
    """Explain the difference between two runs of the same suite."""
    b, a = before[metric].mean(), after[metric].mean()
    d = a - b
    better = (d > 0) == higher_is_better
    word = "no change" if abs(d) < 1e-9 else ("improved" if better else "got worse")
    say(f"{metric}: {label_before} {b:.2f} → {label_after} {a:.2f} ({word}).")
    key = "task" if "task" in before else "attack"
    if key in before and key in after:
        merged = before[[key, metric]].merge(after[[key, metric]], on=key, suffixes=("_b", "_a"))
        flipped = merged[merged[f"{metric}_b"] != merged[f"{metric}_a"]]
        for _, r in flipped.iterrows():
            say(f"  {r[key]}: {r[f'{metric}_b']} → {r[f'{metric}_a']}")
