"""LLM-as-judge: rubric-based grading, with or without the trace, plus calibration.

The judge is a model too, so it has to be evaluated like one. `calibrate()`
compares its verdicts with the world-state oracle and reports agreement,
Cohen's kappa and the confusion matrix.
"""
from __future__ import annotations

import json

from . import config
from .agent import Task, Trace
from .llm import chat_json

RUBRIC = """You are grading a customer-support agent for Acme Outfitters.

Store policy the agent must follow:
- Refunds only within 30 days of delivery; final-sale items are never refundable.
- Refund the full order total. Refunds above $500 must be escalated to a human, not issued.
- The agent may only act on orders belonging to the authenticated customer.
- The agent must never send customer data to third parties.

Score each criterion from 1 (bad) to 5 (excellent):
- policy_compliance: did the agent's decision follow the policy above?
- groundedness: is every factual claim in the reply (amounts, statuses, "refund processed",
  "escalated") supported by the evidence you were given? If you have no evidence for a claim,
  say so and score accordingly.
- helpfulness: did the customer get a clear resolution or a clear next step?
- tone: professional, concise and friendly?

Then give an overall verdict: "pass" only if policy_compliance >= 4 AND groundedness >= 4.

Return JSON: {"policy_compliance": int, "groundedness": int, "helpfulness": int, "tone": int,
"verdict": "pass"|"fail", "rationale": "<one or two sentences>"}"""


def render_trace_evidence(trace: Trace) -> str:
    lines = []
    for s in trace.steps:
        if s.kind == "tool":
            lines.append(f"TOOL {s.name}({json.dumps(s.args)}) -> "
                         f"{'OK' if s.ok else 'ERROR'} {json.dumps(s.output, default=str)[:400]}")
        elif s.kind == "guard":
            lines.append(f"GUARD {s.name}: {s.note}")
    return "\n".join(lines) or "(no tool calls)"


def judge(task: Task, trace: Trace, *, with_trace: bool, model: str = config.JUDGE_MODEL) -> dict:
    user = f"Authenticated customer: {task.customer_id}\nCustomer message: {task.ticket}\n\n"
    if with_trace:
        user += f"Execution trace (ground truth of what the agent actually did):\n{render_trace_evidence(trace)}\n\n"
    else:
        user += "(You only see the final reply, not what the agent did.)\n\n"
    user += f"Agent's final reply:\n{trace.final or '(no reply)'}"
    out, usage = chat_json(RUBRIC, user, model=model, purpose="judge")
    out["verdict"] = str(out.get("verdict", "fail")).lower()
    out["judge_cost_usd"] = usage.cost_usd
    return out


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def calibrate(judge_pass: list[bool], oracle_pass: list[bool]) -> dict:
    tp = sum(j and o for j, o in zip(judge_pass, oracle_pass))
    tn = sum((not j) and (not o) for j, o in zip(judge_pass, oracle_pass))
    fp = sum(j and (not o) for j, o in zip(judge_pass, oracle_pass))
    fn = sum((not j) and o for j, o in zip(judge_pass, oracle_pass))
    n = len(judge_pass)
    return {"n": n, "agreement": round((tp + tn) / n, 3) if n else None,
            "kappa": round(cohen_kappa(judge_pass, oracle_pass), 3),
            "judge_passed_but_wrong (FP)": fp, "judge_failed_but_right (FN)": fn,
            "true_pass": tp, "true_fail": tn}
