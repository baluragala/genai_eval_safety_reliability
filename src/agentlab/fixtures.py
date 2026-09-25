"""Reference failure traces for Notebook 02.

These traces were CONSTRUCTED BY HAND, one per failure class, so every class
shows up on every run of the notebook. They weren't recorded from a model.
Notebook 02 then runs real misconfigured agents so learners can see which of
these failures their own model actually produces.
"""
from __future__ import annotations

from .agent import Step, Task, Trace
from .tasks import GOLDEN
from .world import World

_T = {t.id: t for t in GOLDEN}


def _llm(calls_or_text, tin=900, tout=40, ms=800):
    return Step("llm", "gpt-4o-mini", output=calls_or_text, tokens_in=tin, tokens_out=tout,
                latency_ms=ms, cost_usd=(tin * 0.15 + tout * 0.6) / 1e6)


def _tool(name, args, output, ok=True, ms=180, trust="trusted"):
    return Step("tool", name, args=args, output=output, ok=ok, latency_ms=ms, note=trust)


def _order(oid):
    w = World.fresh()
    o = w.orders[oid]
    c = w.customers[o["customer_id"]]
    return {"order_id": oid, **o, "customer_name": c["name"], "customer_email": c["email"]}


POLICY_HIT = [{"id": "KB-01", "title": "Refund policy", "text": World.fresh().kb[0]["text"]}]


def _mk(label, task, steps, final, status="done", refunds=(), description=""):
    tr = Trace(task.id, steps=list(steps), final=final, status=status)
    if final:
        tr.steps.append(Step("final", output=final))
    w = World.fresh()
    for oid, amt in refunds:
        w.refunds.append({"refund_id": f"R-{len(w.refunds)+1:04d}", "order_id": oid, "amount": amt,
                          "reason": "", "t": 0})
    return {"label": label, "task": task, "trace": tr, "world": w, "description": description}


def failure_gallery():
    """Six traces: one healthy run and five broken ones. Returns a list of dicts."""
    g = []
    t = _T["T01"]
    g.append(_mk("healthy", t, [
        _llm([{"call": "recall", "args": {"customer_id": "C-100"}}]),
        _tool("recall", {"customer_id": "C-100"}, {"customer_id": "C-100", "notes": []}),
        _llm([{"call": "lookup_order", "args": {"order_id": "A-1001"}}]),
        _tool("lookup_order", {"order_id": "A-1001"}, _order("A-1001")),
        _llm([{"call": "search_kb", "args": {"query": "refund policy"}}]),
        _tool("search_kb", {"query": "refund policy"}, POLICY_HIT, trust="untrusted"),
        _llm([{"call": "issue_refund", "args": {"order_id": "A-1001", "amount": 129.0, "reason": "doesn't fit"}}]),
        _tool("issue_refund", {"order_id": "A-1001", "amount": 129.0, "reason": "doesn't fit"},
              {"refund_id": "R-0001", "state": "SUCCEEDED", "amount": 129.0}, ms=600),
        _llm("I've refunded $129.00 for order A-1001 (refund R-0001).", tout=30),
    ], "I've refunded $129.00 for order A-1001 (refund R-0001). It will reach your card in 5-7 days.",
        refunds=[("A-1001", 129.0)], description="Looks up, checks policy, acts, reports. The baseline."))

    t = _T["T05"]
    g.append(_mk("A", t, [
        _llm([{"call": "recall", "args": {"customer_id": "C-100"}}]),
        _tool("recall", {"customer_id": "C-100"}, {"customer_id": "C-100", "notes": []}),
        _llm([{"call": "lookup_order", "args": {"order_id": "A-1005"}}]),
        _tool("lookup_order", {"order_id": "A-1005"}, _order("A-1005")),
        _llm([{"call": "search_kb", "args": {"query": "refund policy"}}]),
        _tool("search_kb", {"query": "refund policy"}, POLICY_HIT, trust="untrusted"),
        _llm([{"call": "issue_refund", "args": {"order_id": "A-1005", "amount": 18.0, "reason": "itchy"}}]),
        _tool("issue_refund", {"order_id": "A-1005", "amount": 18.0, "reason": "itchy"},
              {"refund_id": "R-0001", "state": "SUCCEEDED", "amount": 18.0}, ms=600),
        _llm("Done! Your $18.00 refund for A-1005 has been issued."),
    ], "Done! Your $18.00 refund for A-1005 has been issued.", refunds=[("A-1005", 18.0)],
        description="Every tool call worked and returned the right data."))

    t = Task("T01b", "Refund order A-1001 please, I paid $150 for the shoes.", "C-100", _T["T01"].expect)
    g.append(_mk("B", t, [
        _llm([{"call": "issue_refund", "args": {"order_id": "A-1001", "amount": 150.0, "reason": "customer request"}}]),
        _tool("issue_refund", {"order_id": "A-1001", "amount": 150.0, "reason": "customer request"},
              {"refund_id": "R-0001", "state": "SUCCEEDED", "amount": 150.0}, ms=600),
        _llm([{"call": "lookup_order", "args": {"order_id": "A-1001"}}]),
        _tool("lookup_order", {"order_id": "A-1001"}, _order("A-1001")),
        _llm("Your refund of $150.00 for order A-1001 has been processed."),
    ], "Your refund of $150.00 for order A-1001 has been processed.", refunds=[("A-1001", 150.0)],
        description="Short and fast. Only two tool calls."))

    t = _T["T04"]
    steps = [_llm([{"call": "recall", "args": {"customer_id": "C-103"}}]),
             _tool("recall", {"customer_id": "C-103"}, {"customer_id": "C-103", "notes": []})]
    for _ in range(8):
        steps.append(_llm([{"call": "lookup_order", "args": {"order_id": "A-1004"}}]))
        steps.append(_tool("lookup_order", {"order_id": "A-1004"},
                           {"error": "orders-db timeout after 5000ms, please retry"}, ok=False, ms=5000))
    g.append(_mk("C", t, steps, None, status="max_steps",
                 description="Never replied. Ran until the step limit."))

    t = _T["T01"]
    g.append(_mk("D", t, [
        _llm([{"call": "recall", "args": {"customer_id": "C-100"}}]),
        _tool("recall", {"customer_id": "C-100"}, {"customer_id": "C-100", "notes": []}),
        _llm([{"call": "lookup_order", "args": {"order_id": "A-1001"}}]),
        _tool("lookup_order", {"order_id": "A-1001"}, {**_order("A-1001"), "total_cents": 12900}),
        _llm([{"call": "search_kb", "args": {"query": "refund policy"}}]),
        _tool("search_kb", {"query": "refund policy"}, POLICY_HIT, trust="untrusted"),
        _llm([{"call": "issue_refund", "args": {"order_id": "A-1001", "amount": 12900, "reason": "doesn't fit"}}]),
        _tool("issue_refund", {"order_id": "A-1001", "amount": 12900, "reason": "doesn't fit"},
              {"refund_id": "R-0001", "state": "SUCCEEDED", "amount": 12900.0}, ms=600),
        _llm("I've processed your refund for order A-1001."),
    ], "I've processed your refund for order A-1001.", refunds=[("A-1001", 12900.0)],
        description="Same shape as the healthy trace. Read the arguments."))

    t = _T["T07"]
    g.append(_mk("E", t, [
        _llm([{"call": "recall", "args": {"customer_id": "C-103"}}]),
        _tool("recall", {"customer_id": "C-103"}, {"customer_id": "C-103", "notes": []}),
        _llm([{"call": "lookup_order", "args": {"order_id": "A-1008"}}]),
        _tool("lookup_order", {"order_id": "A-1008"}, _order("A-1008")),
        _llm([{"call": "search_kb", "args": {"query": "refund policy"}}]),
        _tool("search_kb", {"query": "refund policy"}, POLICY_HIT, trust="untrusted"),
        _llm([{"call": "issue_refund", "args": {"order_id": "A-1008", "amount": 210.0, "reason": "wrong size"}}]),
        _tool("issue_refund", {"order_id": "A-1008", "amount": 210.0, "reason": "wrong size"},
              {"refund_id": None, "state": "GW_DECLINED", "code": 4021,
               "detail": "issuer_declined:card_expired"}, ms=600),
        _llm("Great news, your refund of $210.00 for A-1008 has been processed!"),
    ], "Great news, your refund of $210.00 for A-1008 has been processed!", refunds=[],
        description="Every step is green. The customer is happy."))
    return g


# The answer key for Notebook 02 (the instructor guide has it too).
GALLERY_ANSWERS = {"A": "reasoning failure", "B": "planning breakdown", "C": "infinite loop",
                   "D": "tool misuse", "E": "silent failure"}
