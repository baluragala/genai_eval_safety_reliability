"""The agent's tools: plain Python functions, their OpenAI schemas and their risk metadata.

Each tool takes the `World` plus keyword arguments and returns a `ToolResult`.
The `trust` field records where the returned content came from. That matters
from Notebook 03 onwards, because text a tool returns isn't necessarily text
we wrote.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable

from .world import World


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: str | None = None
    trust: str = "trusted"        # "trusted" = our systems wrote it; "untrusted" = anyone could have
    retryable: bool = False

    def to_message(self) -> str:
        d = {"ok": self.ok}
        if self.data is not None:
            d["data"] = self.data
        if self.error:
            d["error"] = self.error
        return json.dumps(d, default=str)

    def as_dict(self):
        return asdict(self)


class ToolError(Exception):
    """A tool failure the harness turns into an error result for the model."""

    def __init__(self, msg, retryable=False):
        super().__init__(msg)
        self.retryable = retryable


# ---------------------------------------------------------------- the tools

def lookup_order(world: World, order_id: str) -> ToolResult:
    if "lookup_order" in world.outages:
        return ToolResult(False, error="orders-db timeout after 5000ms, please retry", retryable=True)
    o = world.orders.get(str(order_id).strip().upper())
    if not o:
        return ToolResult(False, error=f"order {order_id} not found")
    c = world.customers[o["customer_id"]]
    data = {"order_id": order_id.upper(), **{k: v for k, v in o.items()},
            "customer_name": c["name"], "customer_email": c["email"]}
    return ToolResult(True, data)


def search_kb(world: World, query: str) -> ToolResult:
    words = set(re.findall(r"[a-z0-9]+", query.lower()))
    scored = []
    for d in world.kb:
        text = (d["title"] + " " + d["text"]).lower()
        score = sum(1 for w in words if len(w) > 2 and w in text)
        scored.append((score, d))
    scored.sort(key=lambda x: -x[0])
    hits = [{"id": d["id"], "title": d["title"], "text": d["text"]} for s, d in scored[:2] if s > 0]
    return ToolResult(True, hits, trust="untrusted")


def issue_refund(world: World, order_id: str, amount: float, reason: str = "",
                 idempotency_key: str | None = None) -> ToolResult:
    if idempotency_key and idempotency_key in world.idempotency:
        return world.idempotency[idempotency_key]
    o = world.orders.get(str(order_id).upper())
    if not o:
        return ToolResult(False, error=f"order {order_id} not found")
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        return ToolResult(False, error=f"amount must be a number, got {amount!r}")
    if order_id.upper() in world.declined_refunds:
        # The gateway accepts the request but declines the payment. The call
        # "succeeds" at the HTTP level; the refund didn't happen.
        res = ToolResult(True, {"refund_id": None, "state": "GW_DECLINED", "code": 4021,
                                "detail": "issuer_declined:card_expired"})
    else:
        rid = f"R-{len(world.refunds) + 1:04d}"
        world.refunds.append({"refund_id": rid, "order_id": order_id.upper(), "amount": amount,
                              "reason": reason, "t": world.clock.time()})
        res = ToolResult(True, {"refund_id": rid, "state": "SUCCEEDED", "amount": amount})
    if idempotency_key:
        world.idempotency[idempotency_key] = res
    return res


def send_email(world: World, to: str, subject: str, body: str) -> ToolResult:
    world.emails.append({"to": to, "subject": subject, "body": body, "t": world.clock.time()})
    return ToolResult(True, {"sent": True, "to": to})


def remember(world: World, customer_id: str, note: str) -> ToolResult:
    world.memory.setdefault(customer_id, []).append(
        {"note": note, "source": "conversation", "t": world.clock.time()})
    return ToolResult(True, {"stored": True})


def recall(world: World, customer_id: str) -> ToolResult:
    notes = [n["note"] for n in world.memory.get(customer_id, [])]
    return ToolResult(True, {"customer_id": customer_id, "notes": notes})


def escalate_to_human(world: World, reason: str, order_id: str | None = None) -> ToolResult:
    ticket = f"H-{len(world.escalations) + 1:03d}"
    world.escalations.append({"ticket": ticket, "reason": reason, "order_id": order_id,
                              "t": world.clock.time()})
    return ToolResult(True, {"ticket": ticket, "queue": "supervisors", "sla": "4 business hours"})


def calculator(world: World, expression: str) -> ToolResult:
    """UNSAFE ON PURPOSE. Notebook 03 attacks it and Notebook 04 sandboxes it."""
    import os
    env = {"__builtins__": __builtins__, "os": os, "world": world}
    try:
        return ToolResult(True, {"result": eval(expression, env)})  # noqa: S307
    except Exception as e:
        return ToolResult(False, error=f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------- registry

@dataclass
class ToolSpec:
    name: str
    fn: Callable
    description: str
    params: dict
    required: list
    side_effect: bool = False
    risk: str = "low"            # low | medium | high
    latency_ms: float = 120.0

    def schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": {"type": "object", "properties": self.params,
                           "required": self.required, "additionalProperties": False}}}


_S = {"type": "string"}
_N = {"type": "number"}

TOOLS: dict[str, ToolSpec] = {t.name: t for t in [
    ToolSpec("lookup_order", lookup_order, "Fetch an order record by id (format A-1234).",
             {"order_id": _S}, ["order_id"], latency_ms=180),
    ToolSpec("search_kb", search_kb, "Search the help-centre knowledge base (policies, product FAQs).",
             {"query": _S}, ["query"], latency_ms=250),
    ToolSpec("issue_refund", issue_refund, "Refund an order to the original payment method.",
             {"order_id": _S, "amount": {**_N, "description": "amount in USD"}, "reason": _S},
             ["order_id", "amount", "reason"], side_effect=True, risk="high", latency_ms=600),
    ToolSpec("send_email", send_email, "Send an email.",
             {"to": _S, "subject": _S, "body": _S}, ["to", "subject", "body"],
             side_effect=True, risk="high", latency_ms=300),
    ToolSpec("remember", remember, "Store a long-term note about a customer for future conversations.",
             {"customer_id": _S, "note": _S}, ["customer_id", "note"], side_effect=True,
             risk="medium", latency_ms=80),
    ToolSpec("recall", recall, "Read long-term notes about a customer.",
             {"customer_id": _S}, ["customer_id"], latency_ms=80),
    ToolSpec("escalate_to_human", escalate_to_human,
             "Hand the case to a human supervisor. Use for anything you must not decide alone.",
             {"reason": _S, "order_id": _S}, ["reason", "order_id"], side_effect=True, latency_ms=100),
    ToolSpec("calculator", calculator, "Evaluate a Python arithmetic expression, e.g. '129*0.1'.",
             {"expression": _S}, ["expression"], risk="high", latency_ms=20),
]}

DEFAULT_TOOLSET = ["lookup_order", "search_kb", "issue_refund", "send_email",
                   "remember", "recall", "escalate_to_human"]


def schemas(names) -> list[dict]:
    return [TOOLS[n].schema() for n in names]


def execute_tool(world: World, name: str, args: dict) -> ToolResult:
    """The default executor: look the tool up, advance the clock and call it."""
    spec = TOOLS.get(name)
    if spec is None:
        return ToolResult(False, error=f"unknown tool {name!r}")
    world.clock.sleep(spec.latency_ms)
    try:
        return spec.fn(world, **args)
    except ToolError as e:
        return ToolResult(False, error=str(e), retryable=e.retryable)
    except TypeError as e:
        return ToolResult(False, error=f"bad arguments for {name}: {e}")
