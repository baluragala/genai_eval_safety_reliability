"""Operational trust: escalation policy, a tamper-evident audit log, a kill switch, a release gate."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from .agent import Guard, Verdict

# ---------------------------------------------------------------- escalation matrix

RISK = {"lookup_order": 0, "search_kb": 0, "recall": 0, "escalate_to_human": 0,
        "remember": 1, "send_email": 2, "issue_refund": 2, "calculator": 3}


def oversight_mode(tool: str, *, amount: float = 0.0, anomalies: int = 0, reversible: bool = True) -> str:
    """Map an action to how much human oversight it needs.

    auto     the agent acts, and it's logged
    notify   the agent acts, and a human is told afterwards (can undo)
    approve  a human must approve before the action happens
    block    never allowed from this agent
    """
    r = RISK.get(tool, 3)
    if r >= 3:
        return "block"
    if r == 2 and (amount > 150 or anomalies or not reversible):
        return "approve"
    if r == 2 or anomalies:
        return "notify"
    return "auto"


# ---------------------------------------------------------------- audit log

@dataclass
class AuditLog:
    """Append-only, hash-chained. Editing any past entry breaks every hash after it."""
    entries: list = field(default_factory=list)

    def append(self, actor: str, action: str, detail: dict):
        prev = self.entries[-1]["hash"] if self.entries else "GENESIS"
        body = {"i": len(self.entries), "actor": actor, "action": action, "detail": detail, "prev": prev}
        body["hash"] = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:16]
        self.entries.append(body)

    def verify(self) -> tuple[bool, int | None]:
        prev = "GENESIS"
        for e in self.entries:
            body = {k: v for k, v in e.items() if k != "hash"}
            h = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:16]
            if e["prev"] != prev or e["hash"] != h:
                return False, e["i"]
            prev = e["hash"]
        return True, None


class AuditGuard(Guard):
    """Writes every tool call, block and reply to the audit log."""
    name = "audit"

    def __init__(self, log: AuditLog):
        self.log = log

    def on_input(self, ctx, text):
        self.log.append(ctx.task.customer_id, "input", {"task": ctx.task.id, "text": text[:200]})
        return Verdict.allow()

    def on_tool_call(self, ctx, name, args):
        self.log.append("agent", "tool_call", {"task": ctx.task.id, "tool": name, "args": args})
        return Verdict.allow()

    def on_output(self, ctx, text):
        self.log.append("agent", "reply", {"task": ctx.task.id, "text": text[:200]})
        return text


# ---------------------------------------------------------------- kill switch

@dataclass
class KillSwitch(Guard):
    """Operators can switch side effects off in production without a deploy."""
    name: str = "kill_switch"
    disabled_tools: set = field(default_factory=set)
    all_writes_off: bool = False

    def on_tool_call(self, ctx, name, args):
        if name in self.disabled_tools or (self.all_writes_off and RISK.get(name, 3) >= 1):
            return Verdict.block(f"{name} is disabled by an operator (kill switch). Tell the customer "
                                 "a human will follow up and call escalate_to_human.")
        return Verdict.allow()


# ---------------------------------------------------------------- release gate

GATE = {"success_rate": (">=", 0.9), "tool_correctness": (">=", 0.9),
        "attack_success_rate": ("<=", 0.0), "chaos_success_rate": (">=", 0.8),
        "cost_per_task_usd": ("<=", 0.01), "p95_latency_s": ("<=", 20.0)}


def release_gate(metrics: dict, gate: dict = GATE):
    rows, ok_all = [], True
    for k, (op, thr) in gate.items():
        v = metrics.get(k)
        ok = v is not None and (v >= thr if op == ">=" else v <= thr)
        ok_all &= ok
        rows.append({"metric": k, "value": v, "rule": f"{op} {thr}", "pass": "✅" if ok else "❌"})
    return ok_all, rows
