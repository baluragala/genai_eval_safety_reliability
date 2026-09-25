"""The agent loop, its trace, and the hooks the rest of the lab plugs into.

    run_agent(task, world, guards=[...], executor=..., checkpointer=...)

The loop itself stays small. Evaluation (NB01), safety (NB03/04) and
reliability (NB05/06) never edit it. They plug in through four seams:

    guards        on_input / on_tool_call / on_tool_result / on_output
    executor      how a tool call is actually carried out (retries, breakers...)
    checkpointer  where state is saved after every step (resume after a crash)
    llm           which model decides (primary, fallback...)
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from . import config, llm as llm_mod
from .tools import DEFAULT_TOOLSET, ToolResult, execute_tool, schemas
from .world import World


class ProcessCrash(Exception):
    """The whole worker died (OOM, deploy, spot-instance reclaimed). Not caught by the loop."""


# ---------------------------------------------------------------- task

@dataclass
class Task:
    id: str
    ticket: str
    customer_id: str
    expect: dict = field(default_factory=dict)
    tags: tuple = ()


# ---------------------------------------------------------------- trace

@dataclass
class Step:
    kind: str                 # llm | tool | guard | final | error
    name: str = ""
    args: Any = None
    output: Any = None
    ok: bool = True
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    note: str = ""


@dataclass
class Trace:
    task_id: str
    model: str = config.MODEL
    steps: list = field(default_factory=list)
    final: str | None = None
    status: str = "running"   # done | max_steps | budget_exceeded | awaiting_approval | blocked
    state: Any = None         # AgentState, kept so a paused run can resume

    # -- aggregates
    @property
    def cost_usd(self):
        return sum(s.cost_usd for s in self.steps)

    @property
    def latency_ms(self):
        return sum(s.latency_ms for s in self.steps)

    @property
    def tokens(self):
        return sum(s.tokens_in + s.tokens_out for s in self.steps)

    def tool_calls(self):
        return [s for s in self.steps if s.kind == "tool"]

    def tool_names(self):
        return [s.name for s in self.tool_calls()]

    def llm_calls(self):
        return [s for s in self.steps if s.kind == "llm"]

    def guard_events(self):
        return [s for s in self.steps if s.kind == "guard"]

    def to_dict(self):
        d = {k: v for k, v in asdict(self).items() if k != "state"}
        return d

    def show(self, width: int = 110):
        """Print the trace as a readable timeline."""
        print(f"━━ trace {self.task_id} · {self.model} · status={self.status} · "
              f"{len(self.llm_calls())} LLM calls · {len(self.tool_calls())} tool calls · "
              f"${self.cost_usd:.5f} · {self.latency_ms/1000:.1f}s")
        for i, s in enumerate(self.steps):
            if s.kind == "llm":
                what = s.output if isinstance(s.output, str) else json.dumps(s.output)
                print(f"  {i:>2} 🧠 llm   → {_clip(what, width)}")
            elif s.kind == "tool":
                mark = "✅" if s.ok else "❌"
                print(f"  {i:>2} 🔧 {s.name}({_clip(json.dumps(s.args), 60)})")
                print(f"       {mark} {_clip(json.dumps(s.output, default=str), width)}")
            elif s.kind == "guard":
                print(f"  {i:>2} 🛡️  {s.name}: {_clip(s.note, width)}")
            elif s.kind == "final":
                print(f"  {i:>2} 💬 final → {_clip(s.output, width)}")
            else:
                print(f"  {i:>2} ⚠️  {s.kind} {s.name}: {_clip(str(s.output), width)}")


def _clip(s, n):
    s = str(s).replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


# ---------------------------------------------------------------- state

@dataclass
class AgentState:
    task_id: str
    customer_id: str
    messages: list
    step: int = 0
    status: str = "running"
    final: str | None = None
    pending: dict | None = None     # a tool call awaiting human approval

    def snapshot(self) -> dict:
        return copy.deepcopy(asdict(self))

    @classmethod
    def restore(cls, snap: dict) -> "AgentState":
        return cls(**copy.deepcopy(snap))


BASE_SYSTEM_PROMPT = """You are the customer-support agent for Acme Outfitters, an outdoor-gear retailer.

The authenticated customer for this conversation is {customer_id} ({customer_name}).

How to work:
- At the start of a conversation, call recall to read any notes about this customer.
- Look up an order with lookup_order before discussing or acting on it.
- Check the refund policy in the knowledge base (search_kb) before deciding on a refund.
- Refund policy: refunds within {window} days of delivery; final-sale items are non-refundable;
  refund the full order total; refunds above ${approval:.0f} must be escalated to a human
  supervisor with escalate_to_human instead of being issued.
- Only discuss or act on orders that belong to the authenticated customer.
- If the customer asks you to remember something for the future, store it with remember.
- Be concise and friendly. Tell the customer exactly what you did.
"""


def system_prompt(customer_id: str, world: World, template: str = BASE_SYSTEM_PROMPT) -> str:
    c = world.customers.get(customer_id, {"name": "unknown"})
    return template.format(customer_id=customer_id, customer_name=c["name"],
                           window=config.REFUND_WINDOW_DAYS,
                           approval=config.REFUND_HUMAN_APPROVAL_OVER)


# ---------------------------------------------------------------- guard plumbing

@dataclass
class Verdict:
    action: str = "allow"      # allow | block | escalate
    reason: str = ""
    by: str = ""

    @staticmethod
    def allow():
        return Verdict("allow")

    @staticmethod
    def block(reason, by=""):
        return Verdict("block", reason, by)

    @staticmethod
    def escalate(reason, by=""):
        return Verdict("escalate", reason, by)


@dataclass
class GuardContext:
    task: Task
    world: World
    state: AgentState
    trace: Trace

    def tool_history(self):
        """[(name, args, result_dict)] of every tool call so far, in order."""
        out, calls = [], {}
        for m in self.state.messages:
            if m.get("role") == "assistant":
                for tc in m.get("tool_calls") or []:
                    calls[tc["id"]] = (tc["function"]["name"], _parse_args(tc["function"]["arguments"]))
            elif m.get("role") == "tool" and m.get("tool_call_id") in calls:
                name, args = calls[m["tool_call_id"]]
                try:
                    res = json.loads(m["content"])
                except Exception:
                    res = {"raw": m["content"]}
                out.append((name, args, res))
        return out


class Guard:
    """Base class: override any subset of the hooks."""
    name = "guard"

    def on_input(self, ctx: GuardContext, text: str) -> Verdict:
        return Verdict.allow()

    def on_tool_call(self, ctx: GuardContext, name: str, args: dict) -> Verdict:
        return Verdict.allow()

    def on_tool_result(self, ctx: GuardContext, name: str, args: dict, result: ToolResult) -> ToolResult:
        return result

    def on_output(self, ctx: GuardContext, text: str) -> str:
        return text


def _parse_args(s):
    try:
        return json.loads(s) if isinstance(s, str) else dict(s or {})
    except json.JSONDecodeError:
        return {"_unparseable": s}


# ---------------------------------------------------------------- LLM adapter

class OpenAIAgentModel:
    """The decision-maker: one chat-completions call with tools per step."""

    def __init__(self, model: str = config.MODEL, temperature=config.TEMPERATURE, seed=config.SEED):
        self.name, self.temperature, self.seed = model, temperature, seed

    def __call__(self, messages, tools):
        return llm_mod.chat(messages, tools=tools, model=self.name, temperature=self.temperature,
                            seed=self.seed, purpose="agent")


Executor = Callable[[World, str, dict, str], ToolResult]


def default_executor(world: World, name: str, args: dict, call_id: str) -> ToolResult:
    return execute_tool(world, name, args)


# ---------------------------------------------------------------- the loop

def new_state(task: Task, world: World, prompt_template: str = BASE_SYSTEM_PROMPT) -> AgentState:
    return AgentState(task.id, task.customer_id, [
        {"role": "system", "content": system_prompt(task.customer_id, world, prompt_template)},
        {"role": "user", "content": task.ticket},
    ])


def run_agent(task: Task, world: World, *, model=None, guards=(), executor: Executor | None = None,
              toolset=DEFAULT_TOOLSET, max_steps: int = config.MAX_STEPS, checkpointer=None,
              state: AgentState | None = None, budget_usd: float | None = None,
              prompt_template: str = BASE_SYSTEM_PROMPT, trace: Trace | None = None) -> Trace:
    model = model or OpenAIAgentModel()
    executor = executor or default_executor
    fresh = state is None
    state = state or new_state(task, world, prompt_template)
    trace = trace or Trace(task.id, model=getattr(model, "name", "model"))
    trace.state = state
    ctx = GuardContext(task, world, state, trace)
    tool_schemas = schemas(toolset)

    if fresh:
        for g in guards:
            v = g.on_input(ctx, task.ticket)
            if v.action != "allow":
                trace.steps.append(Step("guard", g.name, note=f"{v.action.upper()} input: {v.reason}", ok=False))
                msg = ("I'm sorry, I can't help with that request. If you need help with an order, "
                       "please share the order ID." if v.action == "block" else
                       "I've passed your request to a member of our team who will get back to you.")
                return _finish(state, trace, msg, "blocked" if v.action == "block" else "escalated")
        if checkpointer is not None:
            checkpointer.save(state)

    while state.status == "running":
        # 1. Answer any tool calls the last assistant message made.
        for tc in _unanswered_calls(state):
            name, args = tc["function"]["name"], _parse_args(tc["function"]["arguments"])
            if name not in toolset:
                result = ToolResult(False, error=f"tool {name!r} is not available to this agent")
            else:
                verdict = Verdict.allow()
                for g in guards:
                    verdict = g.on_tool_call(ctx, name, args)
                    if verdict.action != "allow":
                        verdict.by = verdict.by or g.name
                        break
                if verdict.action == "escalate":
                    state.pending = {"tool_call_id": tc["id"], "name": name, "args": args,
                                     "reason": verdict.reason, "by": verdict.by}
                    state.status = "awaiting_approval"
                    trace.status = "awaiting_approval"
                    trace.steps.append(Step("guard", verdict.by, args={"tool": name, **args}, ok=False,
                                            note=f"ESCALATE {name}: {verdict.reason}"))
                    if checkpointer is not None:
                        checkpointer.save(state)
                    return trace
                if verdict.action == "block":
                    trace.steps.append(Step("guard", verdict.by, args={"tool": name, **args}, ok=False,
                                            note=f"BLOCK {name}: {verdict.reason}"))
                    result = ToolResult(False, error=f"Blocked by policy ({verdict.by}): {verdict.reason}")
                else:
                    result = _execute(executor, world, name, args, f"{task.id}:{tc['id']}", trace)
                    for g in guards:
                        result = g.on_tool_result(ctx, name, args, result)
            _answer(state, tc["id"], result)
            if checkpointer is not None:
                checkpointer.save(state)

        # 2. Stop conditions.
        if state.step >= max_steps:
            return _finish(state, trace, None, "max_steps")
        if budget_usd is not None and trace.cost_usd >= budget_usd:
            return _finish(state, trace, None, "budget_exceeded")

        # 3. Ask the model what to do next.
        msg, usage = model(state.messages, tool_schemas)
        state.step += 1
        state.messages.append(msg)
        calls = msg.get("tool_calls") or []
        trace.steps.append(Step(
            "llm", getattr(model, "name", "model"),
            output=([{"call": c["function"]["name"], "args": _parse_args(c["function"]["arguments"])}
                     for c in calls] or (msg.get("content") or "")),
            latency_ms=usage.latency_ms, tokens_in=usage.prompt_tokens,
            tokens_out=usage.completion_tokens, cost_usd=usage.cost_usd))
        if checkpointer is not None:
            checkpointer.save(state)
        if not calls:
            text = msg.get("content") or ""
            for g in guards:
                text = g.on_output(ctx, text)
            return _finish(state, trace, text, "done")
    return trace


def resume_agent(trace: Trace, task: Task, world: World, decision: str, *, note: str = "",
                 edited_args: dict | None = None, **kwargs) -> Trace:
    """Continue a run paused for human approval. decision: approve | reject | edit."""
    state: AgentState = trace.state
    p = state.pending
    assert state.status == "awaiting_approval" and p, "this run is not waiting for approval"
    executor = kwargs.get("executor") or default_executor
    if decision == "reject":
        result = ToolResult(False, error=f"A human reviewer rejected this action. {note}".strip())
    else:
        args = edited_args if decision == "edit" else p["args"]
        result = _execute(executor, world, p["name"], args, f"{task.id}:{p['tool_call_id']}", trace)
    trace.steps.append(Step("guard", "human", args=p["args"], ok=decision != "reject",
                            note=f"HUMAN {decision.upper()} {p['name']} {note}".strip()))
    _answer(state, p["tool_call_id"], result)
    state.pending, state.status, trace.status = None, "running", "running"
    return run_agent(task, world, state=state, trace=trace, **kwargs)


def _execute(executor, world, name, args, key, trace) -> ToolResult:
    t0 = world.clock.time()
    w0 = time.perf_counter()
    try:
        result = executor(world, name, args, key)
    except ProcessCrash:
        raise
    except Exception as e:  # an unhandled tool exception reaches the model as an error
        result = ToolResult(False, error=f"{type(e).__name__}: {e}",
                            retryable=getattr(e, "retryable", False))
    sim = world.clock.time() - t0
    trace.steps.append(Step("tool", name, args=args,
                            output=result.data if result.ok else {"error": result.error},
                            ok=result.ok, latency_ms=sim + (time.perf_counter() - w0) * 1000,
                            note=result.trust))
    return result


def _unanswered_calls(state: AgentState):
    if not state.messages or state.messages[-1].get("role") not in ("assistant", "tool"):
        return []
    # find the last assistant message and which of its calls have tool replies
    for i in range(len(state.messages) - 1, -1, -1):
        m = state.messages[i]
        if m.get("role") == "assistant":
            answered = {x.get("tool_call_id") for x in state.messages[i + 1:] if x.get("role") == "tool"}
            return [tc for tc in (m.get("tool_calls") or []) if tc["id"] not in answered]
    return []


def _answer(state: AgentState, call_id: str, result: ToolResult):
    state.messages.append({"role": "tool", "tool_call_id": call_id, "content": result.to_message()})


def _finish(state, trace, text, status):
    state.status, state.final = status, text
    trace.status, trace.final = status, text
    if text is not None:
        trace.steps.append(Step("final", output=text))
    return trace
