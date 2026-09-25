"""Simulation-based testing: LLM-played customers and ticket perturbations.

A golden set has ten hand-written tickets. Real customers write ten thousand
different ones. Simulation produces some of the variety you'd otherwise only
meet in production.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from . import config
from .agent import Task, Trace, run_agent
from .llm import chat, chat_json
from .world import World

# ---------------------------------------------------------------- perturbations

def _typos(text: str, rng: random.Random, rate=0.08) -> str:
    out = []
    for w in text.split(" "):
        if len(w) > 3 and not re.search(r"\d", w) and rng.random() < rate * 3:
            i = rng.randrange(1, len(w) - 1)
            w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
        out.append(w)
    return " ".join(out)


def perturb(ticket: str, style: str, seed: int = config.SEED) -> str:
    """Deterministic, model-free rewrites. Styles: typos, shouting, terse, padded, hinglish."""
    rng = random.Random(f"{seed}:{style}:{ticket}")
    oid = (re.findall(r"A-\d{4}", ticket) or [""])[0]
    if style == "typos":
        return _typos(ticket, rng)
    if style == "shouting":
        return ticket.upper() + " THIS IS THE THIRD TIME I'M ASKING!!!"
    if style == "terse":
        return f"{oid} refund" if "refund" in ticket.lower() else (oid + " ?" if oid else ticket)
    if style == "padded":
        return ("Hello there, hope you are having a lovely day. I've been a customer for years "
                "and I really love your products, especially the hiking range. Anyway, " + ticket +
                " Also, unrelated, do you sell gift cards? Thanks so much!")
    if style == "hinglish":
        return (f"Hi, mera order {oid} ka " + ("refund chahiye please, " if "refund" in ticket.lower()
                                             else "status kya hai? ") + ticket)
    raise ValueError(style)


STYLES = ["typos", "shouting", "terse", "padded", "hinglish"]


def rewrite_as_persona(ticket: str, persona: str, *, seed: int = config.SEED) -> str:
    """Ask the simulator model to rewrite a ticket in a persona's voice, keeping the facts."""
    sys = ("Rewrite the customer message in the voice of the given persona. Keep every fact "
           "(order ids, what they want) unchanged. Return JSON {\"message\": \"...\"}.")
    out, _ = chat_json(sys, f"Persona: {persona}\nMessage: {ticket}",
                       model=config.SIMULATOR_MODEL, purpose="simulator", temperature=0.9)
    return out.get("message", ticket)


# ---------------------------------------------------------------- multi-turn simulated customer

@dataclass
class Scenario:
    id: str
    customer_id: str
    persona: str
    goal: str                      # what the customer wants, in plain words
    hidden_facts: dict             # things they know but only say when asked
    expect: dict                   # the same oracle expectations as the golden set
    opening: str | None = None     # if None, the simulator writes the first message


SCENARIOS = [
    Scenario("S01", "C-103", "busy parent, types on a phone, short messages",
             "get a refund for the climbing harness that is the wrong size",
             {"order_id": "A-1008"},
             {"outcome": "refund", "order_id": "A-1008", "amount": 210.00,
              "must_call": ["lookup_order", "issue_refund"]},
             opening="hey the harness i bought is the wrong size, can i get my money back"),
    Scenario("S02", "C-101", "polite but persistent, pushes back once if refused",
             "get a refund for the down jacket (delivered 45 days ago)",
             {"order_id": "A-1002"},
             {"outcome": "deny", "order_id": "A-1002", "must_call": ["lookup_order"],
              "must_not_call": ["issue_refund"]},
             opening="Hello, I'd like to return the down jacket I bought, please."),
    Scenario("S03", "C-102", "anxious first-time customer who over-explains",
             "find out whether the tent can be refunded because it leaks",
             {"order_id": "A-1003"},
             {"outcome": "escalate", "order_id": "A-1003", "must_call": ["escalate_to_human"],
              "must_not_call": ["issue_refund"]}),
    Scenario("S04", "C-103", "non-native English speaker, mixes Hindi and English",
             "know when the headlamp will arrive",
             {"order_id": "A-1004"},
             {"outcome": "info", "order_id": "A-1004",
              "answer_any": ["shipped", "on its way", "in transit", "2 business days"]}),
]

SIM_SYSTEM = """You are role-playing a customer of Acme Outfitters chatting with a support agent.
Persona: {persona}
Your goal: {goal}
Facts you know (only share them when asked): {facts}
Rules: write ONE short customer message per turn, in character. Never invent order ids other than
yours. If your goal is achieved, or the agent clearly says it can't be done and you have pushed
back once, reply with exactly [DONE]."""


def _customer_turn(sc: Scenario, transcript: list[dict]) -> str:
    msgs = [{"role": "system", "content": SIM_SYSTEM.format(persona=sc.persona, goal=sc.goal,
                                                            facts=sc.hidden_facts)}]
    # From the simulator's point of view the agent is the "user" and it is the "assistant".
    for t in transcript:
        msgs.append({"role": "assistant" if t["who"] == "customer" else "user", "content": t["text"]})
    if not transcript:
        msgs.append({"role": "user", "content": "(The chat opens. Write your first message.)"})
    msg, _ = chat(msgs, model=config.SIMULATOR_MODEL, temperature=0.7, purpose="simulator")
    return (msg.get("content") or "").strip()


def simulate(sc: Scenario, *, max_turns: int = 4, verbose=True, **agent_kwargs):
    """Run a multi-turn conversation between a simulated customer and the agent."""
    agent_kwargs.setdefault("max_steps", 16)
    world = World.fresh()
    transcript: list[dict] = []
    trace: Trace | None = None
    task = Task(sc.id, "", sc.customer_id, sc.expect)
    for turn in range(max_turns):
        said = sc.opening if (turn == 0 and sc.opening) else _customer_turn(sc, transcript)
        if "[DONE]" in said:
            break
        transcript.append({"who": "customer", "text": said})
        if verbose:
            print(f"  👤 {said}")
        if trace is None:
            task.ticket = said
            trace = run_agent(task, world, **agent_kwargs)
        else:
            st = trace.state
            st.messages.append({"role": "user", "content": said})
            st.status = "running"
            trace.status = "running"
            trace = run_agent(task, world, state=st, trace=trace, **agent_kwargs)
        reply = trace.final or f"(no reply: {trace.status})"
        transcript.append({"who": "agent", "text": reply})
        if verbose:
            print(f"  🤖 {reply}")
        if trace.status != "done":
            break
    return trace, world, transcript
