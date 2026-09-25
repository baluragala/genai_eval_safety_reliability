"""The OpenAI client, the key lookup and the spend meter.

Every OpenAI call in the lab goes through `chat()`, which means every call is
metered, priced and capped in one place.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

from . import config


class SpendLimitExceeded(RuntimeError):
    pass


@dataclass
class Usage:
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0

    @property
    def cost_usd(self) -> float:
        pin, pout = config.PRICES.get(self.model, config.DEFAULT_PRICE)
        return (self.prompt_tokens * pin + self.completion_tokens * pout) / 1_000_000


@dataclass
class SpendMeter:
    limit_usd: float = config.SPEND_LIMIT_USD
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    spent_usd: float = 0.0
    by_purpose: dict = field(default_factory=dict)

    def check(self):
        if self.spent_usd >= self.limit_usd:
            raise SpendLimitExceeded(
                f"Notebook spend limit ${self.limit_usd:.2f} reached "
                f"(${self.spent_usd:.4f} over {self.calls} calls). "
                "Raise agentlab.llm.METER.limit_usd if you really mean it.")

    def record(self, usage: Usage, purpose: str):
        self.calls += 1
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.spent_usd += usage.cost_usd
        p = self.by_purpose.setdefault(purpose, {"calls": 0, "usd": 0.0})
        p["calls"] += 1
        p["usd"] += usage.cost_usd

    def __repr__(self):
        return (f"SpendMeter(calls={self.calls}, tokens={self.prompt_tokens}+"
                f"{self.completion_tokens}, spent=${self.spent_usd:.4f} "
                f"of ${self.limit_usd:.2f})")


METER = SpendMeter()
_CLIENT = None


def _find_key() -> str | None:
    """Colab Secrets, then the environment. Never printed, never stored in a cell."""
    try:
        from google.colab import userdata  # type: ignore
        key = userdata.get("OPENAI_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("OPENAI_API_KEY")


def set_client(client):
    """Install a client explicitly (the test suite uses this to install a fake)."""
    global _CLIENT
    _CLIENT = client


def get_client():
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    key = _find_key()
    if not key:
        try:
            import getpass
            key = getpass.getpass("OPENAI_API_KEY (input hidden): ").strip()
        except Exception:
            key = None
    if not key:
        raise RuntimeError(
            "This lab needs an OpenAI API key.\n"
            "  Colab: 🔑 panel → Add new secret → name OPENAI_API_KEY → enable notebook access.\n"
            "  Local: export OPENAI_API_KEY=... before starting Jupyter.")
    from openai import OpenAI
    _CLIENT = OpenAI(api_key=key, max_retries=2, timeout=60)
    return _CLIENT


def check_connection(model: str = config.MODEL) -> str:
    """One tiny call to make sure the key and the model name both work."""
    msg, usage = chat([{"role": "user", "content": "Reply with the single word: ready"}],
                      model=model, max_tokens=5, purpose="setup")
    return f"{model} says {msg['content']!r}  ({usage.prompt_tokens}+{usage.completion_tokens} tokens)"


def _message_to_dict(m) -> dict:
    out = {"role": "assistant", "content": m.content}
    if getattr(m, "tool_calls", None):
        out["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in m.tool_calls
        ]
    return out


def chat(messages, *, tools=None, model: str = config.MODEL, temperature=config.TEMPERATURE,
         seed=config.SEED, response_format=None, max_tokens=None, purpose="agent"):
    """One chat-completions call. Returns (assistant_message_dict, Usage)."""
    METER.check()
    kwargs = dict(model=model, messages=messages, temperature=temperature)
    if seed is not None:
        kwargs["seed"] = seed
    if tools:
        kwargs["tools"] = tools
        kwargs["parallel_tool_calls"] = False
    if response_format:
        kwargs["response_format"] = response_format
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    t0 = time.perf_counter()
    resp = get_client().chat.completions.create(**kwargs)
    latency = (time.perf_counter() - t0) * 1000
    u = getattr(resp, "usage", None)
    usage = Usage(model=model,
                  prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
                  completion_tokens=getattr(u, "completion_tokens", 0) or 0,
                  latency_ms=latency)
    METER.record(usage, purpose)
    return _message_to_dict(resp.choices[0].message), usage


def chat_json(system: str, user: str, *, model: str = config.JUDGE_MODEL, purpose="judge",
              temperature=config.TEMPERATURE) -> tuple[dict, Usage]:
    """A chat call that must return a JSON object."""
    msg, usage = chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                      model=model, temperature=temperature,
                      response_format={"type": "json_object"}, purpose=purpose)
    try:
        return json.loads(msg["content"] or "{}"), usage
    except json.JSONDecodeError:
        return {"_unparseable": msg["content"]}, usage
