"""agentlab: the runtime behind the Evaluation, Safety & Reliability session (C9-W4-S1).

One fictional retailer (Acme Outfitters), one tool-using support agent
powered by OpenAI, and the instruments to evaluate it, attack it, guard it
and keep it running.
"""
from . import config, explain
from .agent import (AgentState, Guard, GuardContext, OpenAIAgentModel, ProcessCrash, Step, Task,
                    Trace, Verdict, default_executor, resume_agent, run_agent)
from .evals import (attack_breached, check_task, evaluate, pass_at_k, pass_hat_k, red_team,
                    run_attack, run_task, summarize)
from .llm import METER, chat, chat_json, check_connection
from .tasks import ATTACKS, GOLDEN
from .tools import DEFAULT_TOOLSET, TOOLS, ToolResult
from .world import SimClock, World

__all__ = [n for n in dir() if not n.startswith("_")]
__version__ = "1.0.0"
