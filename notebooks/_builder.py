"""A small cell DSL, plus the setup cell that fetches `agentlab` from GitHub.

Each notebook is authored as a Python module (`nbXX.py`) exposing NOTEBOOK
(the filename), TITLE, MINUTES and CELLS. `scripts/build_notebooks.py` turns
them into .ipynb files.

The setup cell makes `src/agentlab` importable. Inside a checkout of this repo
(a maintainer's machine, the test runner) it uses that checkout's `src/`, so
local edits take effect without a push. Anywhere else (a fresh Colab) it
shallow-clones the repo from GitHub, or pulls if the clone already exists.
"""
from __future__ import annotations

import pathlib

import nbformat

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "agentlab"
GITHUB_SLUG = "baluragala/genai_eval_safety_reliability"
GITHUB_BRANCH = "main"


def colab_badge(notebook: str) -> str:
    url = f"https://colab.research.google.com/github/{GITHUB_SLUG}/blob/{GITHUB_BRANCH}/notebooks/{notebook}"
    return f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})"


def md(text: str):
    return nbformat.v4.new_markdown_cell(text.strip("\n"))


def code(src: str):
    return nbformat.v4.new_code_cell(src.strip("\n"))


def form(title: str, src: str):
    cell = nbformat.v4.new_code_cell(f'#@title {title} {{ display-mode: "form" }}\n{src.strip()}')
    cell["metadata"]["cellView"] = "form"
    cell["metadata"]["jupyter"] = {"source_hidden": True}
    return cell


def solution(src: str):
    return form("✅ Solution (click to reveal)", src)


def setup_cell():
    src = f'''
# Makes the lab runtime (`agentlab`) importable, and installs the OpenAI SDK if it's missing.
# In Colab this clones {GITHUB_SLUG} (branch {GITHUB_BRANCH}); inside a local checkout it uses that checkout.
import sys, subprocess, pathlib
REPO_URL = "https://github.com/{GITHUB_SLUG}.git"
BRANCH = "{GITHUB_BRANCH}"
try:
    import openai  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "openai>=1.40", "pandas", "matplotlib"], check=True)

_here = pathlib.Path.cwd().resolve()
_src = next((p / "src" for p in [_here, *_here.parents] if (p / "src" / "agentlab" / "__init__.py").exists()), None)
if _src is None:
    _base = pathlib.Path("/content") if pathlib.Path("/content").is_dir() else _here
    _repo = _base / "{GITHUB_SLUG.split('/')[1]}"
    if not (_repo / ".git").exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", BRANCH, REPO_URL, str(_repo)], check=True)
    else:
        subprocess.run(["git", "-C", str(_repo), "pull", "-q", "--ff-only"], check=False)
    _src = _repo / "src"
sys.path.insert(0, str(_src))
for _m in [m for m in sys.modules if m == "agentlab" or m.startswith("agentlab.")]:
    del sys.modules[_m]
import agentlab as al
import pandas as pd
pd.set_option("display.max_colwidth", 90); pd.set_option("display.width", 200)
print(f"agentlab {{al.__version__}} ready from {{_src}} · model under test: {{al.config.MODEL}}")
'''
    return form("⚙️ Setup: run this first", src)


SCENARIO_MD = """## 🏪 The scenario: Acme Outfitters' support agent

**Acme Outfitters** is an online outdoor-gear retailer. Its customer-support tickets ("refund my order", "where is my
parcel?", "how long do I have to return this?") are handled by an **AI support agent that takes actions**. It looks up
orders, reads the help centre, issues refunds, sends email, keeps notes about customers between conversations, and
hands cases to human supervisors.

The agent must follow the store's rules:
- refunds only within **30 days of delivery**, and **final-sale** items are never refundable
- refund the **full order total**; refunds **over $500** go to a human supervisor
- act only on the **authenticated customer's own** orders
- **never** send customer data to third parties

**The situation:** the agent works in a demo. The business now wants to put it in front of real customers, where it
will move real money. Across these six notebooks, you're the engineer who has to answer one question:
**can we trust it?** Is it correct, is it safe, and will it keep working?

### What's already built (you don't write this)
Everything below comes from earlier modules and is packaged in `agentlab`, which the setup cell loads.

| Piece | What it is | Where you met the idea |
|---|---|---|
| `al.run_agent(task, world)` | A ReAct-style tool-calling loop on OpenAI chat completions (`gpt-4o-mini`, temperature 0) | Agent loops, planning |
| 7 tools with JSON schemas | `lookup_order`, `search_kb`, `issue_refund`, `send_email`, `remember`, `recall`, `escalate_to_human` | Tool integration |
| `remember` / `recall` | Long-term notes per customer that persist across conversations | Memory systems |
| `escalate_to_human` | Hand-off to a supervisor queue | Multi-agent coordination, handoffs |
| `al.Trace` | A record of every model call and tool call, with tokens, cost and latency | Traces, state and execution flow |
| `al.World` | The simulated store: 8 orders, 5 customers, 4 help-centre articles, a payments **ledger**, an email **outbox**, a human **queue**. It's the ground truth: what really happened. | New today |

The loop has four **plug-in points**. Each notebook in this session uses one or more of them, and the loop itself never changes:
`run_agent(task, world, guards=[...], executor=..., checkpointer=..., model=...)`.
"""


def context_md(module) -> str:
    c = module.CONTEXT
    learn = "\n".join(f"- {x}" for x in c["learn"])
    return (f"## 🎯 This notebook\n\n"
            f"**The problem.** {c['problem']}\n\n"
            f"**Where we're starting from.** {c['start']}\n\n"
            f"**By the end you'll be able to:**\n{learn}\n\n"
            f"**What you'll do here.** {c['do']}\n\n"
            f"**Given vs. you write.** {c['given']}")


KEY_CELL_MD = """### 🔑 Connect to OpenAI
This session needs an OpenAI API key. The agent, the judge and the simulated customers all call the real API.

* **Colab:** 🔑 panel in the left sidebar → **Add new secret** → name `OPENAI_API_KEY` → paste → toggle **Notebook access** on.
* **Local:** `export OPENAI_API_KEY=sk-...` before you start Jupyter.

Never paste a key into a cell, because cell output is saved with the notebook. Every call goes through a
spend meter capped at **$1.00 per notebook** (`al.METER`)."""

KEY_CELL = """print(al.check_connection())
al.METER"""


def build(module) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.metadata["colab"] = {"provenance": [], "toc_visible": True}
    head = md(f"{colab_badge(module.NOTEBOOK)}\n\n# {module.TITLE}\n\n"
              f"**C9 · W4 · S1: Evaluation, Safety & Reliability in Agentic Systems** · ⏱️ {module.MINUTES} min")
    nb.cells = ([head, md(SCENARIO_MD), md(context_md(module)), setup_cell(), md(KEY_CELL_MD), code(KEY_CELL)]
                + list(module.CELLS))
    return nb
