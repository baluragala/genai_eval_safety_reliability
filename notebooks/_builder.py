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
    nb.cells = [head, setup_cell(), md(KEY_CELL_MD), code(KEY_CELL)] + list(module.CELLS)
    return nb
