"""A small cell DSL, plus the setup cell that ships `agentlab` inside every notebook.

Each notebook is authored as a Python module (`nbXX.py`) exposing NOTEBOOK
(the filename), TITLE, MINUTES and CELLS. `scripts/build_notebooks.py` turns
them into .ipynb files.

The setup cell carries the whole `src/agentlab` package as a base64 zip and
unpacks it next to the notebook. That way a fresh Colab runtime needs no
clone and no pip install of this repo, only an OpenAI key.
"""
from __future__ import annotations

import base64
import io
import pathlib
import zipfile

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


def package_blob() -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(SRC.glob("*.py")):
            info = zipfile.ZipInfo(f"agentlab/{f.name}", date_time=(2026, 1, 1, 0, 0, 0))
            z.writestr(info, f.read_text())
    return base64.b64encode(buf.getvalue()).decode()


def setup_cell():
    blob = package_blob()
    src = f'''
# Installs the OpenAI SDK and unpacks the lab runtime (`agentlab`) next to this notebook.
# Nothing here needs editing. The runtime's source is readable in ./agentlab/ afterwards.
import sys, subprocess, importlib, base64, io, zipfile, pathlib
try:
    import openai  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "openai>=1.40", "pandas", "matplotlib"], check=True)
_BLOB = "{blob}"
with zipfile.ZipFile(io.BytesIO(base64.b64decode(_BLOB))) as _z:
    _z.extractall(pathlib.Path.cwd())
sys.path.insert(0, str(pathlib.Path.cwd()))
for _m in [m for m in sys.modules if m == "agentlab" or m.startswith("agentlab.")]:
    del sys.modules[_m]
import agentlab as al
import pandas as pd
pd.set_option("display.max_colwidth", 90); pd.set_option("display.width", 200)
print(f"agentlab {{al.__version__}} ready · model under test: {{al.config.MODEL}}")
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
