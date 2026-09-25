"""Every notebook builds, executes top-to-bottom (fake OpenAI client), and ships no secrets."""
import pathlib
import re

import nbformat
import pytest

from scripts.build_notebooks import MODULES
from scripts.run_notebooks import run

ROOT = pathlib.Path(__file__).resolve().parent.parent
NBS = sorted((ROOT / "notebooks").glob("*.ipynb"))


def test_all_notebooks_built():
    assert len(NBS) == len(MODULES)


@pytest.mark.parametrize("path", NBS, ids=lambda p: p.name)
def test_notebook_executes(path):
    run(path, live=False)


@pytest.mark.parametrize("path", NBS, ids=lambda p: p.name)
def test_notebook_has_setup_key_cell_and_no_key(path):
    nb = nbformat.read(path, as_version=4)
    assert "agentlab" in nb.cells[1].source and "check_connection" in nb.cells[3].source
    text = path.read_text()
    assert not re.search(r"sk-[A-Za-z0-9_-]{20,}", text)
    assert all(not c.get("outputs") for c in nb.cells if c.cell_type == "code"), "commit notebooks without outputs"


def test_no_key_anywhere_in_repo():
    for p in ROOT.rglob("*"):
        if p.is_file() and ".venv" not in p.parts and p.suffix in {".py", ".md", ".ipynb", ".txt", ".example"}:
            assert not re.search(r"sk-(proj-)?[A-Za-z0-9_-]{20,}", p.read_text(errors="ignore")), p
