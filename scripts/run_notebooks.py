"""Execute built notebooks headlessly.

  python scripts/run_notebooks.py            # FAKE OpenAI client (no key, no cost): plumbing check
  python scripts/run_notebooks.py --live     # real OpenAI; needs OPENAI_API_KEY; writes executed copies
                                             # to reference_runs/ so you can see real outputs before class
"""
import pathlib
import sys

import nbformat
from nbclient import NotebookClient

ROOT = pathlib.Path(__file__).resolve().parent.parent
FAKE = f"""import sys; sys.path.insert(0, {str(ROOT / 'tests')!r})
from fake_openai import FakeOpenAI
al.llm.set_client(FakeOpenAI())"""


def run(path: pathlib.Path, live: bool, timeout=900):
    nb = nbformat.read(path, as_version=4)
    if not live:
        nb.cells.insert(4, nbformat.v4.new_code_cell(FAKE))
    workdir = ROOT / ("reference_runs" if live else ".nbrun")
    workdir.mkdir(exist_ok=True)
    NotebookClient(nb, timeout=timeout, kernel_name="python3",
                   resources={"metadata": {"path": str(workdir)}}).execute()
    if live:
        nbformat.write(nb, workdir / path.name)
    return nb


if __name__ == "__main__":
    live = "--live" in sys.argv
    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    for p in sorted((ROOT / "notebooks").glob("*.ipynb")):
        if names and not any(n in p.name for n in names):
            continue
        run(p, live)
        print("✓", p.name)
