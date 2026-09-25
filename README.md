# Evaluation, Safety & Reliability in Agentic Systems

**C9 · W4 · S1 · 120-minute live session.** Six Colab notebooks and one agent. Over the session, learners
**evaluate** it, **diagnose** its failures, **attack** it, **guard** it, make it **survive a flaky world**, and
decide where **humans** have to stay in the loop.

> ## The one takeaway
> **Score the world the agent leaves behind, not the story it tells. Then put the controls where the model
> can't talk its way past them.**

## The running case

**Acme Outfitters' support agent** is an OpenAI tool-calling agent (`gpt-4o-mini`, temperature 0, seed 7) with
seven tools: `lookup_order`, `search_kb`, `issue_refund`, `send_email`, `remember`, `recall` and
`escalate_to_human`. It runs against a small simulated retailer (8 orders, 5 customers, 4 help-centre
articles, a payments ledger, an outbox and a human queue). Each notebook plugs something new into the same
agent loop:

```
run_agent(task, world,
          guards=[...],        # NB04: input · context · action · output · human approval
          executor=...,        # NB05: retries · breakers · fallbacks · idempotency · fault injection
          checkpointer=...,    # NB05: crash → resume
          model=...)           # NB05: primary → fallback model
```

## The notebooks

| # | Notebook | Agenda block | Min | Mode |
|---|---|---|---:|---|
| 01 | [`01_evaluating_agents.ipynb`](notebooks/01_evaluating_agents.ipynb) | Introduce evaluation | 20 | Conceptual + demo |
| 02 | [`02_traces_and_failures.ipynb`](notebooks/02_traces_and_failures.ipynb) | Analyse traces & failures | 20 | Guided analysis |
| 03 | [`03_safety_risks.ipynb`](notebooks/03_safety_risks.ipynb) | Understand safety risks | 20 | Demonstration |
| 04 | [`04_guardrails.ipynb`](notebooks/04_guardrails.ipynb) | Implement guardrails | 20 | Guided coding |
| 05 | [`05_reliability.ipynb`](notebooks/05_reliability.ipynb) | Reliability engineering | 25 | Demo + guided practice |
| 06 | [`06_oversight_and_wrapup.ipynb`](notebooks/06_oversight_and_wrapup.ipynb) | Operational trust & oversight + wrap-up | 15 | Conceptual |

Each notebook opens in Colab from its badge. The badges point at
`github.com/baluragala/genai_eval_safety_reliability` on `main`, so they only work once this repo has been
pushed there. Until then, upload the `.ipynb` to Colab (File → Upload notebook).

### What each one covers

* **01 · Evaluation.** Evaluation dimensions (success, tool correctness, trajectory, cost, latency, robustness,
  safety); a 10-task golden set scored by a world-state oracle; trace assertions; **LLM-as-judge** with a rubric,
  run reply-only and reply+trace and **calibrated** against the oracle (agreement, Cohen's κ, false passes);
  **simulation** (perturbed tickets and LLM-played multi-turn customers); pass@k versus pass^k.
* **02 · Traces & failures.** The failure taxonomy (reasoning failure, planning breakdown, infinite loop, tool
  misuse, silent failure); a gallery of traces to classify; detectors that classify automatically; then
  misconfigured live agents, to see which failures the real model actually produces.
* **03 · Safety risks.** Six attacks against the unguarded agent: direct injection, indirect injection through
  a poisoned KB article (exfiltration), injection through a customer-controlled field, memory poisoning,
  code execution through an `eval` calculator, and social-engineered PII exfiltration. Measured as attack success rate.
* **04 · Guardrails.** Built layer by layer and re-attacked after each one: input screening, spotlighting,
  least privilege, sandboxed tools, policy grounded in the system of record, egress control, memory-write
  validation and output redaction. It ends with the safety-versus-utility trade-off.
* **05 · Reliability.** Fault injection, retries with backoff and jitter, idempotency keys, checkpoint and resume
  after a crash (a double refund without the key, a single refund with it), circuit breakers, fallbacks, graceful
  degradation, loop and budget limits, and a chaos experiment.
* **06 · Oversight.** Human approval (approve / reject / edit), an escalation matrix, a tamper-evident audit log,
  a kill switch, and a release gate that scores baseline against hardened. Then the wrap-up.

## Slides

26-slide deck with speaker notes, in the same order as the notebooks: https://claude.ai/artifact/3wfgdBNhmA1zRVAjp5Tn9A (private until shared from its Share menu). Source: `deck/project/`.

## Running it

**An OpenAI API key is required.** The agent, the judge, the simulated customers and the injection classifier
all call the real API.

* **Colab:** 🔑 panel → *Add new secret* → `OPENAI_API_KEY` → turn on **Notebook access**.
* **Local:** `export OPENAI_API_KEY=...`, then `pip install -r requirements.txt` and `jupyter lab`.

You don't need to clone or `pip install` anything else. Each notebook's first cell unpacks the lab runtime
(`agentlab`, from `src/agentlab/`). Every call goes through a **spend meter capped at $1.00 per notebook**
(`al.METER`). A full run of all six costs a few cents with `gpt-4o-mini`; the instructor guide has the estimate.

Model names and prices live in **one file**, `src/agentlab/config.py`. Check them against the current OpenAI
docs before the session, then rebuild the notebooks (below).

## Repository layout

```
src/agentlab/        the runtime (inlined into every notebook's setup cell)
  agent.py           agent loop, Trace, guard hooks, checkpoint/resume, human approval
  world.py tools.py  Acme Outfitters, tools, provenance (trusted / untrusted)
  tasks.py evals.py  golden set, attack set, oracle, metrics, pass@k / pass^k, red team
  judge.py           LLM-as-judge + calibration
  simulation.py      perturbations, simulated customers
  guards.py          reference guardrails (NB04)
  reliability.py     faults, retry, breaker, fallback, idempotency, checkpointer (NB05)
  oversight.py       escalation matrix, audit log, kill switch, release gate (NB06)
  fixtures.py        hand-built failure gallery (NB02)
notebooks/nbXX.py    notebook sources  →  notebooks/*.ipynb (built)
teaching/            instructor guide, learner handout, exercises, solutions
tests/               runtime tests + notebook execution tests (fake OpenAI client)
scripts/             build_notebooks.py, run_notebooks.py
```

## For maintainers

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/build_notebooks.py          # after editing src/agentlab or notebooks/nbXX.py
.venv/bin/python -m pytest                           # no key, no network: uses tests/fake_openai.py
OPENAI_API_KEY=... .venv/bin/python scripts/run_notebooks.py --live   # real run → reference_runs/
```

The test suite runs every notebook top to bottom against **`tests/fake_openai.py`**, a scripted stand-in for the
API that follows the store policy but obeys injected instructions. The tests prove the plumbing works. They
say nothing about what `gpt-4o-mini` will actually do. **Run `--live` once before teaching.** The
executed copies land in `reference_runs/`, so you'll know the real numbers before the room does.

## A note on honesty in the material

* The Notebook 02 failure gallery is **hand-constructed**, one trace per failure class, and it says so. The live
  section of that notebook then shows what the real model does.
* Nothing in the markdown asserts a live number. Every success rate, attack success rate and cost figure is computed in the cell that
  shows it, because a real model may resist an attack the lesson expects to succeed, or fall for one it expects to fail.
  The instructor guide covers how to teach either outcome.
* Prices in `config.py` are illustrative.
