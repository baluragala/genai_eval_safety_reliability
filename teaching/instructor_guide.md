# C9-W4-S1: Instructor guide

**Deck:** https://claude.ai/artifact/3wfgdBNhmA1zRVAjp5Tn9A (26 slides, speaker notes on every slide)

## Evaluation, Safety & Reliability in Agentic Systems · 120 minutes

One running case, six notebooks. The **Acme Outfitters support agent** is an OpenAI tool-calling agent
that looks up orders, reads the help centre, issues refunds, sends email, keeps long-term notes about
customers and escalates to humans. Learners measure it (01), debug it (02), attack it (03), guard it (04),
break its dependencies (05) and put humans around it (06).

Each notebook's setup cell clones this repo from GitHub and imports the runtime (`agentlab`), so learners need only an OpenAI key and network access to GitHub.

---

## Using the notebook format live

- **✋ Predict first:** pause and take a show of hands (or a chat poll) before running the next cell. Twenty seconds is enough; the point is commitment, not accuracy.
- **🔍 Reading the output:** read the bullets aloud only the first time a table shape appears. Leave the ▸ dropdowns for self-study.
- **💡 explanation cells:** these are the safest thing to narrate from, because they describe the room's actual run. The markdown never states live numbers.

## Before the session

**Checklist (do this the day before):**

1. **Key:** put `OPENAI_API_KEY` in Colab Secrets (🔑 panel → Add new secret → toggle *Notebook access*).
   Tell learners to do the same *before* class. The toggle is the step people miss.
2. **Model name:** check that `al.config.MODEL` (`gpt-4o-mini`) and `FALLBACK_MODEL` (`gpt-4.1-nano`) are
   still offered. The second cell of every notebook (`al.check_connection()`) fails loudly if not.
   Edit `src/agentlab/config.py` and rebuild (`python scripts/build_notebooks.py`) if needed.
3. **Run every notebook once, top to bottom, with a real key.** `python scripts/run_notebooks.py --live`
   does this headlessly and saves executed copies in `reference_runs/`. Read the real outputs: which
   attacks breached, which failures your model reproduced in NB02, which chaos cells degraded. Model
   behaviour drifts between releases, so the story you tell has to match what the model does *this week*.
4. **Have the fallback ready:** keep the executed `reference_runs/*.ipynb` open in a tab in case the
   live API is slow or rate-limited during class.

**Expected cost (estimate, not a measurement).** A typical agent step sends about 1,500 input tokens
(system prompt + tool schemas + history) and gets back about 50. At `gpt-4o-mini` list prices
($0.15 / $0.60 per 1M) that's about **$0.00026 per step**, and roughly **$0.0013 per task** at ~5 steps.

| Notebook | Approx. model calls | Approx. cost |
|---|---:|---:|
| 01 evaluation (golden + judge + perturbations + simulation + pass^k) | ~230 | ~$0.06 |
| 02 traces & failures | ~60 | ~$0.02 |
| 03 safety risks (red team) | ~50 | ~$0.01 |
| 04 guardrails (red team + golden × configs) | ~150 | ~$0.04 |
| 05 reliability (chaos sweep) | ~400 | ~$0.10 |
| 06 oversight (HITL + scorecard) | ~250 | ~$0.07 |
| **Total per learner** | **~1,100** | **≈ $0.30–0.60** |

Every notebook has a hard cap of **$1.00** (`al.METER.limit_usd`). The meter refuses further calls
once a notebook reaches the cap, so a runaway loop can't drain anyone's account.

---

## Run sheet

| Time | Block (agenda) | Notebook | Mode |
|---|---|---|---|
| 0:00–0:20 | Introduce evaluation | 01 | Conceptual + demonstration |
| 0:20–0:40 | Analyse traces and failures | 02 | Guided analysis |
| 0:40–1:00 | Understand safety risks | 03 | Demonstration |
| 1:00–1:20 | Implement guardrails | 04 | Guided coding |
| 1:20–1:45 | Reliability engineering | 05 | Demonstration + guided practice |
| 1:45–1:55 | Operational trust & oversight | 06 §1–5 | Conceptual |
| 1:55–2:00 | Wrap-up & Q&A | 06 §6–wrap-up | Q&A |

### 0:00–0:20 · Notebook 01: Evaluating agents

| Min | What happens |
|---|---|
| 0–2 | Setup + key cells. While they run: "You can build agents. Today: can you *trust* one?" |
| 2–5 | ▶ **Live moment 1:** run T01 and `trace.show()`. Put *Agent said* next to *Ledger says*. The whole session builds on this gap. |
| 5–8 | Dimensions table. Run the golden set, then `summarize()`. |
| 8–9 | 🗣️ "100% success: ship it?" Expected answers: we haven't measured safety, robustness, cost under load, or behaviour when tools fail. Ten tasks is not a distribution. |
| 9–11 | Trace assertions. Learners do the `recall_first` exercise (warning, not a gate). |
| 11–16 | ▶ **Live moment 2:** LLM-as-judge, reply-only vs reply+trace. Point at the silent-failure case: the reply-only judge passes "Your refund has been processed!". Explain κ. |
| 16–19 | Perturbations pivot + one simulated conversation. Whatever fails, read its trace. That's the bridge to NB02. |
| 19–20 | pass@k vs pass^k: customers need pass^k. |

### 0:20–0:40 · Notebook 02: Traces & failure taxonomy

| Min | What happens |
|---|---|
| 0–3 | The five classes: reasoning, planning, loop, tool misuse, silent failure. |
| 3–10 | **Gallery exercise:** learners label traces A–E *before* you reveal anything (answer key below). |
| 10–16 | Build the detectors. Each class has a mechanical signature in the trace. |
| 16–20 | Run the misconfigured live agents. Ask: *which of these failures does our model actually produce?* Both outcomes teach: if the model is robust, the detector stays quiet on real traffic; if not, it catches the failure. |

**Answer key: failure gallery**

| Trace | Class | The signature |
|---|---|---|
| healthy | — | recall → lookup → policy → act → truthful report |
| **A** | **Reasoning failure** | Right data in context (`final_sale: true`, policy retrieved), wrong conclusion: refunded clearance socks |
| **B** | **Planning breakdown** | Acted before verifying: `issue_refund` *first*, amount taken from the customer's claim ($150 ≠ $129), lookup afterwards |
| **C** | **Infinite loop** | Same call with identical args 8×, no reply, `status=max_steps` |
| **D** | **Tool misuse** | Unit confusion: `amount=12900` (cents) passed to a dollars API. Same shape as the healthy trace, so you have to read the args |
| **E** | **Silent failure** | Every step green, but the refund result says `GW_DECLINED`; the agent tells the customer it went through |

Discussion prompt: *which of these would an outcome-only eval catch? Which would a reply-only judge catch?*
Expected answer: the oracle catches A, B, C, D and E (the ledger is wrong in every one). A reply-only judge
misses E and D completely, and probably A. That's the argument for trace-aware evaluation.

### 0:40–1:00 · Notebook 03: Safety risks

| Min | What happens |
|---|---|
| 0–4 | The attack surface: anything the model *reads* can become an instruction; anything it can *call* is what the attacker gets. |
| 4–14 | ▶ **Live moment 3:** the red team, six attacks (direct injection, poisoned KB → exfiltration, poisoned order note → refund, memory poisoning across sessions, calculator code execution, social-engineering exfiltration). Show the trace of the most striking breach. |
| 14–20 | Debrief: what each attack exploited, and why "a better system prompt" isn't the fix. |

**If the model resists some or all attacks** (likely with newer models, and it varies by week):
don't treat it as a failed demo. Teach it this way: *"The model held today. Would you bet the refund
ledger on it holding after the next model update? Or on a paraphrase?"* Then run the attack's paraphrase or
let learners write one. The guardrails in NB04 don't depend on the model resisting, and that's the lesson.
**If every attack breaches**, teach it that way: this is a typical agent with a reasonable prompt.

### 1:00–1:20 · Notebook 04: Guardrails (guided coding)

| Min | What happens |
|---|---|
| 0–3 | The layers: input → context → action → oversight → output. |
| 3–8 | Input heuristic, then show its bypass. Screening input is whack-a-mole. |
| 8–15 | **The action layer** (`RefundPolicy`, `EgressPolicy`, `MemoryWriteGuard`, allowlist/sandbox). These check the *system of record*, not the model's claims. |
| 15–20 | Re-run red team **and** golden set. Breaches fall, and success must not fall (over-blocking is a failure too). |

Discussion prompt: *"Spotlighting reduced breaches but didn't eliminate them. Why keep it?"*
Expected answer: defence in depth. It lowers how often the action layer has to fire, and it's cheap.
Only deterministic policy on actions gives a guarantee.

### 1:20–1:45 · Notebook 05: Reliability

| Min | What happens |
|---|---|
| 0–4 | Fault injection: timeouts, 429s, 503s, crashes. Baseline under 20–30% fault rate. |
| 4–9 | Retry with exponential backoff + full jitter, only for retryable errors, within a deadline. **Writes aren't retried** until they're idempotent. |
| 9–13 | Circuit breaker: closed → open → half-open. Fail fast rather than pile on. |
| 13–17 | Fallback (read replica for lookups) and graceful degradation (queue refunds for humans rather than fail or lie). |
| 17–21 | ▶ **Live moment 4:** crash right after `issue_refund` takes effect → resume from checkpoint → **double refund**. Add idempotency keys → exactly one. |
| 21–25 | Chaos sweep plot: success vs fault rate, naive vs hardened. |

### 1:45–2:00 · Notebook 06: Oversight + wrap-up

| Min | What happens |
|---|---|
| 0–3 | HITL: pause → approve / reject / edit → resume. Map to LangGraph `interrupt` / `Command(resume=…)`. |
| 3–5 | Escalation matrix: autonomy is tiered, not on/off. 🗣️ anomaly signals. |
| 5–7 | Audit log with hash chain (tamper demo); kill switch → graceful degradation. |
| 7–10 | Scorecard + release gate: baseline 🛑, hardened 🚀. The gate is the session in one table. |
| 10–15 | Five conclusions, Q&A, readings. |

---

## If things go wrong

| Problem | What to do |
|---|---|
| `RuntimeError: This lab needs an OpenAI API key` | Secret missing or *Notebook access* not toggled. Locally: `export OPENAI_API_KEY=…` in the shell that starts Jupyter. |
| `model_not_found` in the connection cell | Model retired. Change `MODEL` in `config.py`, rebuild, re-share (or set `al.config.MODEL = "..."` right after setup as a stop-gap). |
| `RateLimitError` / slow responses | The client retries twice. If the room saturates a shared org key, switch to the executed `reference_runs/` copies and narrate. Reduce the loops (e.g. `al.GOLDEN[:5]`). |
| `SpendLimitExceeded` | Working as designed. Something looped. Look at `al.METER.by_purpose`; raise `al.METER.limit_usd` only if deliberate. |
| Model resists attacks in NB03 | See NB03 notes: teach "would you bet on it?". Guardrails don't depend on the model. |
| Numbers differ from your rehearsal | Expected: temperature 0 + seed is *mostly* deterministic, not fully. Discuss it: that's why pass^k exists. |
| A learner's notebook is in a weird state | *Runtime → Restart and run all*. Each notebook is independent. |

---

## Discussion prompts with expected answers

- **"What's the difference between tool correctness and task success?"** You can pass one and fail the other.
  Calling the right tools with wrong arguments (trace D) fails success. Succeeding by luck with a skipped
  verification step (trace B) is a latent failure.
- **"Why not just use the LLM judge for everything?"** It's a model with its own biases (verbosity, confident
  wording). It can't see what it isn't shown, and it costs money per case. Use it where only language can judge, and calibrate it.
- **"Where is the security boundary?"** In the code that authorises side effects against the system of record.
  It isn't in the prompt, and it isn't in the model.
- **"When do you *not* want a human in the loop?"** Low-risk, reversible, high-volume actions. Approval fatigue
  turns reviewers into rubber stamps. Tier the actions.
