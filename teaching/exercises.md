# Take-home exercises

Each one extends the lab. Work in any notebook after running its setup and key cells. Everything you need
is in `agentlab` (`al`). Sketch solutions are in `solutions.md`, but try first.

---

### E1 · Grow the golden set (evaluation)
Add **five** new tasks to the golden set, each testing something the current ten don't. Ideas: a ticket with
two order IDs, a refund for an order that is still `processing`, a customer asking about someone else's
order by name, a warranty claim (not a refund), a polite ticket with no request at all.
For each, write an `expect` dict the oracle can check. Run `al.evaluate` on the new tasks.
*Deliverable:* the five tasks, their results, and one sentence per failure on **why** it failed (read the trace).

### E2 · Recalibrate the judge (LLM-as-judge)
Write a second rubric that scores **only** helpfulness and tone (no policy, no groundedness). Run both
rubrics on the same cases as Notebook 01 §5, with and without the trace.
*Deliverable:* a calibration table for all four combinations. Which one would you put in CI, and which one
belongs in a weekly human review?

### E3 · Write a new attack, then the guard for it (safety)
Design a seventh attack the current guardrails **don't** stop. Hint: think about `escalate_to_human`'s `reason`
field, the `remember` note of a *benign-looking* preference, or a refund that obeys policy but targets the
wrong order of the *same* customer. Add it to a copy of `ATTACKS` with a `breach` oracle, show that it
breaches under `default_guardrails()`, then write a `Guard` that stops it **without** reducing the golden-set success rate.

### E4 · Crashes in the chaos sweep (reliability)
Extend Notebook 05's chaos sweep so that, besides transient faults, **one in five** tasks crashes right after its
first side-effecting call (`FaultInjector(crash_after={"issue_refund": 1})`) and is resumed from the
checkpoint. Measure (a) success rate and (b) the number of duplicate refunds, with and without `idempotent(...)`.

### E5 · A fallback model you can trust (reliability + evaluation)
Wrap the agent's model in `ModelWithFallback(primary, fallback, fail_primary_times=2)` using
`al.config.FALLBACK_MODEL`. Run the golden set with the fallback model **only**. Is it good enough to serve
customers during an outage? What would you degrade (e.g. disable refunds) while on the fallback?

### E6 · Escalation matrix for a new domain (oversight)
Pick a different agent: a DevOps agent that can read logs, restart services, change feature flags, scale
clusters and run database migrations. Write its `oversight_mode()`: the tools, risk tiers, anomaly signals,
reversibility, and who approves. Justify every `approve` and every `block`.

### E7 · Your own release gate (operations)
Rewrite `GATE` for your current project's agent (or the DevOps agent from E6). Add at least one metric we did
not compute today (e.g. escalation rate, guard false-positive rate, p95 cost). Explain who owns each threshold
and what happens when the gate fails on a Friday afternoon.
