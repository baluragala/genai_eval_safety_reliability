"""Explanations for the reliability (05) and oversight (06) notebooks.

Every function reads the actual results and prints the reasoning. Nothing
here assumes what a live model did.
"""
from __future__ import annotations

from .core import say


# ---------------------------------------------------------------- Notebook 05: reliability

def explain_faults(df, label="this run"):
    """df from suite(): one row per task with success, faults, why."""
    n, ok = len(df), int(df["success"].sum())
    faults = int(df["faults"].sum())
    hit = df[df["faults"] > 0]
    say(f"{label}: {ok}/{n} tasks passed while {faults} faults were injected across {len(hit)} of the {n} tasks.")
    failed = df[~df["success"]]
    if len(failed):
        with_fault = failed[failed["faults"] > 0]
        say(f"{len(with_fault)} of the {len(failed)} failed task(s) were hit by at least one fault. "
            "With no protection, the model sees the raw error and has to improvise.")
        for _, r in failed.iterrows():
            say(f"  ❌ {r['task']} ({r['faults']} fault(s)): {r['why'] or 'see trace'}")
    lucky = hit[hit["success"]]
    if len(lucky):
        say(f"{len(lucky)} task(s) survived their faults anyway ({', '.join(lucky['task'])}). The model retried or "
            "worked around the error itself. That's luck, not design: the harness guaranteed nothing.")
    if faults == 0:
        say("No faults landed on this seed, so this run tells you nothing about resilience. Try another seed.")


def explain_backoff(table):
    """table: rows 'no jitter (ms)' / 'full jitter (ms)', columns = attempt."""
    nj = list(table.loc["no jitter (ms)"])
    fj = list(table.loc["full jitter (ms)"])
    say(f"Without jitter the wait doubles each attempt ({', '.join(f'{x:.0f}' for x in nj)} ms) until it hits the cap "
        f"of {max(nj):.0f} ms.")
    say("With full jitter each wait is a random point between 0 and that same ceiling: "
        f"{', '.join(f'{x:.0f}' for x in fj)} ms. Clients that failed together now retry at different moments, "
        "so they don't hit the recovering service all at once.")


def explain_retry_effect(before, after):
    b, a = before["success"].mean(), after["success"].mean()
    say(f"Success: {b:.0%} unprotected → {a:.0%} with read retries.")
    say(f"Average virtual time per task: {before['virtual_s'].mean():.1f}s → {after['virtual_s'].mean():.1f}s. "
        "The extra time is spent backing off. Resilience is paid for in latency.")
    still = after[~after["success"]]
    refund_related = still[still["why"].str.contains(r"refund on .*found 0", case=False, na=False)]
    if len(still) == 0:
        say("Nothing failed on this seed. Faults landed only on reads, which retries now fix.")
    else:
        say(f"{len(still)} task(s) still fail. {len(refund_related)} of them are missing their refund: issue_refund is a write, "
            "and with_retry deliberately doesn't retry writes yet." + (" The rest are failures the model didn't recover from."
                                                                        if len(still) > len(refund_related) else ""))


def explain_crash_resume(n_without: int, n_with: int):
    say(f"Without idempotency the ledger shows {n_without} refund(s) for A-1001; with it, {n_with}.")
    if n_without > n_with:
        say("Why: the crash happened after the refund took effect but before the checkpoint recorded the result. "
            "On resume the checkpoint still held the model's request, so the same tool call ran again.")
        say("With idempotent() the replayed call carries the same key (task_id:tool_call_id), so the payments system "
            "returns the original result instead of paying twice.")
    else:
        say("Both counts match, which is unexpected. Check that the crash fired (look for 💥 above).")


def explain_breaker(history, log_df):
    say("State transitions (time in ms, from → to): " + ", ".join(f"{t}ms {a}→{b}" for t, a, b in history))
    ff = int((log_df.loc["result"] == "fast-fail").sum()) if "result" in log_df.index else 0
    errs = int((log_df.loc["result"] == "❌ error").sum()) if "result" in log_df.index else 0
    say(f"{errs} call(s) actually reached the failing dependency; {ff} were failed fast by the open breaker "
        "without touching it. Every fast-fail is load the dependency never had to absorb while recovering.")
    say("Half-open lets exactly one probe through. A failed probe re-opens the breaker for another cooldown, "
        "and a successful one closes it.")


def explain_outage(table):
    r, b = table["retry only"], table["retry + breaker"]
    say(f"Calls that reached the dead database: {int(r['calls that hit the dead DB'])} with retry only, "
        f"{int(b['calls that hit the dead DB'])} with a breaker.")
    say(f"Virtual time for all five customers: {r['virtual seconds (all 5)']}s vs {b['virtual seconds (all 5)']}s.")
    say("The breaker doesn't make any request succeed, because the database is still down. It makes them fail fast. "
        "That frees capacity, gives the database room to recover, and lets a fallback answer sooner.")


def explain_degraded(results):
    """results: list of dicts from degraded_run()."""
    for r in results:
        say(f"{r['task']} with {r['down']} down: status {r['status']}, "
            f"{r['refunds']} refund(s), {r['queued']} item(s) queued for a human.")
        if r["down"] == "lookup_order":
            if r.get("served_by_fallback"):
                say("  The order data came from the read replica (the tool result carries a _degraded label), so the "
                    "customer still got an answer, flagged as possibly stale.")
            else:
                say("  The fallback was never used. Either no lookup reached the dead primary, or the model didn't ask for one.")
        if r["down"] == "issue_refund":
            if r["refunds"] == 0 and r["queued"]:
                say("  No money moved, but the refund is in the human queue. The oracle marks this ❌ (no refund), "
                    "and that's the honest outcome: degraded, not faked.")
            if r["claims_done"]:
                say("  ⚠️ The reply sounds as if the refund is done. Check the wording: it should say queued, not processed.")


def explain_model_fallback(used, trace):
    say(f"Model used per step: {used}.")
    if used and len(set(used)) > 1:
        say(f"Step 1 was answered by the fallback, {used[0]}, because the primary was made to fail once. Later steps "
            f"went back to {used[-1]}: each call tries the primary first, so one outage doesn't pin the whole run to the backup.")
    say(f"The run ended with status {trace.status}. A fallback model is a different model: evaluate it on the golden "
        "set before you trust it during an incident.")


def explain_stops(df):
    for name, r in df.iterrows():
        say(f"{name}: stopped with status '{r['status']}' after {r['llm calls']} model call(s) and "
            f"{r['lookup calls']} lookup(s), costing ${r['cost $']}.")
    cut = [n for n, r in df.iterrows() if r["status"] in ("max_steps", "budget_exceeded")]
    replied = [n for n, r in df.iterrows() if r["status"] == "done"]
    if cut:
        say(f"{', '.join(cut)} cut the run off without a reply: cost is bounded, but the customer is left hanging.")
    if replied:
        say(f"{', '.join(replied)} ended with a reply. The model stopped retrying, either on its own or because a "
            "guard blocked the repeat and told it to escalate.")
    say("All three brakes bound the damage. Only one that leads to a reply leaves the customer a way forward.")


def explain_chaos(success_table, latency_table):
    rates = list(success_table.index)
    for cfg in success_table.columns:
        col = success_table[cfg]
        say(f"{cfg}: success {' → '.join(f'{v:.0%}' for v in col)} as faults go {' → '.join(f'{r:.0%}' for r in rates)}; "
            f"avg virtual time {' → '.join(f'{v:.1f}s' for v in latency_table[cfg])}.")
    last = success_table.iloc[-1]
    best = [c for c in last.index if last[c] == last.max()]
    say(f"At the highest fault rate the most resilient: {', '.join(best)} ({last.max():.0%}). "
        + ("Tied configurations were equally resilient here, so compare their latency in the second table."
           if len(best) > 1 else "Check what it paid for that in latency in the second table."))
    if "full stack" in success_table.columns and (success_table["full stack"] < 1).any():
        say("Some full-stack runs still count as ❌. Refunds degraded into the human queue fail the oracle because no refund "
            "happened. Whether honest degradation counts as success is a product decision, not an engineering one.")


# ---------------------------------------------------------------- Notebook 06: oversight

def explain_pause(trace, world):
    p = trace.state.pending if trace.state else None
    say(f"The run is '{trace.status}'. The agent asked for {p['name']}({p['args']}), and {p['by']} escalated it: "
        f"{p['reason']}." if p else f"The run is '{trace.status}' and nothing is pending.")
    say(f"Ledger right now: {len(world.refunds)} refund(s). The tool never ran; the call waits in the saved state "
        "until a human decides.")


def explain_decisions(df):
    for _, r in df.iterrows():
        amounts = [x["amount"] for x in r["ledger"]]
        say(f"{r['decision']}: {len(amounts)} refund(s) {amounts} in the ledger; run {r['status']}.")
    edit = df[df["decision"] == "edit"]
    if len(edit):
        reply = str(edit.iloc[0]["agent_reply"])
        amt = [x["amount"] for x in edit.iloc[0]["ledger"]]
        if amt and f"{amt[0]:.2f}" not in reply and f"{amt[0]:.0f}" not in reply:
            say(f"⚠️ After the edit the ledger shows ${amt[0]:.2f}, but the reply doesn't mention that amount. "
                "The customer may be told the wrong figure, which is a silent failure in the reply.")
        else:
            say("After the edit the reply matches the ledger amount. The model read the edited tool result.")
    say("In every case the model saw the human's decision as an ordinary tool result and carried on from there.")


def explain_matrix(df):
    for mode in ["auto", "notify", "approve", "block"]:
        rows = df[df["mode"] == mode]
        if len(rows):
            say(f"{mode}: " + "; ".join(f"{r.tool}(${r.amount}, anomalies={r.anomalies}, reversible={r.reversible})"
                                         for r in rows.itertuples()))
    say("The pattern: read-only tools are auto; small reversible writes notify; money over the limit, "
        "anything irreversible, or anything with anomaly signals needs approval; code execution is blocked outright.")


def explain_audit(log, before, after):
    say(f"The log holds {len(log.entries)} entries, each hashing its content plus the previous entry's hash.")
    say(f"Before tampering, verify() returned {before}: the chain is intact.")
    if after[0] is False:
        say(f"After the edit it returned {after}: entry {after[1]} no longer matches its own hash. The hashes after "
            "it still chain to the old value, so the rewrite is detectable at exactly that point.")


def explain_killswitch(trace, world):
    blocked = [s for s in trace.guard_events() if "kill switch" in (s.note or "")]
    say(f"The kill switch blocked {len(blocked)} call(s). Ledger: {len(world.refunds)} refund(s); "
        f"human queue: {len(world.escalations)} item(s).")
    if world.escalations and not world.refunds:
        say("The agent degraded as intended: no refund, the case handed to a human, the customer told what happens next.")
    elif not world.escalations:
        say("No escalation was recorded. The agent stopped but didn't hand the case on. Check its reply.")


def explain_scorecard(metrics: dict):
    b, h = metrics.get("baseline", {}), metrics.get("hardened", {})
    for k in ["success_rate", "tool_correctness", "attack_success_rate", "chaos_success_rate",
              "cost_per_task_usd", "p95_latency_s"]:
        if k in b and k in h:
            say(f"{k}: baseline {b[k]} → hardened {h[k]}")
    if "attack_success_rate" in h and h["attack_success_rate"] < b.get("attack_success_rate", 0):
        say("Guardrails cut the attack success rate.")
    if "chaos_success_rate" in h and h["chaos_success_rate"] > b.get("chaos_success_rate", 0):
        say("The reliability stack raised success under faults.")
    if h.get("cost_per_task_usd", 0) > b.get("cost_per_task_usd", 0):
        say("The hardened agent costs more per task. Blocked attempts and retries add model calls.")


def explain_gate(label, ok, rows):
    failed = [r for r in rows if r["pass"] != "✅"]
    if ok:
        say(f"{label}: every threshold met, so it ships.")
    else:
        say(f"{label}: blocked by {len(failed)} threshold(s): " +
            "; ".join(f"{r['metric']}={r['value']} (needs {r['rule']})" for r in failed))
