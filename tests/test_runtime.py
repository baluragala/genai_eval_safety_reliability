"""Runtime behaviour, checked against the fake OpenAI client (no key, no network)."""
import agentlab as al
from agentlab.fixtures import GALLERY_ANSWERS, failure_gallery
from agentlab.guards import ApprovalGate, RefundPolicy, default_guardrails, safe_eval
from agentlab.oversight import AuditLog, release_gate
from agentlab.reliability import (Checkpointer, CircuitBreaker, FaultInjector, cached_order_lookup,
                                  idempotent, refunds_degraded, with_breakers, with_fallback, with_retry)

T = {t.id: t for t in al.GOLDEN}


def test_golden_baseline_passes():
    df = al.evaluate(al.GOLDEN, verbose=False)
    assert df.success.all(), df[~df.success][["task", "why"]]


def test_oracle_reads_world_not_words():
    tr, w, res = al.run_task(T["T07"], world_factory=lambda: al.World.fresh(declined=["A-1008"]))
    assert "processed" in tr.final.lower()          # the fake model claims success...
    assert not res["success"]                       # ...the ledger disagrees


def test_attacks_breach_baseline_and_hold_with_guards():
    base = al.red_team(verbose=False)
    guarded = al.red_team(verbose=False, guards=default_guardrails())
    assert base.breached.all()
    assert not guarded.breached.any()


def test_guards_do_not_over_block_golden():
    df = al.evaluate(al.GOLDEN, verbose=False, guards=default_guardrails())
    assert df.success.all(), df[~df.success][["task", "why"]]


def test_refund_policy_blocks_unlooked_up_refund():
    w = al.World.fresh()
    st = al.agent.new_state(T["T01"], w)
    ctx = al.GuardContext(T["T01"], w, st, al.Trace("x"))
    v = RefundPolicy().on_tool_call(ctx, "issue_refund", {"order_id": "A-1001", "amount": 129})
    assert v.action == "block" and "looked up" in v.reason


def test_safe_eval():
    assert safe_eval("129*0.15") == 129 * 0.15
    for bad in ["__import__('os')", "open('x')", "(1).__class__", "9**99999"]:
        try:
            safe_eval(bad)
            raise AssertionError(bad)
        except ValueError:
            pass


def test_crash_resume_double_refund_without_idempotency():
    for idem, expected in ((False, 2), (True, 1)):
        w, cp = al.World.fresh(), Checkpointer()
        fi = FaultInjector(crash_after={"issue_refund": 1})
        try:
            al.run_agent(T["T01"], w, executor=idempotent(fi) if idem else fi, checkpointer=cp)
            raise AssertionError("expected crash")
        except al.ProcessCrash:
            pass
        ex = idempotent(al.default_executor) if idem else al.default_executor
        al.run_agent(T["T01"], w, executor=ex, checkpointer=cp, state=cp.latest())
        assert len(w.refunds_for("A-1001")) == expected


def test_retry_recovers_transient_faults():
    ok_plain = ok_retry = 0
    for t in al.GOLDEN:
        for retry in (False, True):
            w = al.World.fresh()
            fi = FaultInjector(rates={"lookup_order": 0.5}, kinds=("server_error",), seed=3)
            ex = with_retry(fi) if retry else fi
            tr = al.run_agent(t, w, executor=ex)
            s = al.check_task(t, tr, w)["success"]
            ok_retry += s if retry else 0
            ok_plain += 0 if retry else s
    assert ok_retry >= ok_plain and ok_retry == len(al.GOLDEN)


def test_circuit_breaker_state_machine():
    clock = al.SimClock()
    b = CircuitBreaker(clock, failure_threshold=2, cooldown_ms=1000)
    b.failure(); b.failure()
    assert b.state == "open" and not b.allow()
    clock.sleep(1000)
    assert b.allow() and b.state == "half_open"
    b.success()
    assert b.state == "closed"


def test_degraded_refund_is_queued_not_lost():
    w = al.World.fresh()
    fi = FaultInjector(rates={"issue_refund": 1.0}, kinds=("server_error",))
    ex = with_fallback(with_breakers(fi, w.clock), {"issue_refund": refunds_degraded,
                                                    "lookup_order": cached_order_lookup})
    tr = al.run_agent(T["T01"], w, executor=ex)
    assert not w.refunds and any(e["order_id"] == "A-1001" for e in w.escalations)
    assert "queued" in tr.final.lower()


def test_human_approval_flow():
    for decision, refunds in (("approve", 1), ("reject", 0)):
        w = al.World.fresh()
        g = [ApprovalGate(150)]
        tr = al.run_agent(T["T07"], w, guards=g)
        assert tr.status == "awaiting_approval" and not w.refunds
        tr = al.resume_agent(tr, T["T07"], w, decision, guards=g)
        assert tr.status == "done" and len(w.refunds) == refunds


def test_audit_log_detects_tampering():
    log = AuditLog()
    for i in range(3):
        log.append("agent", "tool_call", {"i": i})
    assert log.verify() == (True, None)
    log.entries[1]["detail"]["i"] = 99
    assert log.verify() == (False, 1)


def test_release_gate():
    ok, rows = release_gate({"success_rate": 1, "tool_correctness": 1, "attack_success_rate": 0,
                             "chaos_success_rate": 0.9, "cost_per_task_usd": 0.001, "p95_latency_s": 3})
    assert ok and len(rows) == 6


def test_gallery_has_one_of_each_failure():
    g = failure_gallery()
    assert {x["label"] for x in g} == {"healthy", *GALLERY_ANSWERS}
    for x in g:
        res = al.check_task(x["task"], x["trace"], x["world"])
        assert res["success"] == (x["label"] == "healthy"), x["label"]


def test_spend_meter_caps():
    al.METER.limit_usd = 0.0
    try:
        al.chat([{"role": "user", "content": "hi"}])
        raise AssertionError("meter should refuse")
    except al.llm.SpendLimitExceeded:
        pass
