"""Explanations for Notebooks 01-02 (evaluation, traces, failures).

Each function prints reasoning derived only from the data passed in.
"""
from __future__ import annotations

from .core import say


def explain_trajectory(results: dict):
    """results: {task_id: [issues]} from a trajectory check."""
    bad = {k: v for k, v in results.items() if v}
    say(f"{len(results) - len(bad)}/{len(results)} runs satisfied every trajectory rule.")
    for k, v in bad.items():
        say(f"  {k}: {'; '.join(v)}")
    if not bad:
        say("No run skipped verification, looped or stopped early. This check still matters: it would catch a run "
            "that reached the right outcome by the wrong path, which the outcome oracle can't see.")


def explain_judge(judged, calib_reply: dict, calib_trace: dict):
    """judged: DataFrame with case (or task), oracle, judge_reply_only, judge_with_trace."""
    key = "case" if "case" in judged else "task"
    say(f"{len(judged)} cases: {int(judged.oracle.sum())} are truly good (oracle pass) and "
        f"{int((~judged.oracle).sum())} are truly bad.")
    for name, col, cal in (("Reply-only judge", "judge_reply_only", calib_reply),
                           ("Trace-aware judge", "judge_with_trace", calib_trace)):
        fp = judged[(judged[col]) & (~judged.oracle)][key].tolist()
        fn = judged[(~judged[col]) & (judged.oracle)][key].tolist()
        say(f"{name}: agreement {cal['agreement']}, κ {cal['kappa']}. "
            f"False passes {fp or 'none'}; false fails {fn or 'none'}.")
    fp_r = set(judged[(judged.judge_reply_only) & (~judged.oracle)][key])
    fp_t = set(judged[(judged.judge_with_trace) & (~judged.oracle)][key])
    if fp_r - fp_t:
        say(f"Giving the judge the trace fixed {sorted(fp_r - fp_t)}: those replies read fine, and only the trace "
            "shows the action was wrong or never happened.")
    if fp_t:
        say(f"Even with the trace, the judge approved {sorted(fp_t)}. Keep the world-state oracle for outcomes, and use "
            "the judge for what only language can judge.")
    if not fp_r:
        say("The reply-only judge made no false passes this time. That's a result of this run, not a guarantee; "
            "a calibration set exists to measure exactly this.")
    k = calib_trace["kappa"]
    if k == k:  # not NaN
        band = "strong" if k >= 0.8 else "moderate" if k >= 0.6 else "weak" if k >= 0.4 else "poor"
        say(f"κ {k} for the trace-aware judge is {band} agreement beyond chance "
            "(rough bands: ≥0.8 strong, ≥0.6 moderate, ≥0.4 weak).")


def explain_robustness(robust):
    """robust: DataFrame from evaluate() over perturbed variants, with base and style columns."""
    n, ok = len(robust), int(robust.success.sum())
    say(f"{ok}/{n} perturbed tickets still reached the right outcome.")
    fails = robust[~robust.success]
    if fails.empty:
        say("No perturbation style broke the agent. Rephrasing, typos and padding didn't change what it did. "
            "A strong model handles surface noise well; multi-turn and adversarial cases are harder, as the next steps show.")
        return
    by_style = fails.groupby("style").size().sort_values(ascending=False)
    for style, cnt in by_style.items():
        say(f"style '{style}' broke {cnt} ticket(s): {', '.join(fails[fails['style'] == style].base)}")
    for _, r in fails.iterrows():
        say(f"  {r.task}: {r.why}")
    say("The oracle is the same for every variant, so a failure means the wording changed the agent's behaviour, "
        "not that the task changed.")


def explain_simulation(sc, transcript, result: dict):
    turns = sum(1 for t in transcript if t["who"] == "customer")
    say(f"{sc.id}: the simulated customer wrote {turns} message(s). Goal: {sc.goal}.")
    asked = any("order" in t["text"].lower() and "?" in t["text"] for t in transcript if t["who"] == "agent")
    if sc.opening and "A-" not in sc.opening:
        say("The opening message had no order ID, so the agent had to ask for it. "
            + ("It did ask." if asked else "It never asked, which a single-turn golden ticket would not have revealed."))
    say("Oracle: PASS, the world ended in the right state." if result["success"]
        else f"Oracle: FAIL, {'; '.join(result['reasons'])}.")


def explain_passk(trials: dict):
    """trials: {task_id: [bool, ...]}"""
    from ..evals import pass_at_k, pass_hat_k
    for tid, ok in trials.items():
        n, c = len(ok), sum(ok)
        say(f"{tid}: {c}/{n} succeeded. pass@2={pass_at_k(n, c, 2):.2f}, pass^2={pass_hat_k(n, c, 2):.2f}, "
            f"pass^{n}={pass_hat_k(n, c, n):.2f}.")
        if c == n:
            say(f"  Consistent on all {n} tries, so pass@k and pass^k agree. Four tries is a small sample, though: "
                "rare failures need more trials to show up.")
        elif c == 0:
            say("  Failed every time. That's a systematic bug, not variance, so look at the trace.")
        else:
            say(f"  Inconsistent: pass@k looks good because one success is enough, while pass^k exposes that a "
                f"customer retrying would see a failure {n - c} time(s) in {n}.")


def explain_gallery_guess(label: str, answers: dict, guess: str | None = None):
    truth = answers.get(label, "healthy")
    if guess:
        say(f"Run {label}: you said '{guess}', the answer is '{truth}'. "
            + ("✓" if guess.lower() in truth.lower() else "Compare your step-by-step reasoning with the evidence below."))
    else:
        say(f"Run {label}: no guess yet. Fill in my_guesses and re-run (the answer stays hidden until you do).")


def explain_diagnosis(diag, answers: dict | None = None):
    """diag: DataFrame with run, root_cause, also, evidence (and optionally expected/correct)."""
    for _, r in diag.iterrows():
        extra = f" (also: {', '.join(r['also'])})" if len(r["also"]) else ""
        say(f"{r['run']}: root cause '{r['root_cause']}'{extra}. Evidence: {r['evidence'] or 'no detector fired'}.")
    if "correct" in diag:
        wrong = diag[~diag.correct]
        say("The detectors reproduce the answer key for every run." if wrong.empty else
            f"Mismatch on {', '.join(wrong.run)}: read the evidence column to see which detector fired first.")
    multi = diag[diag["also"].apply(len) > 0]
    if len(multi):
        say(f"{', '.join(multi.run)} tripped several detectors. The priority order picks the earliest cause, because "
            "fixing that one usually removes the downstream symptoms.")


def explain_live_failure(name: str, diagnosis: dict, oracle: dict | None = None):
    rc = diagnosis["root_cause"]
    if rc == "healthy":
        say(f"{name}: no detector fired. The model avoided this failure on this run. That could be good judgement, "
            "or luck at temperature 0. Nothing in the system forced it.")
    else:
        say(f"{name}: the real model produced a '{rc}'. Evidence: {diagnosis['evidence']}.")
    if oracle is not None:
        say(f"Oracle agrees it {'passed' if oracle['success'] else 'failed'}"
            + ("" if oracle["success"] else f": {'; '.join(oracle['reasons'])}") + ".")
