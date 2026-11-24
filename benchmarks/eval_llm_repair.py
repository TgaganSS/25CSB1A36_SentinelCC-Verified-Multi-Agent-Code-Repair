"""
Does an optional language model repair programs the rule-based healer cannot?  (measured, not assumed)

For every program in test_cases/ and test_cases/mini/ the agentic pipeline is run twice:
  A) rules only          (the default, offline)
  B) rules + LLM         (a model proposes a fix for what the rules could not repair; every proposal must pass the
                          guardrails: safety gate, no new sentinel indicators, < 50% of the file changed, fewer
                          compiler errors, whole project still links)
Only programs that fail to build at the start are evaluated; blocked (unsafe) inputs are reported and skipped.

Usage (Colab or any machine with g++):
    export GEMINI_API_KEY=...                      # free key from Google AI Studio
    python benchmarks/eval_llm_repair.py --provider gemini
    python benchmarks/eval_llm_repair.py --provider none          # baseline only, no network
Options: --limit N (first N failing programs), --model NAME, --min-interval SECONDS (free-tier pacing)
Writes benchmarks/llm_repair_results.json and .md
"""
import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("GSC_DISABLE_DATASET_LOGGING", "1")
os.environ.setdefault("GSC_CACHE_DIR", tempfile.mkdtemp(prefix="gsc_eval_cache_"))

from agents import llm as L                                   # noqa: E402
from agents.orchestrator import Options, run_agentic         # noqa: E402


def one(path, provider):
    work = tempfile.mkdtemp(prefix="gsc_eval_")
    p = os.path.join(work, os.path.basename(path))
    shutil.copy(path, p)
    try:
        res = run_agentic(p, Options(llm=provider, harden=False, optimize=False, detect_build=False))
        b = res.baseline
        out = {"verdict": res.verdict,
               "built_before": bool(getattr(b, "built", False)) if b is not None else None,
               "built_after": bool(res.after.built) if res.after is not None else None,
               "errors_before": getattr(b, "error_count", None) if b is not None else None,
               "errors_after": getattr(res.after, "error_count", None) if res.after is not None else None,
               "llm_accepted": sum(1 for c in res.board.changes if c.rule == "REPAIR-LLM"),
               "llm_rejected": [c.why if hasattr(c, "why") else str(c) for c in res.board.rejected if c.rule == "REPAIR-LLM"],
               "rule_fixes": sum(1 for c in res.board.changes if c.rule.startswith("REPAIR-") and c.rule != "REPAIR-LLM")}
        res.proj.close()
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="none", choices=("none", "gemini", "anthropic"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-interval", type=float, default=None)
    a = ap.parse_args()
    if a.model:
        os.environ["GSC_LLM_MODEL"] = a.model
    if a.min_interval is not None:
        os.environ["GSC_LLM_MIN_INTERVAL"] = str(a.min_interval)
    files = sorted(glob.glob(os.path.join(ROOT, "test_cases", "*.cpp")) + glob.glob(os.path.join(ROOT, "test_cases", "mini", "*.cpp")))
    rows, blocked = [], []
    for f in files:
        name = os.path.relpath(f, os.path.join(ROOT, "test_cases"))
        base = one(f, None)
        if base["verdict"] == "BLOCKED":
            blocked.append(name)
            continue
        if base["built_before"]:
            continue                                   # builds already: nothing to repair
        rows.append({"name": name, "rules": base})
    if a.limit:
        rows = rows[:a.limit]
    prov = L.make_provider(None if a.provider == "none" else a.provider)
    if prov is not None and hasattr(prov, "check"):
        ok, msg = prov.check()
        print(("preflight ok: " if ok else "PREFLIGHT FAILED: ") + msg, flush=True)
        if not ok:
            sys.exit("The model service did not answer, so nothing was evaluated. Fix the key/model above and run again.")
    t0 = time.time()
    for r in rows:
        r["llm"] = one(os.path.join(ROOT, "test_cases", r["name"]), prov) if prov else r["rules"]
        print(f"{r['name']:<38} rules:{'built' if r['rules']['built_after'] else 'fail ':<5}  "
              f"+LLM:{'built' if r['llm']['built_after'] else 'fail ':<5} accepted={r['llm']['llm_accepted']}", flush=True)
    n = len(rows)
    rb = sum(1 for r in rows if r["rules"]["built_after"])
    lb = sum(1 for r in rows if r["llm"]["built_after"])
    summary = {"provider": a.provider, "model": getattr(prov, "model", None), "programs_failing_at_start": n,
               "blocked_skipped": blocked, "repaired_by_rules": rb, "repaired_with_llm": lb,
               "extra_from_llm": lb - rb, "llm_calls": getattr(prov, "stats", {}), "last_provider_error": getattr(prov, "last_error", ""), "seconds": round(time.time() - t0, 1)}
    json.dump({"summary": summary, "rows": rows}, open(os.path.join(ROOT, "benchmarks", "llm_repair_results.json"), "w"), indent=1)
    with open(os.path.join(ROOT, "benchmarks", "llm_repair_results.md"), "w") as w:
        w.write(f"# LLM repair evaluation ({a.provider}{' / ' + str(summary['model']) if summary['model'] else ''})\n\n")
        w.write(f"Programs failing to build at the start: **{n}** (blocked and skipped: {len(blocked)}).\n\n")
        w.write(f"Repaired by rules only: **{rb}/{n}**. Repaired with rules + LLM: **{lb}/{n}** (extra: {lb - rb}).\n\n")
        st = summary["llm_calls"]
        if st and st.get("calls") and st.get("failed") == st.get("calls"):
            w.write(f"> **Every request failed** ({summary['last_provider_error']}). These numbers say nothing about the model.\n\n")
        w.write("| Program | Errors at start | Rules only | Rules + LLM | Proposals accepted | Rejected (reason) |\n|---|---|---|---|---|---|\n")
        for r in rows:
            w.write(f"| {r['name']} | {r['rules']['errors_before']} | {'built' if r['rules']['built_after'] else 'fails'} | "
                    f"{'built' if r['llm']['built_after'] else 'fails'} | "
                    f"{'n/a' if a.provider == 'none' else r['llm']['llm_accepted']} | "
                    f"{'n/a (no model)' if a.provider == 'none' else '; '.join(r['llm']['llm_rejected'])[:80]} |\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
