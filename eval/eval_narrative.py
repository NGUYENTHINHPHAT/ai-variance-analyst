"""
Groundedness + structure eval for llm_analyst.py's generated narrative.

The README's headline claim is "no hallucination, full auditability." This
script is what actually tests that, instead of it being an unverified claim:
every dollar amount and percentage the model cites in its report must trace
back to the variance_data/summary_metrics JSON that was in its prompt. A
number that matches nothing in the source data (within a small rounding
tolerance) is a hallucination candidate.

This is NOT a pytest unit test. It calls the real Claude API (costs money,
needs ANTHROPIC_API_KEY) and depends on live/regenerated data, so it isn't
deterministic run-to-run and doesn't belong in CI as a hard gate. Run it
manually after touching llm_analyst.py's prompts or system rules:

    python eval/eval_narrative.py

Exits non-zero if groundedness drops below GROUNDEDNESS_THRESHOLD or a
required report section is missing, so it *can* be wired into CI later once
someone decides that per-PR API spend is worth it.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from variance_engine import load_data, build_variance_report, compute_summary_metrics, get_top_variances  # noqa: E402
from rag_pipeline import FinancialRAG  # noqa: E402
from llm_analyst import VarianceAnalyst  # noqa: E402

GROUNDEDNESS_THRESHOLD = 0.85
REQUIRED_HEADERS = ["EXECUTIVE SUMMARY", "DRIVER ATTRIBUTION", "RISK FLAGS", "RECOMMENDED ACTIONS"]
EXEC_SUMMARY_WORD_LIMIT = 150
EXEC_SUMMARY_WORD_BUFFER = 1.3  # the prompt says "under 150 words"; LLMs overshoot, warn rather than fail past this

DOLLAR_RE = re.compile(r"\(?[+-]?\$\s?[+-]?\d[\d,]*(?:\.\d+)?\s?[MmKk]?\)?")
PCT_RE = re.compile(r"[+-]?\d+(?:\.\d+)?\s?%")


# ── number parsing (shared by both source-data pool building and extraction) ─

def parse_dollar_str(s: str):
    """'$1.2M' -> 1200000.0, '$(10,000)' -> 10000.0, '$+15,000' -> 15000.0. Sign is discarded on purpose: groundedness matches by magnitude only."""
    s = s.strip()
    multiplier = 1.0
    if s and s[-1] in "Mm":
        multiplier = 1_000_000.0
        s = s[:-1]
    elif s and s[-1] in "Kk":
        multiplier = 1_000.0
        s = s[:-1]
    digits = re.sub(r"[^\d.]", "", s)
    if not digits:
        return None
    try:
        return float(digits) * multiplier
    except ValueError:
        return None


def parse_pct_str(s: str):
    digits = re.sub(r"[^\d.]", "", s)
    if not digits:
        return None
    try:
        return float(digits)
    except ValueError:
        return None


def extract_dollar_amounts(text: str) -> list[float]:
    return [v for v in (parse_dollar_str(m) for m in DOLLAR_RE.findall(text)) if v is not None]


def extract_percentages(text: str) -> list[float]:
    return [v for v in (parse_pct_str(m) for m in PCT_RE.findall(text)) if v is not None]


# ── source-of-truth number pools ─────────────────────────────────────────────

def build_source_pools(variance_data: list[dict], summary_metrics: dict):
    """Every $ and % figure the model was actually given, as two flat pools."""
    dollar_pool, pct_pool = set(), set()

    for row in variance_data:
        for key in ("budget", "actual", "variance_amt"):
            v = parse_dollar_str(str(row.get(key, "")))
            if v is not None:
                dollar_pool.add(round(v, 2))
        v = parse_pct_str(str(row.get("variance_pct", "")))
        if v is not None:
            pct_pool.add(round(v, 2))

    for key, val in summary_metrics.items():
        if not isinstance(val, (int, float)):
            continue
        if key.endswith("_pct"):
            pct_pool.add(round(abs(val) * 100, 2))
        else:
            dollar_pool.add(round(abs(val), 2))

    return dollar_pool, pct_pool


def is_grounded(value: float, pool: set, rel_tol: float, abs_tol: float):
    """True + the closest source value if `value` is within tolerance of anything in `pool`."""
    best = None
    for p in pool:
        tol = max(abs_tol, rel_tol * max(abs(p), 1.0))
        if abs(value - p) <= tol:
            if best is None or abs(value - p) < abs(value - best):
                best = p
    return best is not None, best


# ── checks ────────────────────────────────────────────────────────────────

def check_groundedness(text: str, dollar_pool: set, pct_pool: set) -> dict:
    dollars = extract_dollar_amounts(text)
    pcts = extract_percentages(text)

    dollar_results = [(v, *is_grounded(v, dollar_pool, rel_tol=0.03, abs_tol=1.0)) for v in dollars]
    pct_results = [(v, *is_grounded(v, pct_pool, rel_tol=0.02, abs_tol=0.5)) for v in pcts]

    all_results = dollar_results + pct_results
    total = len(all_results)
    grounded = sum(1 for _, ok, _ in all_results if ok)

    return {
        "total_claims": total,
        "grounded_claims": grounded,
        "score": (grounded / total) if total else None,
        "ungrounded": [v for v, ok, _ in all_results if not ok],
        "dollar_claims": dollar_results,
        "pct_claims": pct_results,
    }


def check_structure(text: str) -> dict:
    upper = text.upper()
    headers_present = {h: (h in upper) for h in REQUIRED_HEADERS}

    exec_words = None
    m = re.search(r"EXECUTIVE SUMMARY\s*\n(.*?)(?:\n##|\Z)", text, re.IGNORECASE | re.DOTALL)
    if m:
        exec_words = len(m.group(1).split())

    return {
        "headers_present": headers_present,
        "all_headers_present": all(headers_present.values()),
        "exec_summary_words": exec_words,
        "exec_summary_over_limit": (
            exec_words is not None and exec_words > EXEC_SUMMARY_WORD_LIMIT * EXEC_SUMMARY_WORD_BUFFER
        ),
    }


# ── report printing ─────────────────────────────────────────────────────────

def print_groundedness(label: str, g: dict) -> bool:
    print(f"\n  [{label}] groundedness")
    if g["total_claims"] == 0:
        print("    ⚠ no dollar/percentage claims found in the text at all (unexpected for this prompt)")
        return False
    print(f"    {g['grounded_claims']}/{g['total_claims']} numeric claims traced to source data "
          f"({g['score']:.0%})")
    if g["ungrounded"]:
        print(f"    ✗ ungrounded values (not in source data within tolerance): {g['ungrounded']}")
    ok = g["score"] >= GROUNDEDNESS_THRESHOLD
    print(f"    {'✅ PASS' if ok else '❌ FAIL'} (threshold {GROUNDEDNESS_THRESHOLD:.0%})")
    return ok


def print_structure(s: dict) -> bool:
    print("\n  structure")
    for h, present in s["headers_present"].items():
        print(f"    {'✅' if present else '❌'} '## {h}' present: {present}")
    if s["exec_summary_words"] is not None:
        flag = "⚠" if s["exec_summary_over_limit"] else "✅"
        print(f"    {flag} Executive Summary length: {s['exec_summary_words']} words "
              f"(prompt asks for <{EXEC_SUMMARY_WORD_LIMIT})")
    else:
        print("    ⚠ could not locate an Executive Summary section to measure")
    return s["all_headers_present"]


# ── eval runs ────────────────────────────────────────────────────────────────

def run_full_report_eval(analyst: VarianceAnalyst, rag: FinancialRAG) -> bool:
    print("\n" + "=" * 70)
    print("EVAL 1: generate_full_report()")
    print("=" * 70)

    df = load_data(os.path.join(ROOT, "data", "processed", "erp_combined.csv"))
    report = build_variance_report(df)
    metrics = compute_summary_metrics(df)
    top_variances = get_top_variances(report, n=8)  # mirrors app.py's default slider value

    # Mirror app.py's exact context-building: retrieve per top-5 variance, truncate to 400 chars.
    full_context = ""
    for var in top_variances[:5]:
        ctx = rag.retrieve_for_variance(
            var["account"], var["department"], var["period"],
            float(var["variance_pct"].replace("%", "").replace("+", "")) / 100,
        )
        full_context += f"\n\n### Context for {var['account']} ({var['department']}, {var['period']}):\n{ctx[:400]}"

    report_text = analyst.generate_full_report(top_variances, metrics, full_context)
    print("\n--- generated report ---\n")
    print(report_text)
    print("\n--- checks ---")

    dollar_pool, pct_pool = build_source_pools(top_variances, metrics)
    g = check_groundedness(report_text, dollar_pool, pct_pool)
    g_ok = print_groundedness("full report", g)
    s_ok = print_structure(check_structure(report_text))
    return g_ok and s_ok


def run_drilldown_eval(analyst: VarianceAnalyst, rag: FinancialRAG) -> bool:
    print("\n" + "=" * 70)
    print("EVAL 2: explain_single_variance()")
    print("=" * 70)

    df = load_data(os.path.join(ROOT, "data", "processed", "erp_combined.csv"))
    report = build_variance_report(df)
    if report.empty:
        print("  ⚠ no flagged variances in current data — skipping drill-down eval")
        return True
    row = report.iloc[0]

    ctx = rag.retrieve_for_variance(row["account_name"], row["department_name"],
                                     row["period"], float(row["variance_pct"]))
    text = analyst.explain_single_variance(
        account=row["account_name"], department=row["department_name"], period=row["period"],
        budget=float(row["budget_amount"]), actual=float(row["actual_amount"]), rag_context=ctx,
    )
    print("\n--- generated explanation ---\n")
    print(text)
    print("\n--- checks ---")

    variance = float(row["actual_amount"]) - float(row["budget_amount"])
    variance_data = [{
        "budget": f"${row['budget_amount']:,.0f}",
        "actual": f"${row['actual_amount']:,.0f}",
        "variance_amt": f"${variance:+,.0f}",
        "variance_pct": f"{row['variance_pct']:+.1%}",
    }]
    dollar_pool, pct_pool = build_source_pools(variance_data, {})
    g = check_groundedness(text, dollar_pool, pct_pool)
    return print_groundedness("drill-down", g)


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set — this eval calls the real Claude API and needs it.")
        print("Export it (or put it in .env) and re-run: python eval/eval_narrative.py")
        return 2

    print("Loading RAG index (this builds/loads the ChromaDB store)...")
    rag = FinancialRAG(docs_dir=os.path.join(ROOT, "rag_docs"),
                        persist_dir=os.path.join(ROOT, "data", "chroma_db"))
    analyst = VarianceAnalyst()

    ok1 = run_full_report_eval(analyst, rag)
    ok2 = run_drilldown_eval(analyst, rag)

    print("\n" + "=" * 70)
    print(f"RESULT: {'✅ ALL CHECKS PASSED' if (ok1 and ok2) else '❌ SOME CHECKS FAILED'}")
    print("=" * 70)
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    sys.exit(main())
