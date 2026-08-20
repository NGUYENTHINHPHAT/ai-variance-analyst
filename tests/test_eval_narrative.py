"""
Unit tests for eval/eval_narrative.py's own parsing/matching logic.

These are deterministic and need no API key — the point is to make sure the
groundedness checker itself is trustworthy before trusting what it says about
llm_analyst.py's output. A bug here (e.g. a regex that misses "$1.2M") would
silently make every groundedness score meaningless.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))

from eval_narrative import (
    parse_dollar_str,
    parse_pct_str,
    extract_dollar_amounts,
    extract_percentages,
    build_source_pools,
    is_grounded,
    check_groundedness,
    check_structure,
)


# ── parsing ──────────────────────────────────────────────────────────────────

def test_parse_dollar_str_variants():
    assert parse_dollar_str("$100,000") == 100_000.0
    assert parse_dollar_str("$1.2M") == 1_200_000.0
    assert parse_dollar_str("$500K") == 500_000.0
    assert parse_dollar_str("$(10,000)") == 10_000.0   # accounting-style negative -> magnitude
    assert parse_dollar_str("$+15,000") == 15_000.0
    assert parse_dollar_str("-$10,000") == 10_000.0
    assert parse_dollar_str("no digits here") is None


def test_parse_pct_str_variants():
    assert parse_pct_str("+15.0%") == 15.0
    assert parse_pct_str("-5.2 %") == 5.2
    assert parse_pct_str("20%") == 20.0
    assert parse_pct_str("no pct") is None


def test_extract_dollar_amounts_and_percentages_from_prose():
    text = ("Marketing spend rose to $134,900 vs a $95,000 budget, a +42.0% overrun, "
            "or roughly $1.3M annualized.")
    assert extract_dollar_amounts(text) == [134_900.0, 95_000.0, 1_300_000.0]
    assert extract_percentages(text) == [42.0]


# ── source pool construction ─────────────────────────────────────────────────

def test_build_source_pools_from_variance_data_and_summary_metrics():
    variance_data = [{
        "budget": "$100,000", "actual": "$115,000", "variance_amt": "$+15,000",
        "variance_pct": "+15.0%",
    }]
    summary_metrics = {
        "total_budget": 500_000.0,
        "total_variance_pct": 0.0523,  # fraction -> should become 5.23 in the pct pool
        "not_a_number": "ignored",
    }
    dollar_pool, pct_pool = build_source_pools(variance_data, summary_metrics)

    assert {100_000.0, 115_000.0, 15_000.0, 500_000.0}.issubset(dollar_pool)
    assert 15.0 in pct_pool
    assert 5.23 in pct_pool


# ── grounding tolerance ───────────────────────────────────────────────────────

def test_is_grounded_within_and_outside_tolerance():
    pool = {100_000.0}
    ok, matched = is_grounded(100_500.0, pool, rel_tol=0.03, abs_tol=1.0)  # within 3%
    assert ok and matched == 100_000.0

    ok, matched = is_grounded(250_000.0, pool, rel_tol=0.03, abs_tol=1.0)  # nowhere close
    assert not ok and matched is None


def test_check_groundedness_flags_fabricated_number():
    dollar_pool, pct_pool = build_source_pools(
        [{"budget": "$95,000", "actual": "$134,900", "variance_amt": "$+39,900", "variance_pct": "+42.0%"}],
        {},
    )
    text = "Spend hit $134,900 against a $95,000 budget (+42.0%), plus an unrelated $87,432 figure."
    result = check_groundedness(text, dollar_pool, pct_pool)

    assert result["total_claims"] == 4
    assert result["grounded_claims"] == 3
    assert 87_432.0 in result["ungrounded"]


# ── structure ────────────────────────────────────────────────────────────────

def test_check_structure_detects_all_headers_present():
    text = """## EXECUTIVE SUMMARY
This is a short summary.

## DRIVER ATTRIBUTION
Details here.

## RISK FLAGS
Risks here.

## RECOMMENDED ACTIONS
Actions here.
"""
    result = check_structure(text)
    assert result["all_headers_present"] is True
    assert result["exec_summary_words"] == 5


def test_check_structure_detects_missing_header():
    text = "## EXECUTIVE SUMMARY\nSummary only, nothing else.\n"
    result = check_structure(text)
    assert result["all_headers_present"] is False
    assert result["headers_present"]["RISK FLAGS"] is False
