"""
AI Variance Analyst — FP&A Copilot
Streamlit Dashboard with LLM-powered variance analysis
"""

import os
import sys
import json
import re
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import subprocess

from variance_engine import (
    build_variance_report, compute_summary_metrics, get_top_variances
)
from seed_erp_db import build_database
from sql_engine import connect, load_variance_base, run_data_quality_checks, read_sql
from rag_pipeline import FinancialRAG
from llm_analyst import VarianceAnalyst

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Variance Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Theme colors live in .streamlit/config.toml; this only tightens spacing.
st.markdown("""
<style>
  .block-container { padding-top: 2rem; padding-bottom: 2rem; }
  [data-testid="stMetricValue"] { font-size: 1.7rem; }
</style>
""", unsafe_allow_html=True)

# ── Palette: red and green only ever mean "worse" / "better" than budget ──────
GOOD, BAD, NEUTRAL = "#16A34A", "#DC2626", "#64748B"
BLUE, GRAY_LINE = "#2563EB", "#E5E7EB"

SEVERITY_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
SEVERITY_STYLE = {
    "CRITICAL": "background-color:#FEE4E2;color:#B42318;font-weight:600",
    "HIGH":     "background-color:#FFEAD5;color:#B93815;font-weight:600",
    "MEDIUM":   "background-color:#FEF0C7;color:#93370D;font-weight:600",
    "LOW":      "background-color:#F2F4F7;color:#475467;font-weight:600",
}
FLAG_REASON = {
    "RULE_BASED":  "Over 10% and $5K off budget",
    "STATISTICAL": "Unusual for this account",
    "TREND":       "Same direction 3 months running",
}
# Revenue above budget is good; costs above budget are bad.
FAVORABLE_SIGN = {"Revenue": 1, "COGS": -1, "OpEx": -1}


# ── Session state ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading financial data & RAG index...")
def load_all():
    # Regenerate mock data once per server process, guarded by this cache's own
    # locking -- NOT unconditionally at module scope. Streamlit reruns the whole
    # script on every widget interaction for every session; running the
    # subprocess at module top level let concurrent sessions race to overwrite
    # data/processed/erp_combined.csv mid-write, so a read here could land on a
    # truncated or empty file. That's what caused the intermittent blank
    # heatmap / flat trend-chart line in production -- confirmed by reproducing
    # torn reads locally under concurrent regeneration.
    subprocess.run([sys.executable, "src/generate_data.py"], check=True)

    # Load the generated data into the mock ERP database, then read the
    # analysis table back out through SQL (posted entries in locked periods only).
    con = connect(build_database())
    try:
        dq_results = run_data_quality_checks(con)
        df = load_variance_base(con)
    finally:
        con.close()
    report = build_variance_report(df)
    metrics = compute_summary_metrics(df)
    rag = FinancialRAG(docs_dir="rag_docs", persist_dir="data/chroma_db")
    return df, report, metrics, rag, dq_results


# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt_currency(v: float) -> str:
    sign = "-" if v < 0 else ""
    v = abs(v)
    if v >= 1_000_000:
        return f"{sign}${v/1_000_000:.2f}M"
    if v >= 1_000:
        return f"{sign}${v/1_000:.1f}K"
    return f"{sign}${v:.0f}"

def fmt_signed_currency(v: float) -> str:
    return ("+" if v >= 0 else "") + fmt_currency(v)

def month_label(period: str) -> str:
    return pd.Timestamp(f"{period}-01").strftime("%b %Y")

def favorable_amount(row) -> float:
    return row["variance_amount"] * FAVORABLE_SIGN.get(row["account_type"], -1)

def style_chart(fig, height):
    fig.update_layout(
        template="plotly_white",
        font=dict(color="#374151", size=12),
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig

def report_markdown(text: str) -> str:
    """Prepare report text for st.markdown.

    Streamlit renders text between two $ signs as a LaTeX formula, which
    mangles any report with dollar amounts in it; escape them. Also shrink
    the report's ## section headings so they don't dwarf the page title.
    """
    text = re.sub(r"(?<!\\)\$", r"\\$", text)
    return re.sub(r"^#{1,3} ", "#### ", text, flags=re.MULTILINE)

def show_ai_text(text: str):
    with st.container(border=True):
        st.markdown(report_markdown(text))


# ── Load data ─────────────────────────────────────────────────────────────────
try:
    df, report, metrics, rag, dq_results = load_all()
except FileNotFoundError:
    st.error("Data not found. Run `python src/generate_data.py` and `python src/seed_erp_db.py` first.")
    st.stop()

all_periods = sorted(df["period"].unique())
closed_periods = sorted(df.loc[df["actual_amount"].notna(), "period"].unique())
current_period = closed_periods[-1]
has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")

    labels = [month_label(p) for p in all_periods]
    start_label, end_label = st.select_slider(
        "Months", options=labels, value=(labels[0], labels[-1]))
    selected_periods = all_periods[labels.index(start_label):labels.index(end_label) + 1]

    dept_options = sorted(df["department_name"].dropna().unique())
    selected_depts = st.multiselect(
        "Departments", dept_options, placeholder="All departments") or dept_options

    min_severity = st.select_slider(
        "Show flags from severity", options=SEVERITY_ORDER, value="MEDIUM",
        help="LOW flags are mostly small statistical blips; they're hidden by default.")
    severity_filter = SEVERITY_ORDER[SEVERITY_ORDER.index(min_severity):]

    st.divider()
    with st.expander("AI features (optional)", expanded=not has_api_key):
        st.caption("The dashboard works without a key. AI reports and Q&A need an "
                   "[Anthropic API key](https://console.anthropic.com); it is used only "
                   "for this session.")
        api_key = st.text_input("Anthropic API key", type="password", placeholder="sk-ant-...")
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
            has_api_key = True
            st.success("Key set for this session")


# Apply filters
report_filtered = report[
    report["period"].isin(selected_periods) &
    report["department_name"].isin(selected_depts) &
    report["severity"].isin(severity_filter)
].copy()

df_filtered = df[
    df["period"].isin(selected_periods) &
    df["department_name"].isin(selected_depts)
].copy()
df_closed = df_filtered.dropna(subset=["actual_amount"])


# ── Header ────────────────────────────────────────────────────────────────────
st.title("AI Variance Analyst")
st.markdown(
    f"Compares **budget vs. actual** results for NovaTech, a fictional SaaS company "
    f"(synthetic data, FY2024, actuals through **{month_label(current_period)}**), flags the "
    f"differences worth a closer look, and uses AI to explain them."
)
st.caption("Colors: **:green[green = better than budget]**, **:red[red = worse than budget]**. "
           "For revenue, above budget is better; for costs, below budget is better.")


# ── KPI Cards (year to date, all departments) ─────────────────────────────────
m = metrics
op_income_budget = m["revenue_budget"] - m["expense_budget"]
op_income_actual = m["revenue_actual"] - m["expense_actual"]
op_income_delta = (op_income_actual - op_income_budget) / op_income_budget
n_critical = int((report["severity"] == "CRITICAL").sum())
n_high = int((report["severity"] == "HIGH").sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Revenue (YTD)", fmt_currency(m["revenue_actual"]),
          f"{m['revenue_variance_pct']:+.1%} vs budget", border=True)
c2.metric("Costs (YTD)", fmt_currency(m["expense_actual"]),
          f"{m['expense_variance_pct']:+.1%} vs budget", delta_color="inverse", border=True,
          help="Cost of revenue plus operating expenses")
c3.metric("Operating income (YTD)", fmt_currency(op_income_actual),
          f"{op_income_delta:+.1%} vs budget", border=True,
          help=f"Revenue minus costs. Budget: {fmt_currency(op_income_budget)}")
c4.metric("Variances needing review", n_critical + n_high,
          f"{n_critical} critical · {n_high} high", delta_color="off", delta_arrow="off", border=True,
          help="Flags rated CRITICAL (30%+ off budget) or HIGH (20–30%)")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "AI report", "Explain a variance", "Trends"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Flagged variances")
    st.caption(f"{len(report_filtered)} flags match the filters, worst first. Severity is the size "
               "of the miss: CRITICAL 30%+, HIGH 20–30%, MEDIUM 10–20%.")

    if report_filtered.empty:
        st.info("No flagged variances match the current filters.")
    else:
        fav = report_filtered.apply(favorable_amount, axis=1)
        table = pd.DataFrame({
            "Severity":    report_filtered["severity"],
            "Account":     report_filtered["account_name"],
            "Department":  report_filtered["department_name"],
            "Month":       report_filtered["period"].map(month_label),
            "Budget":      report_filtered["budget_amount"].map(fmt_currency),
            "Actual":      report_filtered["actual_amount"].map(fmt_currency),
            "Difference":  report_filtered["variance_amount"].map(fmt_signed_currency),
            "Diff %":      report_filtered["variance_pct"].map(lambda x: f"{x:+.1%}"),
            "Result":      np.where(fav >= 0, "Better than budget", "Worse than budget"),
            "Why flagged": report_filtered["flag_type"].map(FLAG_REASON),
        })
        styled = (table.style
                  .map(lambda v: SEVERITY_STYLE.get(v, ""), subset=["Severity"])
                  .map(lambda v: f"color:{GOOD if v.startswith('Better') else BAD};font-weight:600",
                       subset=["Result"]))
        st.dataframe(styled, use_container_width=True, hide_index=True,
                     height=min(38 + 35 * len(table), 460))

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Where results differ from budget")
        st.caption("Percent better (+) or worse (−) than budget, by month")
        heat = df_closed.assign(fav=df_closed.apply(favorable_amount, axis=1))
        heat = heat.groupby(["account_type", "period"]).agg(
            fav=("fav", "sum"), budget=("budget_amount", "sum")).reset_index()
        heat["fav_pct"] = heat["fav"] / heat["budget"] * 100
        pivot = heat.pivot(index="account_type", columns="period", values="fav_pct")
        pivot = pivot.reindex([t for t in ["OpEx", "COGS", "Revenue"] if t in pivot.index])
        row_names = {"Revenue": "Revenue", "COGS": "Cost of revenue", "OpEx": "Operating expenses"}

        if pivot.empty:
            st.info("No closed months in the selected range.")
        else:
            limit = float(np.nanmax(np.abs(pivot.values))) or 1.0
            fig_heat = go.Figure(go.Heatmap(
                z=pivot.values.tolist(),
                x=[pd.Timestamp(f"{p}-01").strftime("%b") for p in pivot.columns],
                y=[row_names[t] for t in pivot.index],
                zmin=-limit, zmax=limit,
                colorscale=[[0, BAD], [0.5, "#FFFFFF"], [1, GOOD]],
                text=[[f"{v:+.0f}%" if pd.notna(v) else "" for v in r] for r in pivot.values],
                texttemplate="%{text}",
                textfont={"size": 11},
                hovertemplate="%{y}, %{x}: %{text}<extra></extra>",
                showscale=False,
                xgap=2, ygap=2,
            ))
            st.plotly_chart(style_chart(fig_heat, 260), use_container_width=True)

    with col_right:
        st.subheader("Budget to actual: operating income")
        st.caption("How each part of the P&L moved operating income away from budget")
        sums = df_closed.groupby("account_type").agg(
            budget=("budget_amount", "sum"), actual=("actual_amount", "sum"))
        get = lambda t, c: float(sums.loc[t, c]) if t in sums.index else 0.0
        oi_budget = get("Revenue", "budget") - get("COGS", "budget") - get("OpEx", "budget")
        steps = [
            ("Revenue", get("Revenue", "actual") - get("Revenue", "budget")),
            ("Cost of revenue", -(get("COGS", "actual") - get("COGS", "budget"))),
            ("Operating expenses", -(get("OpEx", "actual") - get("OpEx", "budget"))),
        ]
        oi_actual = oi_budget + sum(v for _, v in steps)

        fig_wf = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "total"],
            x=["Budget"] + [s for s, _ in steps] + ["Actual"],
            y=[oi_budget] + [v for _, v in steps] + [oi_actual],
            text=[fmt_currency(oi_budget)] + [fmt_signed_currency(v) for _, v in steps]
                 + [fmt_currency(oi_actual)],
            textposition="outside",
            increasing=dict(marker_color=GOOD),
            decreasing=dict(marker_color=BAD),
            totals=dict(marker_color=NEUTRAL),
            connector=dict(line=dict(color=GRAY_LINE)),
        ))
        # Zoom in on the range that matters; from $0 the differences would look flat.
        levels = [oi_budget, oi_actual]
        running = oi_budget
        for _, v in steps:
            running += v
            levels.append(running)
        span = max(levels) - min(levels) or abs(oi_budget) * 0.1 or 1.0
        fig_wf.update_layout(showlegend=False, yaxis=dict(
            showticklabels=False, showgrid=False,
            range=[min(levels) - span * 0.6, max(levels) + span * 0.5]))
        st.plotly_chart(style_chart(fig_wf, 260), use_container_width=True)

    # ── Data quality checks (run in SQL before the variance analysis)
    n_failed = sum(not r.passed for r in dq_results)
    dq_label = ("Data quality: all checks passed" if n_failed == 0
                else f"Data quality: {n_failed} of {len(dq_results)} checks found issues")
    with st.expander(dq_label):
        st.caption("Run in SQL against the ERP database (DuckDB) before the variance analysis. "
                   "Queries live in sql/checks/.")
        for r in dq_results:
            icon = "✅" if r.passed else ("❌" if r.check.severity == "ERROR" else "⚠️")
            status = "passed" if r.passed else f"{r.issue_count} issue(s)"
            st.markdown(f"{icon} **{r.check.name}**: {r.check.description}: {status}")
            if not r.passed:
                st.dataframe(r.rows, use_container_width=True, hide_index=True)
        if st.toggle("Show the extract SQL"):
            st.code(read_sql("extract_variance_base.sql"), language="sql")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — AI EXECUTIVE REPORT
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("AI executive report")
    st.caption("Claude writes a CFO-style summary of the top flagged variances, using context "
               "retrieved from company documents (budget assumptions, seasonality, past variances).")

    if not has_api_key:
        st.info("Add an Anthropic API key under **AI features** in the sidebar to generate a report.")
        sample_path = os.path.join("outputs", "sample_ai_report.md")
        if os.path.exists(sample_path):
            with st.expander("Preview the report format (pre-written example, not live output)"):
                with open(sample_path) as f:
                    st.markdown(report_markdown(f.read()))
    else:
        col_btn, col_info = st.columns([2, 3])
        with col_btn:
            n_variances = st.slider("Top variances to analyze", 5, 15, 8)
            generate_btn = st.button("Generate report", type="primary", use_container_width=True)
        with col_info:
            st.info(f"Analyzes the top {n_variances} flagged variances. Takes about 10–15 seconds.")

        if generate_btn:
            with st.spinner("Retrieving context from company documents..."):
                top_variances = get_top_variances(report_filtered, n=n_variances)
                full_context = ""
                for var in top_variances[:5]:
                    ctx = rag.retrieve_for_variance(
                        var["account"], var["department"],
                        var["period"], float(var["variance_pct"].replace("%", "").replace("+", "")) / 100
                    )
                    full_context += f"\n\n### Context for {var['account']} ({var['department']}, {var['period']}):\n{ctx[:400]}"

            with st.spinner("Claude is analyzing the variances..."):
                try:
                    analyst = VarianceAnalyst()
                    st.session_state["exec_report"] = analyst.generate_full_report(
                        top_variances, metrics, full_context)
                except Exception as e:
                    st.error(f"Error generating report: {e}")
                    st.stop()

        if "exec_report" in st.session_state:
            show_ai_text(st.session_state["exec_report"])
            st.download_button(
                "Download report",
                st.session_state["exec_report"],
                file_name="variance_report_fy2024.md",
                mime="text/markdown",
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EXPLAIN A VARIANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Explain a single variance")
    st.caption("Pick a flagged variance to see its numbers and get an AI explanation of the likely driver.")

    if report_filtered.empty:
        st.info("No flagged variances match the current filters.")
    else:
        col_sel1, col_sel2, col_sel3 = st.columns(3)
        with col_sel1:
            selected_account = st.selectbox("Account", sorted(report_filtered["account_name"].unique()))
        acct_rows = report_filtered[report_filtered["account_name"] == selected_account]
        with col_sel2:
            selected_dept = st.selectbox("Department", sorted(acct_rows["department_name"].unique()))
        combo = acct_rows[acct_rows["department_name"] == selected_dept]
        with col_sel3:
            selected_period = st.selectbox("Month", sorted(combo["period"].unique()),
                                           format_func=month_label)

        row = combo[combo["period"] == selected_period].iloc[0]
        var_pct = row["variance_pct"]
        is_cost = row["account_type"] != "Revenue"

        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        col_d1.metric("Budget", fmt_currency(row["budget_amount"]), border=True)
        col_d2.metric("Actual", fmt_currency(row["actual_amount"]), border=True)
        col_d3.metric("Difference", fmt_signed_currency(row["variance_amount"]),
                      f"{var_pct:+.1%}", delta_color="inverse" if is_cost else "normal", border=True)
        col_d4.metric("Severity", row["severity"], FLAG_REASON.get(row["flag_type"], ""),
                      delta_color="off", delta_arrow="off", border=True)

        if not has_api_key:
            st.info("Add an Anthropic API key under **AI features** in the sidebar to get an AI explanation.")
        elif st.button("Explain this variance", type="primary"):
            with st.spinner("Retrieving context..."):
                context = rag.retrieve_for_variance(
                    selected_account, selected_dept, selected_period, float(var_pct))
            with st.spinner("Claude is analyzing..."):
                try:
                    analyst = VarianceAnalyst()
                    explanation = analyst.explain_single_variance(
                        selected_account, selected_dept, selected_period,
                        row["budget_amount"], row["actual_amount"], context
                    )
                    show_ai_text(explanation)
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()
    st.subheader("Ask a question")
    question = st.text_input(
        "Ask about the financials",
        placeholder="e.g. Why did marketing spend spike in Q3? What's driving the revenue miss?",
    )
    if question and not has_api_key:
        st.info("Add an Anthropic API key under **AI features** in the sidebar to ask questions.")
    elif question and st.button("Ask", type="primary"):
        with st.spinner("Thinking..."):
            context = rag.retrieve(question)
            variance_summary = json.dumps(get_top_variances(report_filtered, n=8), indent=2)
            try:
                analyst = VarianceAnalyst()
                show_ai_text(analyst.answer_question(question, variance_summary, context))
            except Exception as e:
                st.error(f"Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TRENDS
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Budget vs. actual by month")

    acct_names = sorted(df["account_name"].unique())
    chart_account = st.selectbox(
        "Account", acct_names, index=acct_names.index("Marketing & Advertising"), key="chart_acct")
    acct_type = df.loc[df["account_name"] == chart_account, "account_type"].iloc[0]
    sign = FAVORABLE_SIGN.get(acct_type, -1)

    acct_data = df[df["account_name"] == chart_account].groupby("period").agg(
        budget=("budget_amount", "sum"),
        actual=("actual_amount", lambda x: x.sum(min_count=1)),  # NaN for open months
        forecast=("forecast_amount", "sum"),
    ).reset_index()
    acct_data["month"] = acct_data["period"].map(lambda p: pd.Timestamp(f"{p}-01").strftime("%b"))
    # Forecast only for open months; for closed months the actual is the truth
    forecast_open = acct_data["forecast"].where(acct_data["actual"].isna())
    cutoff_month = pd.Timestamp(f"{current_period}-01").strftime("%b")

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=acct_data["month"], y=acct_data["budget"], name="Budget",
        line=dict(color=NEUTRAL, dash="dash", width=2), mode="lines"))
    fig_trend.add_trace(go.Scatter(
        x=acct_data["month"], y=acct_data["actual"], name="Actual",
        line=dict(color=BLUE, width=3), mode="lines+markers", connectgaps=False))
    fig_trend.add_trace(go.Scatter(
        x=acct_data["month"], y=forecast_open, name="Forecast",
        line=dict(color="#93C5FD", dash="dot", width=3), mode="lines+markers", connectgaps=False))
    fig_trend.add_vline(x=cutoff_month, line=dict(color=GRAY_LINE, width=2))
    fig_trend.add_annotation(x=cutoff_month, y=1.06, yref="paper", showarrow=False,
                             text="actuals ← | → forecast", font=dict(color=NEUTRAL, size=11))
    fig_trend.update_layout(legend=dict(orientation="h", y=-0.15), yaxis=dict(tickprefix="$"))
    st.plotly_chart(style_chart(fig_trend, 360), use_container_width=True)

    acct_data["fav_pct"] = (acct_data["actual"] - acct_data["budget"]) / acct_data["budget"] * 100 * sign
    closed = acct_data.dropna(subset=["fav_pct"])
    fig_bars = go.Figure(go.Bar(
        x=closed["month"], y=closed["fav_pct"],
        marker_color=[GOOD if v >= 0 else BAD for v in closed["fav_pct"]],
        text=[f"{v:+.1f}%" for v in closed["fav_pct"]], textposition="outside",
    ))
    fig_bars.update_layout(
        title=dict(text="Better (+) or worse (−) than budget, % by month", font=dict(size=14)),
        showlegend=False, yaxis=dict(ticksuffix="%", zeroline=True, zerolinecolor="#9CA3AF"))
    st.plotly_chart(style_chart(fig_bars, 300), use_container_width=True)

    st.subheader("Year to date by account")
    ytd = df.dropna(subset=["actual_amount"]).groupby(["account_name", "account_type"]).agg(
        budget=("budget_amount", "sum"), actual=("actual_amount", "sum")).reset_index()
    ytd["fav_pct"] = ((ytd["actual"] - ytd["budget"]) / ytd["budget"] * 100
                      * ytd["account_type"].map(FAVORABLE_SIGN))
    ytd = ytd.sort_values("fav_pct")
    fig_all = go.Figure(go.Bar(
        x=ytd["fav_pct"], y=ytd["account_name"], orientation="h",
        marker_color=[GOOD if v >= 0 else BAD for v in ytd["fav_pct"]],
        text=[f"{v:+.1f}%" for v in ytd["fav_pct"]], textposition="outside",
    ))
    fig_all.update_layout(
        title=dict(text="Better (+) or worse (−) than budget, % year to date", font=dict(size=14)),
        showlegend=False, xaxis=dict(ticksuffix="%", zeroline=True, zerolinecolor="#9CA3AF"))
    st.plotly_chart(style_chart(fig_all, 480), use_container_width=True)
