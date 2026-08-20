"""
AI Variance Analyst — FP&A Copilot
Streamlit Dashboard with LLM-powered variance analysis
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import subprocess

from variance_engine import (
    load_data, build_variance_report, compute_summary_metrics, get_top_variances
)
from rag_pipeline import FinancialRAG
from llm_analyst import VarianceAnalyst

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Variance Analyst | FP&A Copilot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

  .main { background: #0a0f1e; }
  .block-container { padding: 1.5rem 2rem; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #0d1424 !important;
    border-right: 1px solid #1e2d4a;
  }

  /* Cards */
  .metric-card {
    background: linear-gradient(135deg, #0f1e3a 0%, #0a1628 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
  }
  .metric-value { font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem; font-weight: 600; }
  .metric-label { font-size: 0.75rem; color: #6b8aad; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px; }
  .metric-delta { font-size: 0.85rem; margin-top: 6px; }

  .positive { color: #22c55e; }
  .negative { color: #ef4444; }
  .neutral  { color: #94a3b8; }

  /* Severity badges */
  .badge-critical { background: #7f1d1d; color: #fca5a5; padding: 2px 10px; border-radius: 20px; font-size: 0.7rem; font-weight: 600; }
  .badge-high     { background: #7c2d12; color: #fdba74; padding: 2px 10px; border-radius: 20px; font-size: 0.7rem; font-weight: 600; }
  .badge-medium   { background: #713f12; color: #fde047; padding: 2px 10px; border-radius: 20px; font-size: 0.7rem; font-weight: 600; }
  .badge-low      { background: #14532d; color: #86efac; padding: 2px 10px; border-radius: 20px; font-size: 0.7rem; font-weight: 600; }

  /* Report output */
  .report-box {
    background: #0d1424;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.5rem;
    font-family: 'IBM Plex Sans', sans-serif;
    line-height: 1.7;
    color: #cbd5e1;
  }
  .report-box h2 { color: #60a5fa; border-bottom: 1px solid #1e3a5f; padding-bottom: 0.5rem; }

  /* Header */
  .app-header {
    background: linear-gradient(90deg, #0f2d5a 0%, #1a1a2e 100%);
    border-bottom: 1px solid #1e3a5f;
    padding: 1rem 0;
    margin-bottom: 1.5rem;
    border-radius: 12px;
  }
  .app-title { font-size: 1.5rem; font-weight: 700; color: #e2e8f0; font-family: 'IBM Plex Mono', monospace; }
  .app-subtitle { font-size: 0.85rem; color: #64748b; }

  /* Sticker */
  .ai-badge {
    display: inline-block;
    background: linear-gradient(90deg, #1d4ed8, #7c3aed);
    color: white;
    font-size: 0.65rem;
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: 600;
    letter-spacing: 0.05em;
  }

  div[data-testid="stHorizontalBlock"] { gap: 1rem; }
</style>
""", unsafe_allow_html=True)


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

    df = load_data("data/processed/erp_combined.csv")
    report = build_variance_report(df)
    metrics = compute_summary_metrics(df)
    rag = FinancialRAG(docs_dir="rag_docs", persist_dir="data/chroma_db")
    return df, report, metrics, rag


# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt_currency(v: float) -> str:
    if abs(v) >= 1_000_000:
        return f"${v/1_000_000:.2f}M"
    if abs(v) >= 1_000:
        return f"${v/1_000:.1f}K"
    return f"${v:.0f}"

def fmt_pct(v: float) -> str:
    return f"{v:+.1%}"

SEVERITY_COLOR = {
    "CRITICAL": "#ef4444",
    "HIGH":     "#f97316",
    "MEDIUM":   "#eab308",
    "LOW":      "#22c55e",
}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="color:#60a5fa;font-family:IBM Plex Mono,monospace;font-size:1rem;font-weight:700;margin-bottom:1rem;">⬡ AI VARIANCE ANALYST</div>', unsafe_allow_html=True)

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Required for AI-generated reports. Get yours at console.anthropic.com",
    )
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key
        st.success("✓ API Key set")

    st.divider()
    st.markdown("**Filters**")
    period_options = [f"2024-{str(m).zfill(2)}" for m in range(1, 13)]
    selected_periods = st.multiselect("Periods", period_options, default=period_options)

    dept_options = ["Engineering", "Sales", "Marketing", "G&A", "Customer Success", "Product"]
    selected_depts = st.multiselect("Departments", dept_options, default=dept_options)

    severity_filter = st.multiselect(
        "Severity", ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        default=["CRITICAL", "HIGH", "MEDIUM"]
    )

    st.divider()
    st.markdown('<span class="ai-badge">RAG-GROUNDED · NO HALLUCINATION</span>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.72rem;color:#475569;margin-top:8px;">LLM answers are grounded in company context docs. All figures sourced from variance data.</div>', unsafe_allow_html=True)


# ── Load data ─────────────────────────────────────────────────────────────────
try:
    df, report, metrics, rag = load_all()
    DATA_OK = True
except FileNotFoundError:
    DATA_OK = False
    st.error("⚠️ Data not found. Run `python src/generate_data.py` first to generate mock ERP data.")
    st.stop()

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

# Closed months only (actuals exist) — used for dashboard charts
df_closed = df_filtered.dropna(subset=["actual_amount"])
current_period = df["period"][df["actual_amount"].notna()].max()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header" style="padding:1.2rem 1.5rem;">
  <span class="app-title">📊 AI Variance Analyst</span>
  <span style="color:#475569;margin:0 1rem;">|</span>
  <span class="app-subtitle">FP&A Copilot · NovaTech Solutions Inc. · FY2024 · YTD through {current_period}</span>
  <span style="float:right;"><span class="ai-badge">POWERED BY CLAUDE</span></span>
</div>
""", unsafe_allow_html=True)


# ── KPI Cards ─────────────────────────────────────────────────────────────────
m = metrics
rev_delta_pct = m["revenue_variance_pct"]
exp_delta_pct = m["expense_variance_pct"]

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    color = "positive" if rev_delta_pct >= 0 else "negative"
    st.markdown(f"""<div class="metric-card">
      <div class="metric-value {color}">{fmt_currency(m['revenue_actual'])}</div>
      <div class="metric-label">Total Revenue (Actual)</div>
      <div class="metric-delta {color}">{fmt_pct(rev_delta_pct)} vs Budget</div>
    </div>""", unsafe_allow_html=True)

with c2:
    color = "negative" if exp_delta_pct > 0 else "positive"
    st.markdown(f"""<div class="metric-card">
      <div class="metric-value {color}">{fmt_currency(m['expense_actual'])}</div>
      <div class="metric-label">Total Expenses (Actual)</div>
      <div class="metric-delta {color}">{fmt_pct(exp_delta_pct)} vs Budget</div>
    </div>""", unsafe_allow_html=True)

with c3:
    net_var = m["total_variance"]
    color = "positive" if net_var >= 0 else "negative"
    st.markdown(f"""<div class="metric-card">
      <div class="metric-value {color}">{fmt_currency(net_var)}</div>
      <div class="metric-label">Net Variance (Total)</div>
      <div class="metric-delta {color}">{fmt_pct(m['total_variance_pct'])}</div>
    </div>""", unsafe_allow_html=True)

with c4:
    n_critical = len(report_filtered[report_filtered["severity"] == "CRITICAL"])
    n_high     = len(report_filtered[report_filtered["severity"] == "HIGH"])
    st.markdown(f"""<div class="metric-card">
      <div class="metric-value negative">{n_critical + n_high}</div>
      <div class="metric-label">Critical / High Flags</div>
      <div class="metric-delta neutral">{n_critical} CRITICAL · {n_high} HIGH</div>
    </div>""", unsafe_allow_html=True)

with c5:
    fc_delta = m["forecast_vs_budget"]
    color = "positive" if fc_delta >= 0 else "negative"
    st.markdown(f"""<div class="metric-card">
      <div class="metric-value {color}">{fmt_pct(fc_delta)}</div>
      <div class="metric-label">Forecast vs Budget</div>
      <div class="metric-delta neutral">Full-Year Outlook</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Variance Dashboard",
    "🤖 AI Executive Report",
    "🔍 Drill-Down Analyst",
    "📈 Trend Charts",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — VARIANCE DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("🚩 Flagged Variances")

        # Variance table
        display_cols = ["account_name", "department_name", "period",
                        "budget_amount", "actual_amount", "variance_amount",
                        "variance_pct", "severity", "direction"]
        disp = report_filtered[display_cols].copy()
        disp.columns = ["Account", "Dept", "Period", "Budget", "Actual", "Variance $", "Variance %", "Severity", "Direction"]
        disp["Budget"]    = disp["Budget"].apply(fmt_currency)
        disp["Actual"]    = disp["Actual"].apply(fmt_currency)
        disp["Variance $"]= disp["Variance $"].apply(lambda x: f"${x:+,.0f}")
        disp["Variance %"]= disp["Variance %"].apply(lambda x: f"{x:+.1%}")

        def color_severity(val):
            colors = {"CRITICAL": "background-color:#7f1d1d;color:#fca5a5",
                      "HIGH":     "background-color:#7c2d12;color:#fdba74",
                      "MEDIUM":   "background-color:#713f12;color:#fde047",
                      "LOW":      "background-color:#14532d;color:#86efac"}
            return colors.get(val, "")

        styled = disp.style.applymap(color_severity, subset=["Severity"])
        st.dataframe(styled, use_container_width=True, height=420)

    with col_right:
        st.subheader("📊 Variance Heatmap")

        # Pivot for heatmap: account_type vs period (closed months only)
        heat_data = df_closed.groupby(["account_type", "period"]).agg(
            variance_pct=("variance_amount", lambda x: x.sum() / df_closed.loc[x.index, "budget_amount"].sum())
        ).reset_index()

        pivot = heat_data.pivot(index="account_type", columns="period", values="variance_pct")

        fig_heat = go.Figure(data=go.Heatmap(
            z=pivot.values * 100,
            x=[p[-2:] + "/" + p[:4] for p in pivot.columns],
            y=pivot.index,
            colorscale=[[0, "#1a4532"], [0.5, "#0d1424"], [1, "#7f1d1d"]],
            zmid=0,
            text=np.round(pivot.values * 100, 1),
            texttemplate="%{text}%",
            textfont={"size": 10, "color": "white"},
            colorbar=dict(title="% Var", tickfont=dict(color="#94a3b8")),
        ))
        fig_heat.update_layout(
            paper_bgcolor="#0d1424", plot_bgcolor="#0d1424",
            font=dict(color="#94a3b8"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=300,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # Severity donut
        st.subheader("🎯 Flag Severity Mix")
        sev_counts = report_filtered["severity"].value_counts().reset_index()
        sev_counts.columns = ["Severity", "Count"]
        colors_map = {"CRITICAL": "#ef4444", "HIGH": "#f97316", "MEDIUM": "#eab308", "LOW": "#22c55e"}
        fig_donut = go.Figure(data=go.Pie(
            labels=sev_counts["Severity"],
            values=sev_counts["Count"],
            hole=0.65,
            marker_colors=[colors_map.get(s, "#888") for s in sev_counts["Severity"]],
        ))
        fig_donut.update_layout(
            paper_bgcolor="#0d1424",
            font=dict(color="#94a3b8"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=220,
            showlegend=True,
            legend=dict(font=dict(color="#94a3b8")),
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # ── Waterfall chart
    st.subheader("💧 Budget-to-Actual Waterfall (by Account Type)")
    wf_data = df_closed.groupby("account_type").agg(
        budget=("budget_amount","sum"),
        actual=("actual_amount","sum")
    ).reset_index()
    wf_data["variance"] = wf_data["actual"] - wf_data["budget"]
    wf_data = wf_data.sort_values("variance")

    fig_wf = go.Figure(go.Waterfall(
        name="Variance",
        orientation="v",
        measure=["relative"] * len(wf_data),
        x=wf_data["account_type"],
        y=wf_data["variance"],
        text=[f"${v:+,.0f}" for v in wf_data["variance"]],
        textposition="outside",
        decreasing=dict(marker_color="#22c55e"),
        increasing=dict(marker_color="#ef4444"),
        totals=dict(marker_color="#60a5fa"),
    ))
    fig_wf.update_layout(
        paper_bgcolor="#0d1424", plot_bgcolor="#0d1424",
        font=dict(color="#94a3b8"),
        yaxis=dict(gridcolor="#1e2d4a"),
        height=300, margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig_wf, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — AI EXECUTIVE REPORT
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🤖 AI-Generated Executive Variance Report")
    st.caption("Grounded in company context documents via RAG · No hallucination · Powered by Claude")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning("⚠️ Enter your Anthropic API key in the sidebar to generate AI reports.")
    else:
        col_btn, col_info = st.columns([2, 3])
        with col_btn:
            n_variances = st.slider("Top variances to analyze", 5, 15, 8)
            generate_btn = st.button("🚀 Generate Executive Report", type="primary", use_container_width=True)

        with col_info:
            st.info(f"Will analyze top {n_variances} flagged variances with RAG-retrieved context. Takes ~10-15 seconds.")

        if generate_btn or "exec_report" in st.session_state:
            if generate_btn:
                with st.spinner("🔍 Retrieving relevant context from RAG index..."):
                    top_variances = get_top_variances(report_filtered, n=n_variances)
                    full_context = ""
                    for var in top_variances[:5]:
                        ctx = rag.retrieve_for_variance(
                            var["account"], var["department"],
                            var["period"], float(var["variance_pct"].replace("%","").replace("+","")) / 100
                        )
                        full_context += f"\n\n### Context for {var['account']} ({var['department']}, {var['period']}):\n{ctx[:400]}"

                with st.spinner("🤖 Claude is analyzing variances..."):
                    try:
                        analyst = VarianceAnalyst()
                        report_text = analyst.generate_full_report(top_variances, metrics, full_context)
                        st.session_state["exec_report"] = report_text
                    except Exception as e:
                        st.error(f"Error generating report: {e}")
                        st.stop()

            if "exec_report" in st.session_state:
                st.markdown(f'<div class="report-box">{st.session_state["exec_report"].replace(chr(10), "<br>")}</div>',
                           unsafe_allow_html=True)

                st.download_button(
                    "📥 Download Report",
                    st.session_state["exec_report"],
                    file_name="variance_report_fy2024.txt",
                    mime="text/plain",
                )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DRILL-DOWN ANALYST
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🔍 Drill-Down Variance Explainer")
    st.caption("Select any flagged variance for a detailed AI explanation")

    col_sel1, col_sel2 = st.columns(2)

    with col_sel1:
        unique_accounts = sorted(report_filtered["account_name"].unique())
        selected_account = st.selectbox("Account", unique_accounts)

    with col_sel2:
        acct_rows = report_filtered[report_filtered["account_name"] == selected_account]
        depts_for_acct = sorted(acct_rows["department_name"].unique())
        selected_dept = st.selectbox("Department", depts_for_acct)

    periods_for_combo = sorted(
        report_filtered[
            (report_filtered["account_name"] == selected_account) &
            (report_filtered["department_name"] == selected_dept)
        ]["period"].unique()
    )

    if periods_for_combo:
        selected_period = st.selectbox("Period", periods_for_combo)

        # Show variance detail
        row = report_filtered[
            (report_filtered["account_name"] == selected_account) &
            (report_filtered["department_name"] == selected_dept) &
            (report_filtered["period"] == selected_period)
        ].iloc[0]

        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        with col_d1:
            st.metric("Budget", fmt_currency(row["budget_amount"]))
        with col_d2:
            st.metric("Actual", fmt_currency(row["actual_amount"]))
        with col_d3:
            var_pct = row["variance_pct"]
            st.metric("Variance %", f"{var_pct:+.1%}", delta=f"${row['variance_amount']:+,.0f}")
        with col_d4:
            sev = row["severity"]
            color = SEVERITY_COLOR.get(sev, "#888")
            st.markdown(f'<div style="padding-top:1rem;"><span style="background:{color}22;color:{color};padding:4px 14px;border-radius:20px;font-weight:700;font-size:0.9rem;">{sev}</span></div>', unsafe_allow_html=True)

        if not os.environ.get("ANTHROPIC_API_KEY"):
            st.warning("⚠️ Enter your Anthropic API key in the sidebar to get AI explanations.")
        else:
            if st.button("🔍 Explain This Variance", type="primary"):
                with st.spinner("Retrieving context..."):
                    context = rag.retrieve_for_variance(
                        selected_account, selected_dept,
                        selected_period, float(var_pct)
                    )

                with st.spinner("Claude is analyzing..."):
                    try:
                        analyst = VarianceAnalyst()
                        explanation = analyst.explain_single_variance(
                            selected_account, selected_dept, selected_period,
                            row["budget_amount"], row["actual_amount"], context
                        )
                        st.markdown(f'<div class="report-box">{explanation.replace(chr(10), "<br>")}</div>',
                                   unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error: {e}")

    # Q&A section
    st.divider()
    st.subheader("💬 Ask the FP&A Copilot")
    question = st.text_input(
        "Ask a question about the financials",
        placeholder="e.g. Why did marketing spend spike in Q3? What's driving the revenue miss?",
    )
    if question and os.environ.get("ANTHROPIC_API_KEY"):
        if st.button("Ask", type="primary"):
            with st.spinner("Thinking..."):
                context = rag.retrieve(question)
                variance_summary = json.dumps(get_top_variances(report_filtered, n=8), indent=2)
                try:
                    analyst = VarianceAnalyst()
                    answer = analyst.answer_question(question, variance_summary, context)
                    st.markdown(f'<div class="report-box">{answer.replace(chr(10), "<br>")}</div>',
                               unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TREND CHARTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📈 Budget vs Actual Trends")

    # Select account to chart
    acct_names = sorted(df["account_name"].unique())
    chart_account = st.selectbox(
        "Select Account",
        acct_names,
        index=acct_names.index("Marketing & Advertising"),
        key="chart_acct"
    )

    acct_data = df[df["account_name"] == chart_account].groupby("period").agg(
        budget=("budget_amount","sum"),
        actual=("actual_amount", lambda x: x.sum(min_count=1)),  # NaN for open months
        forecast=("forecast_amount","sum"),
    ).reset_index()

    # Show forecast only for open months — actual is ground truth for closed months
    forecast_open = acct_data["forecast"].where(acct_data["actual"].isna())

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=acct_data["period"], y=acct_data["budget"],
        name="Budget", line=dict(color="#60a5fa", dash="dash", width=2),
        mode="lines+markers",
    ))
    fig_trend.add_trace(go.Scatter(
        x=acct_data["period"], y=acct_data["actual"],
        name="Actual", line=dict(color="#f97316", width=3),
        mode="lines+markers", marker=dict(size=8),
        connectgaps=False,
    ))
    fig_trend.add_trace(go.Scatter(
        x=acct_data["period"], y=forecast_open,
        name="Forecast", line=dict(color="#a78bfa", dash="dot", width=2),
        mode="lines+markers", marker=dict(size=8),
        connectgaps=False,
    ))
    fig_trend.add_shape(
        type="line",
        x0=current_period, x1=current_period,
        y0=0, y1=1, yref="paper",
        line=dict(color="#64748b", width=1.5, dash="dash"),
    )
    fig_trend.add_annotation(
        x=current_period, y=1.02, yref="paper",
        text="← Actuals | Forecast →",
        showarrow=False,
        font=dict(color="#64748b", size=11),
        xanchor="center",
    )
    fig_trend.update_layout(
        paper_bgcolor="#0d1424", plot_bgcolor="#0d1424",
        font=dict(color="#94a3b8"),
        legend=dict(font=dict(color="#94a3b8"), bgcolor="#0d1424"),
        yaxis=dict(gridcolor="#1e2d4a"),
        xaxis=dict(gridcolor="#1e2d4a"),
        title=dict(text=f"Budget vs Actual vs Forecast — {chart_account}", font=dict(color="#e2e8f0")),
        height=380, margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # Variance bars
    acct_data["variance_pct"] = (acct_data["actual"] - acct_data["budget"]) / acct_data["budget"] * 100
    fig_bars = go.Figure(go.Bar(
        x=acct_data["period"],
        y=acct_data["variance_pct"],
        marker_color=["#ef4444" if v > 0 else "#22c55e" for v in acct_data["variance_pct"]],
        text=[f"{v:+.1f}%" for v in acct_data["variance_pct"]],
        textposition="outside",
        textfont=dict(color="#94a3b8"),
    ))
    fig_bars.update_layout(
        paper_bgcolor="#0d1424", plot_bgcolor="#0d1424",
        font=dict(color="#94a3b8"),
        yaxis=dict(gridcolor="#1e2d4a", title="Variance %"),
        xaxis=dict(gridcolor="#1e2d4a"),
        title=dict(text="Monthly Variance % vs Budget", font=dict(color="#e2e8f0")),
        height=300, margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig_bars, use_container_width=True)

    # Multi-account variance comparison
    st.subheader("📊 All Accounts — YTD Variance Summary")
    ytd_summary = df.dropna(subset=["actual_amount"]).groupby(["account_name", "account_type"]).agg(
        budget=("budget_amount","sum"),
        actual=("actual_amount","sum"),
    ).reset_index()
    ytd_summary["variance_pct"] = (ytd_summary["actual"] - ytd_summary["budget"]) / ytd_summary["budget"] * 100
    ytd_summary = ytd_summary.sort_values("variance_pct")

    fig_all = go.Figure(go.Bar(
        x=ytd_summary["variance_pct"],
        y=ytd_summary["account_name"],
        orientation="h",
        marker_color=["#ef4444" if v > 0 else "#22c55e" for v in ytd_summary["variance_pct"]],
        text=[f"{v:+.1f}%" for v in ytd_summary["variance_pct"]],
        textposition="outside",
    ))
    fig_all.update_layout(
        paper_bgcolor="#0d1424", plot_bgcolor="#0d1424",
        font=dict(color="#94a3b8"),
        xaxis=dict(gridcolor="#1e2d4a", title="Variance %"),
        yaxis=dict(gridcolor="#1e2d4a"),
        title=dict(text="Full-Year Variance % by Account", font=dict(color="#e2e8f0")),
        height=500, margin=dict(l=20, r=200, t=50, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig_all, use_container_width=True)
