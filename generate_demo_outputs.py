"""
Generate pre-computed demo outputs for GitHub showcase.
Run this after generating data to create sample reports and charts.
"""

import sys, os
sys.path.insert(0, "src")

import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from variance_engine import load_data, build_variance_report, compute_summary_metrics, get_top_variances

os.makedirs("outputs", exist_ok=True)

print("Loading data...")
df = load_data()
report = build_variance_report(df)
metrics = compute_summary_metrics(df)
top_vars = get_top_variances(report, n=12)

# ── 1. Save variance report CSV ───────────────────────────────────────────────
report.to_csv("outputs/variance_report_fy2024.csv", index=False)
print("✅ Saved outputs/variance_report_fy2024.csv")

# ── 2. Save summary metrics JSON ──────────────────────────────────────────────
with open("outputs/summary_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("✅ Saved outputs/summary_metrics.json")

# ── 3. Save top variances JSON ────────────────────────────────────────────────
with open("outputs/top_variances.json", "w") as f:
    json.dump(top_vars, f, indent=2)
print("✅ Saved outputs/top_variances.json")

# ── 4. Generate sample AI report (static) ────────────────────────────────────
SAMPLE_REPORT = """## EXECUTIVE SUMMARY

NovaTech Solutions FY2024 shows a total net variance of -$112K (-0.5%) against budget, 
masking meaningful divergences at the account level. Revenue underperformed budget by 2.0% 
(-$257K), driven primarily by a confirmed SaaS revenue miss in Q3 (July–August) following 
a strategic customer churn event. Partially offsetting, Professional Services Revenue beat 
plan by 30.2% in October as backlog implementations completed ahead of schedule. On the expense 
side, Marketing overspent by $87K in Q3 (consistent with the planned NovaSummit brand 
campaign), Cloud Infrastructure costs exceeded budget by $28K in Q4 (linked to an 
architecture migration), and Legal fees spiked 64% in November due to strategic contract 
renegotiations. Five items warrant immediate CFO attention.

---

## DRIVER ATTRIBUTION

**1. Legal & Professional Fees — G&A — November 2024**
- Variance: +$9K (+64.0%) — CRITICAL — OVER_BUDGET
- Driver: Strategic partner contract renegotiation requiring external counsel engagement.  
  Per the budget assumptions, legal spikes of 50–100% are explicitly noted as possible  
  due to board-directed or contract-related activity. This is categorized as UNEXPECTED  
  in timing but not in nature.
- Status: Requires CFO sign-off; check whether retainer scope has expanded.

**2. Travel & Entertainment — Sales — April 2024**
- Variance: +$5K (+52.9%) — CRITICAL — OVER_BUDGET
- Driver: Annual Sales Kickoff (SKO) held April 8–10 in Las Vegas. Per the seasonality  
  calendar, April T&E variances of +50–60% for the Sales department are explicitly  
  budgeted. This variance is EXPECTED and consistent with plan.
- Status: No action required. Close monitoring of T&E normalization in May confirmed.

**3. Marketing & Advertising — Marketing — July 2024**
- Variance: +$38K (+44.5%) — CRITICAL — OVER_BUDGET
- Driver: Q3 brand awareness campaign launch and NovaSummit 2024 user conference  
  preparation. Budget assumptions explicitly state Q3 marketing is front-loaded for  
  the annual conference; variances up to +40% are planned. July came in at +44.5%,  
  slightly above the +40% guidance — the excess ~$4K warrants a brief note but is  
  not material.
- Status: EXPECTED per plan. Conference spend should normalize sharply in October.

**4. Product Revenue (SaaS) — Sales — July & August 2024**
- Variance: -$202K (-20.2% and -23.4%) — HIGH — MISS
- Driver: Confirmed churn of a financial services customer representing ~$420K ARR  
  following an M&A event at the customer (flagged in company overview Q3 business events).  
  Revenue recognition ceased in July; the miss persists through Q3 before partially  
  recovering as new logos ramp.
- Status: UNEXPECTED churn event (M&A-driven, not performance-related). Forecast  
  has been revised to reflect a -5% headwind on 4001 for remainder of year.

**5. Cloud Infrastructure — Engineering — October & November 2024**
- Variance: +$17K (+34.9%) and +$12K (+24.4%) — CRITICAL / HIGH — OVER_BUDGET
- Driver: Architecture migration (batch processing change) in September triggered  
  unexpected compute auto-scaling in Q4. Per the company overview, an infrastructure  
  incident in October/November is explicitly documented as a known event.
- Status: Engineering team has since implemented reserved instance changes. Expect  
  normalization in December.

**6. Professional Services Revenue — Sales — October 2024**
- Variance: +$37K (+30.2%) — CRITICAL — BEAT
- Driver: Q3 implementations completing ahead of schedule, pulling milestone revenue  
  recognition into October. Directly tied to the Enterprise logo onboarding backlog  
  noted in Q3 business events. High-quality favorable variance.
- Status: Positive. Validate with CS team that remaining milestones are on track for Q4.

---

## RISK FLAGS

1. **SaaS Revenue Headwind Persists:** The Q3 churn event reduces ARR by ~$420K  
   annualized. Unless offset by new logo closings in Q4, full-year SaaS revenue  
   will likely end 3–5% below budget. CRO should provide updated Q4 pipeline confidence  
   by mid-December.

2. **Infrastructure Cost Baseline Has Shifted:** The October/November cloud overruns  
   suggest the prior budget infrastructure cost model may be understated. FY2025  
   budget should reflect the new compute baseline post-migration, not the legacy rate.

3. **Legal Retainer Scope Creep:** A second significant legal spike in the same fiscal  
   year (Q4 2023 + November 2024) suggests the base legal retainer budget may be  
   structurally low. Consider increasing the FY2025 legal budget by $5–8K/month  
   or establishing a $25K contingency reserve.

4. **Full-Year Forecast at Risk:** Forecast vs. budget is -0.4%. Combined with the  
   confirmed SaaS revenue miss and elevated Q4 infrastructure costs, operating margin  
   will likely end below the -3% target. Recommend updated Q4 scenario modeling  
   before board presentation.

5. **Sales Capacity Risk:** The Sales Kickoff investment in April and two new rep  
   hires in May are trailing indicators. If Q4 pipeline coverage is insufficient  
   (<3x quota), FY2025 H1 revenue is at risk.

---

## RECOMMENDED ACTIONS

1. **CFO Review (This Week):** Obtain invoice detail for November legal fees.  
   Determine if the strategic partner renegotiation has a defined scope end-date  
   or if additional engagement is expected in December/Q1.

2. **Revenue Bridge to Board:** Prepare a one-page ARR bridge for the Q3 board  
   package showing: opening ARR → new logos → expansion → churn (with M&A attribution)  
   → closing ARR. Contextualize the SaaS miss as a non-recurring churn event.

3. **Infrastructure Cost Review (Engineering/Finance):** Engineering to provide a  
   post-migration infrastructure cost run-rate estimate for December and FY2025  
   planning. Reserve 12% of SaaS revenue (vs. current 11%) as the infrastructure  
   cost model for FY2025 budgeting.

4. **Q4 Reforecast:** Given 3 material variances with forecast implications (SaaS  
   revenue, infrastructure, legal), issue a revised full-year forecast by December 15  
   before the board package is due.
"""

with open("outputs/sample_ai_report.md", "w") as f:
    f.write(SAMPLE_REPORT)
print("✅ Saved outputs/sample_ai_report.md")

# ── 5. Generate charts as HTML ────────────────────────────────────────────────

# Chart 1: Variance heatmap
heat_data = df.groupby(["account_type", "period"]).apply(
    lambda g: pd.Series({
        "variance_pct": (g["actual_amount"].sum() - g["budget_amount"].sum()) / g["budget_amount"].sum() * 100
    })
).reset_index()
pivot = heat_data.pivot(index="account_type", columns="period", values="variance_pct")

fig1 = go.Figure(data=go.Heatmap(
    z=pivot.values,
    x=[p[-2:]+"/"+p[:4] for p in pivot.columns],
    y=pivot.index,
    colorscale=[[0,"#166534"],[0.5,"#0f172a"],[1,"#991b1b"]],
    zmid=0,
    text=np.round(pivot.values,1),
    texttemplate="%{text}%",
    textfont={"size":11,"color":"white"},
    colorbar=dict(title="Var %", tickfont=dict(color="#94a3b8")),
))
fig1.update_layout(
    title="FY2024 Budget Variance Heatmap by Account Type & Month",
    paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
    font=dict(color="#94a3b8", size=12),
    height=300,
)
fig1.write_html("outputs/chart_heatmap.html")
print("✅ Saved outputs/chart_heatmap.html")

# Chart 2: Top variances bar
top_df = report.head(10).copy()
colors = [{"CRITICAL":"#ef4444","HIGH":"#f97316","MEDIUM":"#eab308","LOW":"#22c55e"}[s]
          for s in top_df["severity"]]
fig2 = go.Figure(go.Bar(
    x=top_df["variance_pct"] * 100,
    y=[f"{r['account_name'][:25]} ({r['department_name'][:8]}) {r['period']}"
       for _, r in top_df.iterrows()],
    orientation="h",
    marker_color=colors,
    text=[f"{v:+.1f}%" for v in top_df["variance_pct"]*100],
    textposition="outside",
))
fig2.update_layout(
    title="Top 10 Flagged Variances by Severity",
    paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
    font=dict(color="#94a3b8"),
    xaxis=dict(gridcolor="#1e2d4a", title="Variance %"),
    height=420,
)
fig2.write_html("outputs/chart_top_variances.html")
print("✅ Saved outputs/chart_top_variances.html")

# Chart 3: SaaS revenue trend
saas = df[df["account_id"]=="4001"].groupby("period").agg(
    budget=("budget_amount","sum"),
    actual=("actual_amount","sum"),
    forecast=("forecast_amount","sum"),
).reset_index()
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=saas["period"],y=saas["budget"],name="Budget",
    line=dict(color="#60a5fa",dash="dash",width=2),mode="lines+markers"))
fig3.add_trace(go.Scatter(x=saas["period"],y=saas["actual"],name="Actual",
    line=dict(color="#f97316",width=3),mode="lines+markers",marker=dict(size=8)))
fig3.add_trace(go.Scatter(x=saas["period"],y=saas["forecast"],name="Forecast",
    line=dict(color="#a78bfa",dash="dot",width=2),mode="lines"))
fig3.update_layout(
    title="Product Revenue (SaaS) — Budget vs Actual vs Forecast FY2024",
    paper_bgcolor="#0f172a",plot_bgcolor="#0f172a",
    font=dict(color="#94a3b8"),
    yaxis=dict(gridcolor="#1e2d4a"),
    legend=dict(bgcolor="#0f172a",font=dict(color="#94a3b8")),
    height=380,
)
fig3.write_html("outputs/chart_saas_revenue.html")
print("✅ Saved outputs/chart_saas_revenue.html")

print(f"\n🎉 Demo outputs generated in /outputs/")
print(f"   - variance_report_fy2024.csv ({len(report)} flagged variances)")
print(f"   - summary_metrics.json")
print(f"   - top_variances.json")
print(f"   - sample_ai_report.md (pre-written executive report)")
print(f"   - 3 interactive HTML charts")
