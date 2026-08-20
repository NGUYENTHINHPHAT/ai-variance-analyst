---
title: AI Variance Analyst
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.32.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# 📊 AI Variance Analyst — FP&A Copilot

> **Live Demo:** https://huggingface.co/spaces/PhatNguyen39/ai-variance-analyst

> **Automated budget vs. actual analysis powered by LLMs + RAG — no hallucination, full auditability.**

A production-grade ML portfolio project demonstrating data engineering, statistical anomaly detection, Retrieval-Augmented Generation (RAG), and LLM-powered executive reporting for Finance & FP&A use cases.

---

## 🎯 Problem Statement

Finance teams spend significant time each month answering the same questions:
- *"Why did marketing spend spike in Q3?"*
- *"What's driving the SaaS revenue miss?"*
- *"Which variances are expected vs. actually concerning?"*

This system automates that workflow — ingesting structured financial data, detecting significant deviations statistically, grounding an LLM in company-specific context via RAG, and generating CFO-ready variance reports in seconds.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         STREAMLIT UI                            │
│   Dashboard │ AI Report │ Drill-Down Analyst │ Trend Charts     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
          ┌────────────────┼──────────────────┐
          ▼                ▼                  ▼
┌─────────────────┐ ┌───────────────┐ ┌──────────────────┐
│  VARIANCE       │ │  RAG PIPELINE │ │  LLM ANALYST     │
│  ENGINE         │ │               │ │  (Claude API)    │
│                 │ │ ┌───────────┐ │ │                  │
│ • Rule-based    │ │ │ Chroma DB │ │ │ • System prompt  │
│   flags (>10%)  │ │ │ (vector   │ │ │ • Grounded       │
│ • Z-score       │ │ │  store)   │ │ │   prompting      │
│   outliers      │ │ └─────┬─────┘ │ │ • Structured     │
│ • Trend         │ │       │       │ │   output         │
│   detection     │ │ Embed │       │ │ • No hallucin-   │
│                 │ │ Query │       │ │   ation          │
└────────┬────────┘ └───────┼───────┘ └──────────────────┘
         │                  │
         │          ┌───────┴──────────┐
         │          │   RAG DOCUMENTS  │
         │          │                  │
         │          │ • Budget assump. │
         │          │ • Company overv. │
         │          │ • Hist. variances│
         │          │ • Account defs.  │
         │          │ • Seasonality    │
         │          └──────────────────┘
         │
┌────────┴────────────────────────┐
│         DATA LAYER              │
│                                 │
│  Mock ERP (CSV / DuckDB)        │
│  • budget.csv (828 rows)        │
│  • actuals.csv (828 rows)       │
│  • forecast.csv (828 rows)      │
│  • 14 GL accounts               │
│  • 6 departments                │
│  • 12 months FY2024             │
│  • Planted anomalies for demo   │
└─────────────────────────────────┘
```

---

## 🔑 Key Technical Decisions

### Why RAG instead of fine-tuning?
The LLM needs company-specific context (budget assumptions, business events, seasonality patterns) that changes every fiscal year. RAG allows updating context without retraining. It also ensures auditability — every claim can be traced to a source document.

### How hallucination is prevented
- The system prompt explicitly forbids inventing numbers
- All variance figures are injected as structured JSON (not described in natural language)
- RAG context is retrieved per-variance, grounding explanations in documented company knowledge
- Uncertainty is flagged explicitly when context doesn't explain a variance

### Statistical approach
Variance detection combines three methods:
1. **Rule-based**: |actual - budget| / budget > 10% AND > $5K
2. **Statistical (Z-score)**: Per-account Z-score > 2.0 across all periods
3. **Trend detection**: 3+ consecutive periods in same direction > 5%

This avoids over-flagging noise while catching both magnitude-based and pattern-based anomalies.

---

## 📁 Project Structure

```
ai-variance-analyst/
│
├── app.py                          # Main Streamlit application
├── generate_demo_outputs.py        # Pre-compute charts & reports for demo
├── requirements.txt
├── .env.example
│
├── src/
│   ├── generate_data.py            # Mock ERP data generator
│   ├── variance_engine.py          # Statistical variance detection
│   ├── rag_pipeline.py             # ChromaDB + sentence-transformers RAG
│   └── llm_analyst.py              # Claude API integration
│
├── tests/                          # Deterministic unit tests (pytest, no API calls)
│   ├── test_variance_engine.py     # Flagging rules, thresholds, dedup, direction logic
│   └── test_eval_narrative.py      # Tests the eval framework's own parsing logic
│
├── eval/
│   └── eval_narrative.py           # LLM output groundedness + structure eval (calls Claude API)
│
├── tableau/
│   ├── build_extract.py            # Builds the Tableau .hyper extract from variance_engine
│   ├── extract/                    # Generated budget_variance.hyper (gitignored)
│   └── dist/
│       └── budget_variance.twbx    # Packaged Tableau workbook, ready to open
│
├── rag_docs/                       # Company context documents for RAG
│   ├── budget_assumptions_fy2024.md
│   ├── company_overview.md
│   ├── historical_variances.md
│   ├── account_definitions.md
│   └── seasonality_patterns.md
│
├── data/
│   ├── raw/                        # Generated CSVs (budget, actuals, forecast)
│   └── processed/                  # Combined dataset + variance report
│
└── outputs/                        # Pre-computed demo artifacts
    ├── variance_report_fy2024.csv
    ├── summary_metrics.json
    ├── top_variances.json
    ├── sample_ai_report.md         # Full AI report (no API key needed to view)
    ├── chart_heatmap.html
    ├── chart_top_variances.html
    └── chart_saas_revenue.html
```

---

## 🚀 Quickstart

### Prerequisites
- Python 3.10+
- An Anthropic API key ([get one here](https://console.anthropic.com))

### 1. Clone the repository
```bash
git clone https://github.com/PhatNguyen39/ai-variance-analyst.git
cd ai-variance-analyst
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your API key
```bash
cp .env.example .env
# Edit .env and add your key:
#   ANTHROPIC_API_KEY=sk-ant-your-key-here

# Or export directly:
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 5. Generate mock ERP data
```bash
python src/generate_data.py
```
Output: `data/raw/` and `data/processed/erp_combined.csv`

### 6. (Optional) Pre-generate demo outputs
```bash
python generate_demo_outputs.py
```
This creates static charts and a sample AI report in `outputs/` — useful for GitHub showcase without an API key.

### 7. Launch the Streamlit app
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 8. (Optional) Run the test suite
```bash
pytest tests/ -v
```
No API key needed — see [Testing & Evaluation](#-testing--evaluation) below.

### 9. (Optional) Build the Tableau extract
```bash
python tableau/build_extract.py
```
Or just open the pre-built `tableau/dist/budget_variance.twbx` directly — see [Tableau Dashboard](#-tableau-dashboard).

---

## 📸 Demo Walkthrough

### Tab 1 — Variance Dashboard
- Color-coded severity table (CRITICAL / HIGH / MEDIUM / LOW)
- Budget variance heatmap by account type and month
- Severity distribution donut chart
- Budget-to-Actual waterfall chart

### Tab 2 — AI Executive Report
1. Enter your API key in the sidebar
2. Select number of top variances to analyze
3. Click **"Generate Executive Report"**
4. The system:
   - Retrieves relevant context from the RAG index for each variance
   - Sends grounded prompt to Claude
   - Returns a structured Executive Summary, Driver Attribution, Risk Flags, and Recommended Actions

### Tab 3 — Drill-Down Analyst
- Select any account / department / period combination
- Click **"Explain This Variance"** for a detailed AI explanation
- Ask free-form questions to the FP&A Copilot chatbot

### Tab 4 — Trend Charts
- Budget vs Actual vs Forecast line charts per account
- Monthly variance % bar chart
- Full-year YTD variance comparison across all accounts

---

## 🌱 Planted Anomalies (Demo Scenarios)

The mock dataset includes realistic anomalies for demonstration:

| Event | Period | Variance | Context |
|---|---|---|---|
| Marketing brand campaign overspend | Jul–Sep 2024 | +38–42% | Planned NovaSummit conference (expected) |
| SaaS revenue miss (customer churn) | Jul–Sep 2024 | -15–21% | Enterprise customer M&A churn event |
| Cloud infrastructure spike | Oct–Nov 2024 | +28–31% | Architecture migration side effect |
| T&E spike — Sales SKO | Apr 2024 | +51% | Annual Sales Kickoff (expected) |
| Legal fees spike | Nov 2024 | +66% | Strategic contract renegotiation |
| PS Revenue beat | Oct–Nov 2024 | +19–27% | Backlog implementations completing early |

This mix of **expected** and **unexpected** variances demonstrates the system's ability to distinguish between planned events and genuine surprises.

---

## 🧪 Testing & Evaluation

This project has two very different kinds of correctness to check, and tests them differently.

### Deterministic engine tests
`variance_engine.py`'s flagging logic (rule-based %, Z-score outliers, trend detection, severity
classification, direction/BEAT-MISS labeling) is pure functions over data — tested like any other
code, with hand-computed synthetic fixtures and no network calls.

```bash
pytest tests/ -v
```
33 tests, runs in a couple seconds, no API key required. Covers exact threshold boundaries
(10% / 20% / 30% severity cutoffs, the $5K minimum-dollar floor), Z-score outlier detection
(hand-verified against `scipy.stats.zscore`), 3-consecutive-period trend firing, dedup when a row
is flagged by multiple methods at once, and the Revenue BEAT/MISS direction flip.

### LLM narrative groundedness eval
This README claims *"no hallucination, full auditability."* `eval/eval_narrative.py` is what
actually tests that claim rather than leaving it asserted: it extracts every `$` amount and `%`
figure Claude cites in a generated report, and checks each one traces back to the
`variance_data` / `summary_metrics` JSON that was actually in its prompt (within a small rounding
tolerance). A number that matches nothing in the source data is flagged as a hallucination
candidate. It also checks structural compliance — all four required report sections present,
Executive Summary under the prompt's own 150-word limit.

```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here
python eval/eval_narrative.py
```
This calls the real Claude API (a handful of requests) and depends on live data, so it's a manual
script rather than a CI gate — run it after touching `llm_analyst.py`'s prompts or system rules.
Its own parsing/matching logic is unit-tested separately in `tests/test_eval_narrative.py` (no API
key needed), so a bug in the checker can't silently make every groundedness score meaningless.

### Not yet covered
RAG retrieval quality (`rag_pipeline.py`) — precision/recall@k against a hand-labeled set of
queries and expected source documents — is the natural next layer. See
[Extending This Project](#-extending-this-project).

---

## 📊 Tableau Dashboard

The same variance analysis the Streamlit app shows is also available as a Tableau extract, for
audiences who'd rather work in Tableau than run a Python app — no server, no API key, no per-view
LLM cost.

`tableau/build_extract.py` reuses `variance_engine.build_variance_report()` directly — it doesn't
re-derive the analysis — so the workbook always shows exactly the same numbers as the live app. It
writes `tableau/extract/budget_variance.hyper` with two tables:

| Table | Grain | Powers |
|---|---|---|
| `Extract` | period × account × department (wide) | KPI tiles, flagged-variance table, heatmap, severity mix |
| `Trend` | period × account × department × metric_type (long) | Trend line chart, budget/actual/forecast comparison |

`Trend` rows exist as `Actual` only for closed periods and `Forecast` only for open ones, so a
Tableau line mark breaks cleanly at the actual/forecast boundary with no null-connect tricks.

```bash
python tableau/build_extract.py
```

A pre-built, packaged workbook is checked in at `tableau/dist/budget_variance.twbx` — open it
directly in Tableau Desktop or Tableau Public, no build step required. It's self-contained (the
extract is embedded) with seven worksheets — KPI Summary, Variance by Department, Flag Severity
Mix, Variance Heatmap, Flagged Variances, Trend, Metric Totals — combined into one
"Budget Variance Overview" dashboard.

> **Note:** this workbook was hand-authored directly against the Tableau `.twb` XML schema (no
> Tableau Desktop was available in the environment it was built in). Every field reference is
> validated against the real `.hyper` schema, but visual rendering hasn't been confirmed inside
> Tableau itself — treat it as a strong first draft, not a GUI-verified export.

The Hyper extract is a snapshot, not a live connection — regenerate it (`python
tableau/build_extract.py`) any time the underlying data changes. `*.hyper` files are gitignored as
build artifacts.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Data generation | Python, pandas, numpy |
| Statistical detection | scipy (Z-score), pandas |
| Vector store / RAG | ChromaDB, sentence-transformers |
| LLM | Anthropic Claude (claude-sonnet-4-20250514) |
| Frontend | Streamlit, Plotly |
| Storage | CSV / DuckDB-compatible format |
| Testing | pytest (deterministic engine tests + eval-framework self-tests) |
| BI extract | Tableau Hyper API (`tableauhyperapi`) |

---

## 💡 Extending This Project

Some ideas for further development:

- **Connect real ERP data**: Replace `generate_data.py` with a NetSuite/SAP API connector
- **Add DuckDB**: Replace CSV reads with SQL queries for better performance at scale
- **MLflow tracking**: Log variance detection model performance over time
- **Slack/email alerts**: Auto-notify department heads when their accounts exceed thresholds
- **Multi-company**: Extend RAG index to support multiple subsidiaries or business units
- **Forecast accuracy tracking**: Build a model that tracks how accurate monthly forecasts have been
- **RAG retrieval eval**: Precision/recall@k against a hand-labeled query → expected-source-doc set
- **Tableau Server/Cloud publishing**: Replace the manual `.hyper` snapshot with a scheduled extract refresh

---

## 🎓 What This Demonstrates

| Skill | How It's Shown |
|---|---|
| **Data Engineering** | ERP schema design, multi-table joins, fiscal period modeling |
| **Statistical ML** | Multi-method anomaly detection (rule + Z-score + trend) |
| **LLM Grounding / RAG** | ChromaDB vector store, context retrieval, hallucination prevention |
| **Prompt Engineering** | Structured system prompt with strict output format requirements |
| **Financial Domain Knowledge** | GL account modeling, FP&A reporting conventions, GAAP terminology |
| **Production Engineering** | Modular codebase, caching, error handling, API key management |
| **Executive Communication** | CFO-level report format, actionable recommendations |
| **Testing & LLM Evaluation** | Deterministic regression tests + a groundedness eval that verifies the "no hallucination" claim numerically |
| **BI / Analytics Integration** | Tableau Hyper extract reusing the same engine, so BI and app numbers never diverge |

---

## 📄 License

MIT License — free to use, modify, and distribute.
