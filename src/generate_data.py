"""
Mock ERP Data Generator for AI Variance Analyst
Generates realistic financial data mimicking NetSuite/SAP exports
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

np.random.seed(42)

# ── Company configuration ──────────────────────────────────────────────────────
COMPANY = "NovaTech Solutions Inc."
FISCAL_YEAR = 2024
PERIODS = [f"2024-{str(m).zfill(2)}" for m in range(1, 13)]
CURRENT_PERIOD = "2024-11"  # Analysis cutoff: Jan–Nov 2024 closed, Dec open

DEPARTMENTS = {
    "D01": "Engineering",
    "D02": "Sales",
    "D03": "Marketing",
    "D04": "G&A",
    "D05": "Customer Success",
    "D06": "Product",
}

#14 accounts
GL_ACCOUNTS = {
    # Revenue
    "4001": {"name": "Product Revenue - SaaS",       "type": "Revenue",   "normal": "credit"},
    "4002": {"name": "Professional Services Revenue", "type": "Revenue",   "normal": "credit"},
    "4003": {"name": "Support & Maintenance Revenue", "type": "Revenue",   "normal": "credit"},
    # COGS
    "5001": {"name": "Cloud Infrastructure Costs",   "type": "COGS",      "normal": "debit"},
    "5002": {"name": "Third-Party Software Licenses","type": "COGS",      "normal": "debit"},
    "5003": {"name": "Professional Services COGS",   "type": "COGS",      "normal": "debit"},
    # OpEx
    "6001": {"name": "Salaries & Benefits",          "type": "OpEx",      "normal": "debit"},
    "6002": {"name": "Sales Commissions",            "type": "OpEx",      "normal": "debit"},
    "6003": {"name": "Marketing & Advertising",      "type": "OpEx",      "normal": "debit"},
    "6004": {"name": "Travel & Entertainment",       "type": "OpEx",      "normal": "debit"},
    "6005": {"name": "Software & Tools",             "type": "OpEx",      "normal": "debit"},
    "6006": {"name": "Depreciation & Amortization",  "type": "OpEx",      "normal": "debit"},
    "6007": {"name": "Rent & Facilities",            "type": "OpEx",      "normal": "debit"},
    "6008": {"name": "Legal & Professional Fees",    "type": "OpEx",      "normal": "debit"},
}

# Monthly budget base amounts ($000s) by account
BUDGET_BASE = {
    "4001": 850,  "4002": 180,  "4003": 120,
    "5001": 95,   "5002": 42,   "5003": 88,
    "6001": 310,  "6002": 68,   "6003": 95,
    "6004": 22,   "6005": 18,   "6006": 25,
    "6007": 35,   "6008": 28,
}

# Seasonality multipliers (1.0 = flat) for 12 months
SEASONALITY = {
    "4001": [0.85, 0.88, 0.95, 0.98, 1.02, 1.05, 1.08, 1.10, 1.12, 1.15, 1.18, 1.20],
    "4002": [0.70, 0.80, 1.00, 1.10, 1.05, 1.00, 0.90, 0.95, 1.10, 1.15, 1.10, 1.15],
    "4003": [0.90, 0.90, 0.95, 1.00, 1.00, 1.00, 1.00, 1.00, 1.05, 1.05, 1.05, 1.10],
    "5001": [1.00] * 12,
    "5002": [1.00] * 12,
    "5003": [0.75, 0.85, 1.00, 1.10, 1.05, 1.00, 0.90, 0.95, 1.10, 1.10, 1.05, 1.15],
    "6001": [1.00] * 12,
    "6002": [0.70, 0.80, 1.00, 1.10, 1.05, 1.00, 0.90, 0.95, 1.10, 1.15, 1.10, 1.15],
    "6003": [0.60, 0.70, 0.90, 1.00, 1.10, 1.20, 1.30, 1.20, 1.10, 1.00, 0.90, 0.90],
    "6004": [0.80, 0.80, 1.10, 1.20, 1.10, 0.90, 0.70, 0.80, 1.20, 1.20, 1.10, 0.90],
    "6005": [1.00] * 12,
    "6006": [1.00] * 12,
    "6007": [1.00] * 12,
    "6008": [1.00] * 12,
}

# Department allocation: how each account's total is split across departments
DEPT_WEIGHTS = {
    "4001": {"D01":0.0, "D02":0.5, "D03":0.2, "D04":0.0, "D05":0.2, "D06":0.1},
    "4002": {"D01":0.0, "D02":0.6, "D03":0.0, "D04":0.0, "D05":0.3, "D06":0.1},
    "4003": {"D01":0.0, "D02":0.2, "D03":0.0, "D04":0.0, "D05":0.7, "D06":0.1},
    "5001": {"D01":0.5, "D02":0.1, "D03":0.1, "D04":0.1, "D05":0.1, "D06":0.1},
    "5002": {"D01":0.4, "D02":0.1, "D03":0.1, "D04":0.1, "D05":0.2, "D06":0.1},
    "5003": {"D01":0.3, "D02":0.1, "D03":0.0, "D04":0.0, "D05":0.5, "D06":0.1},
    "6001": {"D01":0.35,"D02":0.20,"D03":0.12,"D04":0.10,"D05":0.13,"D06":0.10},
    "6002": {"D01":0.0, "D02":0.80,"D03":0.0, "D04":0.0, "D05":0.20,"D06":0.0},
    "6003": {"D01":0.0, "D02":0.15,"D03":0.70,"D04":0.05,"D05":0.05,"D06":0.05},
    "6004": {"D01":0.20,"D02":0.35,"D03":0.15,"D04":0.10,"D05":0.10,"D06":0.10},
    "6005": {"D01":0.30,"D02":0.15,"D03":0.15,"D04":0.10,"D05":0.15,"D06":0.15},
    "6006": {"D01":0.25,"D02":0.15,"D03":0.15,"D04":0.15,"D05":0.15,"D06":0.15},
    "6007": {"D01":0.20,"D02":0.15,"D03":0.15,"D04":0.20,"D05":0.15,"D06":0.15},
    "6008": {"D01":0.10,"D02":0.10,"D03":0.10,"D04":0.50,"D05":0.10,"D06":0.10},
}


def generate_budget():
    rows = []
    for acct, base in BUDGET_BASE.items():
        for i, period in enumerate(PERIODS):
            for dept, weight in DEPT_WEIGHTS[acct].items():
                if weight == 0:
                    continue
                amount = base * SEASONALITY[acct][i] * weight * 1000
                rows.append({
                    "period": period,
                    "account_id": acct,
                    "account_name": GL_ACCOUNTS[acct]["name"],
                    "account_type": GL_ACCOUNTS[acct]["type"],
                    "department_id": dept,
                    "department_name": DEPARTMENTS[dept],
                    "budget_amount": round(amount, 2),
                })
    return pd.DataFrame(rows)


def generate_actuals(budget_df):
    """Generate actuals with planted variances for demo realism."""
    rows = []
    closed_df = budget_df[budget_df["period"] <= CURRENT_PERIOD]

    # Planted anomalies: (account_id, dept_id, period_index, variance_pct)
    anomalies = {
        # Marketing overspend Q3 brand campaign
        ("6003", "D03", 6):  +0.42,   # July  +42%
        ("6003", "D03", 7):  +0.38,   # Aug   +38%
        ("6003", "D03", 8):  +0.25,   # Sep   +25%
        # Revenue miss - SaaS in Q3 (churn event)
        ("4001", "D02", 6):  -0.18,   # July  -18%
        ("4001", "D02", 7):  -0.21,   # Aug   -21%
        ("4001", "D02", 8):  -0.15,   # Sep   -15%
        # Cloud cost spike (infra scaling issue)
        ("5001", "D01", 9):  +0.31,   # Oct   +31%
        ("5001", "D01", 10): +0.28,   # Nov   +28%
        # T&E spike - Sales team conference
        ("6004", "D02", 3):  +0.55,   # Apr   +55%
        # Legal fees spike (contract negotiations)
        ("6008", "D04", 10): +0.62,   # Nov   +62%
        # PS Revenue beat Q4
        ("4002", "D02", 9):  +0.24,   # Oct   +24%
        ("4002", "D02", 10): +0.19,   # Nov   +19%
    }

    for _, row in closed_df.iterrows():
        period_idx = PERIODS.index(row["period"])
        key = (row["account_id"], row["department_id"], period_idx)
        planted_var = anomalies.get(key, 0)

        # Natural noise: mean 0, std dev 4%
        noise = np.random.normal(0, 0.04)
        total_var = planted_var + noise

        actual = row["budget_amount"] * (1 + total_var)
        rows.append({
            "period": row["period"],
            "account_id": row["account_id"],
            "account_name": row["account_name"],
            "account_type": row["account_type"],
            "department_id": row["department_id"],
            "department_name": row["department_name"],
            "actual_amount": round(actual, 2),
        })
    return pd.DataFrame(rows)


def generate_forecast(budget_df, actuals_df):
    """Rolling forecast: use actuals YTD + updated forward estimates."""
    rows = []
    merged = budget_df.merge(actuals_df, on=["period","account_id","account_name",
                                               "account_type","department_id","department_name"],
                             how="left")
    for _, row in merged.iterrows():
        # Closed months: anchor forecast to actuals; open months: revised budget estimate
        if row["period"] <= CURRENT_PERIOD and pd.notna(row.get("actual_amount")):
            forecast = row["actual_amount"] * (1 + np.random.normal(0, 0.01))
        else:
            adj = -0.05 if row["account_id"] == "4001" else np.random.normal(0, 0.02)
            forecast = row["budget_amount"] * (1 + adj)
        rows.append({
            "period": row["period"],
            "account_id": row["account_id"],
            "account_name": row["account_name"],
            "account_type": row["account_type"],
            "department_id": row["department_id"],
            "department_name": row["department_name"],
            "forecast_amount": round(forecast, 2),
        })
    return pd.DataFrame(rows)


def main():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    print("Generating budget data...")
    budget = generate_budget()
    budget.to_csv("data/raw/budget.csv", index=False)

    print("Generating actuals data...")
    actuals = generate_actuals(budget)
    actuals.to_csv("data/raw/actuals.csv", index=False)

    print("Generating forecast data...")
    forecast = generate_forecast(budget, actuals)
    forecast.to_csv("data/raw/forecast.csv", index=False)

    # Combined dataset: left-join so Nov–Dec retain budget/forecast rows with NaN actuals
    combined = budget.merge(actuals, on=["period","account_id","account_name",
                                          "account_type","department_id","department_name"],
                            how="left")
    combined = combined.merge(forecast, on=["period","account_id","account_name",
                                             "account_type","department_id","department_name"])
    combined.to_csv("data/processed/erp_combined.csv", index=False)

    # GL metadata
    gl_meta = [{"account_id": k, **v} for k, v in GL_ACCOUNTS.items()]
    with open("data/raw/gl_metadata.json", "w") as f:
        json.dump(gl_meta, f, indent=2)

    print(f"✅ Data generated: {len(budget)} budget rows, {len(actuals)} actual rows (through {CURRENT_PERIOD})")
    print(f"   Periods: {PERIODS[0]} → {PERIODS[-1]}")
    print(f"   Accounts: {len(GL_ACCOUNTS)} | Departments: {len(DEPARTMENTS)}")


if __name__ == "__main__":
    main()
