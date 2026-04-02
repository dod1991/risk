from data_pipeline import run_pipeline
from risk_agent import apply_risk_agent
from quant_agent import apply_quant_agent
from macro_agent import apply_macro_agent
from decision_engine import run_decision_engine
from trade_engine import apply_trade_engine   # 🔥 NEW
from backtest_engine import apply_backtest

import os
from datetime import datetime

print("\n--- RUNNING FULL SYSTEM ---\n")

# ============================================
# STEP 1: DATA
# ============================================

df = run_pipeline()

# ============================================
# STEP 2: AGENTS
# ============================================

df = apply_risk_agent(df)
df = apply_quant_agent(df)
df = apply_macro_agent(df)

# ============================================
# STEP 3: DECISION ENGINE
# ============================================

df = run_decision_engine(df)

# ============================================
# STEP 3.5: TRADE ENGINE (NEW)
# ============================================

df = apply_trade_engine(df)

# ============================================
# STEP 4: BACKTEST
# ============================================

df, metrics = apply_backtest(df)

# ============================================
# STEP 5: ADD PnL FEATURES
# ============================================

def classify_pnl(x):
    if x > 0:
        return "PROFIT"
    elif x < 0:
        return "LOSS"
    else:
        return "FLAT"

df["PnL_flag"] = df["net_returns"].apply(classify_pnl)
df["PnL_value"] = df["net_returns"]
df["Cumulative_PnL"] = df["equity_curve"] - 1

# ============================================
# STEP 6: REORDER COLUMNS
# ============================================

priority_cols = [
    "PnL_flag",
    "PnL_value",
    "Cumulative_PnL",
    "exposure",
    "decision_reason"
]

remaining_cols = [col for col in df.columns if col not in priority_cols]
df = df[priority_cols + remaining_cols]

# ============================================
# STEP 7: CLEAN DATA
# ============================================

df = df.dropna()

# ============================================
# STEP 8: SAVE OUTPUT
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE_PATH = os.path.join(BASE_DIR, "backtest_output")

today = datetime.today()
year = str(today.year)
month = f"{today.month:02d}"
date_str = today.strftime("%Y-%m-%d")

folder_path = os.path.join(OUTPUT_BASE_PATH, year, month)
os.makedirs(folder_path, exist_ok=True)

file_path = os.path.join(folder_path, f"backtest_output_{date_str}.xlsx")

print("\n--- SAVING FINAL OUTPUT ---")
print("Saving to:", file_path)

df.to_excel(file_path)

print("SUCCESS: File saved")
print("\n--- SYSTEM COMPLETE ---\n")