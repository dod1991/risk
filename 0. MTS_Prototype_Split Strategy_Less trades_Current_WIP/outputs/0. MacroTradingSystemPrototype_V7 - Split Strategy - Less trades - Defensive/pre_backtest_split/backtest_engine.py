import pandas as pd
import numpy as np

# ============================================
# CONFIG
# ============================================

TARGET_VOL = 0.15
MIN_VOL = 0.01
MAX_LEVERAGE = 3.0
MAX_POSITION = 1.0


# ============================================
# TRANSACTION COSTS
# ============================================

def apply_transaction_costs(df, cost_rate=0.0005):
    df["turnover"] = df["position"].diff().abs()
    df["costs"] = df["turnover"] * cost_rate
    return df


# ============================================
# BACKTEST CALCULATION
# ============================================

def run_backtest(df):

    # ========================================
    # VOL TARGETING
    # ========================================

    df["adj_vol"] = df["realised_vol_20"].clip(lower=MIN_VOL)

    df["vol_scaler"] = TARGET_VOL / df["adj_vol"]
    df["vol_scaler"] = df["vol_scaler"].clip(upper=MAX_LEVERAGE)

    # ========================================
    # BASE POSITION
    # ========================================

    df["raw_position"] = df["exposure"].shift(1)

    # ========================================
    # 🔥 STEP 6 FIX: CONTROLLED SIGNAL SIZING
    # ========================================

    df["signal_strength"] = df["quant_score_smooth"].abs()

    # Controlled boost
    signal_boost = 1 + 0.3 * df["signal_strength"]

    # Cap aggressiveness
    signal_boost = np.clip(signal_boost, 1.0, 1.25)

    df["position"] = (
        df["raw_position"]
        * df["vol_scaler"]
        * signal_boost
    )

    # ========================================
    # POSITION CAP
    # ========================================

    df["position"] = df["position"].clip(-MAX_POSITION, MAX_POSITION)

    # ========================================
    # SMOOTHING (STEP 2)
    # ========================================

    df["position"] = df["position"].ewm(span=3).mean()

    # ========================================
    # RETURNS
    # ========================================

    df["strategy_returns"] = df["position"] * df["returns"]

    # ========================================
    # COSTS
    # ========================================

    df = apply_transaction_costs(df)

    df["net_returns"] = df["strategy_returns"] - df["costs"]

    # ========================================
    # EQUITY CURVE
    # ========================================

    df["equity_curve"] = (1 + df["net_returns"]).cumprod()

    return df


# ============================================
# PERFORMANCE METRICS
# ============================================

def compute_metrics(df):

    returns = df["net_returns"].dropna()

    total_return = df["equity_curve"].iloc[-1] - 1

    sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-6)

    cum = df["equity_curve"]
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    max_dd = drawdown.min()

    trade_count = (df["position"].diff().abs() > 0).sum()
    avg_turnover = df["turnover"].mean()

    return {
        "Total Return": total_return,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Trades": trade_count,
        "Avg Turnover": avg_turnover
    }


# ============================================
# FULL BACKTEST PIPELINE
# ============================================

def apply_backtest(df):

    print("\n--- RUNNING BACKTEST (STEP 6 FIXED: CONTROLLED SIZING) ---\n")

    df = run_backtest(df)
    metrics = compute_metrics(df)

    print("\n--- BACKTEST RESULTS ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return df, metrics