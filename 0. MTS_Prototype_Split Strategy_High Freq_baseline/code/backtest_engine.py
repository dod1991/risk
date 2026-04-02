import numpy as np

# ============================================
# TRANSACTION COSTS (STEP 5: BALANCED MODEL)
# ============================================

def apply_transaction_costs(df):

    # ----------------------------------------
    # Base turnover
    # ----------------------------------------
    df["turnover"] = df["position"].diff().abs()
    prev_pos = df["position"].shift(1).abs().fillna(0)

    # Leverage-aware turnover
    df["effective_turnover"] = df["turnover"] * (1 + prev_pos)

    # ----------------------------------------
    # REGIME DETECTION (USING VIX)
    # ----------------------------------------
    vix = df["VIX"]

    low_vol = vix < 15
    mid_vol = (vix >= 15) & (vix < 25)
    high_vol = vix >= 25

    # ----------------------------------------
    # SPREAD MODEL (REGIME DEPENDENT)
    # ----------------------------------------
    spread = np.where(
        low_vol, 0.00005,      # 0.5 bp
        np.where(mid_vol, 0.0001, 0.0002)   # 1 bp / 2 bp
    )

    # ----------------------------------------
    # SLIPPAGE MODEL (BALANCED)
    # ----------------------------------------
    vol = df["realised_vol_20"].fillna(0.01)

    # Middle ground between original and step 4
    slippage = 0.00015 + 0.0003 * (vol / 0.02)

    # Increase slippage for larger trades
    slippage *= (1 + 0.75 * df["effective_turnover"])

    # ----------------------------------------
    # MARKET IMPACT (NONLINEAR, MODERATED)
    # ----------------------------------------
    impact = 0.0002 * np.sqrt(df["effective_turnover"])

    # ----------------------------------------
    # COMBINE COST COMPONENTS
    # ----------------------------------------
    total_cost_rate = spread + slippage + impact

    # ----------------------------------------
    # APPLY BASE COSTS
    # ----------------------------------------
    df["costs"] = df["effective_turnover"] * total_cost_rate

    # ----------------------------------------
    # REGIME MULTIPLIER (PARTIAL)
    # ----------------------------------------
    regime_multiplier = np.where(
        high_vol, 1.5,
        np.where(mid_vol, 1.1, 1.0)
    )

    df["costs"] *= regime_multiplier

    # ----------------------------------------
    # FLIP PENALTY (MODERATED)
    # ----------------------------------------
    flip = (df["position"] * df["position"].shift(1) < 0).astype(int)

    df["costs"] += flip * (spread + slippage)

    return df


# ============================================
# BACKTEST ENGINE
# ============================================

def run_backtest(df):

    df["strategy_returns"] = df["position"] * df["returns"]

    df = apply_transaction_costs(df)

    df["net_returns"] = df["strategy_returns"] - df["costs"]

    df["equity_curve"] = (1 + df["net_returns"]).cumprod()

    return df


# ============================================
# METRICS
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
# FULL PIPELINE ENTRY POINT
# ============================================

def apply_backtest(df):

    print("\n--- RUNNING BACKTEST (STEP 5: BALANCED COST MODEL) ---\n")

    df = run_backtest(df)
    metrics = compute_metrics(df)

    print("\n--- BACKTEST RESULTS ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return df, metrics