import numpy as np

# ============================================
# TRANSACTION COSTS
# ============================================

def apply_transaction_costs(df, cost_rate=0.0005):
    df["turnover"] = df["position"].diff().abs()
    df["costs"] = df["turnover"] * cost_rate
    return df


# ============================================
# BACKTEST ENGINE (PURE)
# ============================================

def run_backtest(df):

    # Strategy returns
    df["strategy_returns"] = df["position"] * df["returns"]

    # Costs
    df = apply_transaction_costs(df)

    df["net_returns"] = df["strategy_returns"] - df["costs"]

    # Equity curve
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
# FULL PIPELINE
# ============================================

def apply_backtest(df):

    print("\n--- RUNNING BACKTEST (PURE ENGINE) ---\n")

    df = run_backtest(df)
    metrics = compute_metrics(df)

    print("\n--- BACKTEST RESULTS ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return df, metrics