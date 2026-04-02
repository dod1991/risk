import numpy as np
import pandas as pd

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
        low_vol, 0.00005,
        np.where(mid_vol, 0.0001, 0.0002)
    )

    # ----------------------------------------
    # SLIPPAGE MODEL (BALANCED)
    # ----------------------------------------
    vol = df["realised_vol_20"].fillna(0.01)

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
# CORE BACKTEST
# ============================================

def run_backtest(df):

    df["strategy_returns_gross"] = df["position"] * df["returns"]

    df = apply_transaction_costs(df)

    df["net_returns"] = df["strategy_returns_gross"] - df["costs"]
    df["equity_curve"] = (1 + df["net_returns"]).cumprod()

    return df


# ============================================
# HELPER METRICS
# ============================================

def annualized_return(returns, periods=252):
    returns = pd.Series(returns).dropna()
    if len(returns) == 0:
        return np.nan
    total = (1 + returns).prod()
    years = len(returns) / periods
    if years <= 0:
        return np.nan
    return total ** (1 / years) - 1


def annualized_volatility(returns, periods=252):
    returns = pd.Series(returns).dropna()
    if len(returns) == 0:
        return np.nan
    return returns.std() * np.sqrt(periods)


def sharpe_ratio(returns, periods=252):
    returns = pd.Series(returns).dropna()
    if len(returns) == 0:
        return np.nan
    return np.sqrt(periods) * returns.mean() / (returns.std() + 1e-6)


def compute_drawdown(equity_curve):
    equity_curve = pd.Series(equity_curve).dropna()
    if len(equity_curve) == 0:
        return np.nan
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()


# ============================================
# BENCHMARK + VALIDATION FEATURES
# ============================================

def add_validation_features(df):

    # Buy & hold benchmark
    df["benchmark_returns"] = df["returns"]
    df["benchmark_equity"] = (1 + df["benchmark_returns"]).cumprod()

    # Excess return
    df["excess_returns"] = df["net_returns"] - df["benchmark_returns"]
    df["excess_equity"] = (1 + df["excess_returns"]).cumprod()

    # Rolling Sharpe (1Y)
    rolling_mean = df["net_returns"].rolling(252).mean()
    rolling_std = df["net_returns"].rolling(252).std()
    df["rolling_sharpe_252"] = np.sqrt(252) * rolling_mean / (rolling_std + 1e-6)

    # Rolling benchmark Sharpe
    bench_rolling_mean = df["benchmark_returns"].rolling(252).mean()
    bench_rolling_std = df["benchmark_returns"].rolling(252).std()
    df["benchmark_rolling_sharpe_252"] = np.sqrt(252) * bench_rolling_mean / (bench_rolling_std + 1e-6)

    # Rolling alpha proxy
    df["rolling_excess_return_252"] = df["net_returns"].rolling(252).sum() - df["benchmark_returns"].rolling(252).sum()

    # Cost drag diagnostics
    df["cost_drag_cumulative"] = df["costs"].cumsum()
    df["gross_equity_curve"] = (1 + df["strategy_returns_gross"]).cumprod()

    # Trade direction diagnostics
    df["long_flag"] = (df["position"] > 0).astype(int)
    df["short_flag"] = (df["position"] < 0).astype(int)
    df["flat_flag"] = (df["position"] == 0).astype(int)

    return df


# ============================================
# REGIME BREAKDOWN
# ============================================

def compute_regime_metrics(df):

    regime_results = {}

    if "market_regime" not in df.columns:
        return regime_results

    for regime in ["trend", "chop"]:
        subset = df[df["market_regime"] == regime].copy()

        if len(subset) < 20:
            regime_results[regime] = {
                "Days": len(subset),
                "Total Return": np.nan,
                "Annual Return": np.nan,
                "Sharpe": np.nan,
                "Avg Net Return": np.nan,
                "Avg Cost": np.nan
            }
            continue

        regime_equity = (1 + subset["net_returns"].fillna(0)).cumprod()

        regime_results[regime] = {
            "Days": len(subset),
            "Total Return": regime_equity.iloc[-1] - 1,
            "Annual Return": annualized_return(subset["net_returns"]),
            "Sharpe": sharpe_ratio(subset["net_returns"]),
            "Avg Net Return": subset["net_returns"].mean(),
            "Avg Cost": subset["costs"].mean()
        }

    return regime_results


# ============================================
# COST STRESS TEST
# ============================================

def compute_cost_stress_tests(df):

    scenarios = {
        "Base": 1.0,
        "Costs x1.5": 1.5,
        "Costs x2.0": 2.0
    }

    results = {}

    for name, multiple in scenarios.items():
        stressed_net = df["strategy_returns_gross"] - (df["costs"] * multiple)
        stressed_equity = (1 + stressed_net.fillna(0)).cumprod()

        results[name] = {
            "Total Return": stressed_equity.iloc[-1] - 1,
            "Annual Return": annualized_return(stressed_net),
            "Sharpe": sharpe_ratio(stressed_net),
            "Max Drawdown": compute_drawdown(stressed_equity)
        }

    return results


# ============================================
# MAIN METRICS
# ============================================

def compute_metrics(df):

    returns = df["net_returns"].dropna()
    benchmark_returns = df["benchmark_returns"].dropna()

    total_return = df["equity_curve"].iloc[-1] - 1
    benchmark_total_return = df["benchmark_equity"].iloc[-1] - 1
    excess_total_return = total_return - benchmark_total_return

    sharpe = sharpe_ratio(returns)
    benchmark_sharpe = sharpe_ratio(benchmark_returns)

    max_dd = compute_drawdown(df["equity_curve"])
    benchmark_max_dd = compute_drawdown(df["benchmark_equity"])

    trade_count = (df["position"].diff().abs() > 0).sum()
    avg_turnover = df["turnover"].mean()
    avg_cost = df["costs"].mean()
    total_cost = df["costs"].sum()

    annual_ret = annualized_return(returns)
    annual_vol = annualized_volatility(returns)

    benchmark_annual_ret = annualized_return(benchmark_returns)
    benchmark_annual_vol = annualized_volatility(benchmark_returns)

    return {
        "Total Return": total_return,
        "Annual Return": annual_ret,
        "Annual Volatility": annual_vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Trades": trade_count,
        "Avg Turnover": avg_turnover,
        "Avg Cost": avg_cost,
        "Total Cost Drag": total_cost,
        "Benchmark Total Return": benchmark_total_return,
        "Benchmark Annual Return": benchmark_annual_ret,
        "Benchmark Annual Volatility": benchmark_annual_vol,
        "Benchmark Sharpe": benchmark_sharpe,
        "Benchmark Max Drawdown": benchmark_max_dd,
        "Excess Return vs Benchmark": excess_total_return
    }


# ============================================
# REPORTING
# ============================================

def print_metrics(metrics):

    print("\n--- BACKTEST RESULTS ---")
    ordered_keys = [
        "Total Return",
        "Annual Return",
        "Annual Volatility",
        "Sharpe",
        "Max Drawdown",
        "Trades",
        "Avg Turnover",
        "Avg Cost",
        "Total Cost Drag",
        "Benchmark Total Return",
        "Benchmark Annual Return",
        "Benchmark Annual Volatility",
        "Benchmark Sharpe",
        "Benchmark Max Drawdown",
        "Excess Return vs Benchmark"
    ]

    for k in ordered_keys:
        v = metrics.get(k, np.nan)
        if isinstance(v, (int, np.integer)):
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v:.4f}")


def print_regime_metrics(regime_metrics):

    print("\n--- REGIME BREAKDOWN ---")

    for regime, stats in regime_metrics.items():
        print(f"\n[{regime.upper()}]")
        for k, v in stats.items():
            if isinstance(v, (int, np.integer)):
                print(f"{k}: {v}")
            else:
                print(f"{k}: {v:.4f}")


def print_cost_stress_tests(stress_tests):

    print("\n--- COST STRESS TEST ---")

    for scenario, stats in stress_tests.items():
        print(f"\n[{scenario}]")
        for k, v in stats.items():
            print(f"{k}: {v:.4f}")


# ============================================
# FULL PIPELINE ENTRY POINT
# ============================================

def apply_backtest(df):

    print("\n--- RUNNING BACKTEST (STEP 3: VALIDATION + ROBUSTNESS) ---\n")

    df = run_backtest(df)
    df = add_validation_features(df)

    metrics = compute_metrics(df)
    regime_metrics = compute_regime_metrics(df)
    stress_tests = compute_cost_stress_tests(df)

    print_metrics(metrics)
    print_regime_metrics(regime_metrics)
    print_cost_stress_tests(stress_tests)

    return df, metrics