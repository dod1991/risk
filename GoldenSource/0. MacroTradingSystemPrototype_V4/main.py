import os
import pandas as pd
from datetime import datetime

from data_pipeline import run_pipeline
from risk_agent import apply_risk_agent
from quant_agent import apply_quant_agent
from macro_agent import apply_macro_agent
from decision_engine import run_decision_engine
from trade_engine import apply_trade_engine
from backtest_engine import apply_backtest


def save_backtest_summary(output_path, results_dict):

    summary_df = pd.DataFrame({
        "Metric": [
            "Total Return",
            "Sharpe",
            "Max Drawdown",
            "Trades",
            "Avg Turnover"
        ],
        "Value": [
            results_dict["total_return"],
            results_dict["sharpe"],
            results_dict["max_drawdown"],
            results_dict["trades"],
            results_dict["avg_turnover"]
        ]
    })

    dir_name = os.path.dirname(output_path)
    file_name = os.path.basename(output_path)

    summary_file_name = file_name.replace("backtest_output", "backtest_summary")
    summary_path = os.path.join(dir_name, summary_file_name)

    summary_df.to_excel(summary_path, index=False)


if __name__ == "__main__":

    df = run_pipeline()

    df = apply_risk_agent(df)
    df = apply_quant_agent(df)
    df = apply_macro_agent(df)

    # ✅ SINGLE EXPOSURE
    df = run_decision_engine(df)

    df = apply_trade_engine(df)

    df, results = apply_backtest(df)

    total_return = results["total_return"]
    sharpe = results["sharpe"]
    max_drawdown = results["max_drawdown"]
    trades = results["trades"]
    avg_turnover = results["avg_turnover"]

    print("\n--- BACKTEST RESULTS ---")
    print(f"Total Return: {total_return:.4f}")
    print(f"Sharpe: {sharpe:.4f}")
    print(f"Max Drawdown: {max_drawdown:.4f}")
    print(f"Trades: {trades:.4f}")
    print(f"Avg Turnover: {avg_turnover:.4f}")

    today = datetime.today()
    output_dir = f"backtest_output/{today.year}/{today.month:02d}"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        f"backtest_output_{today.strftime('%Y-%m-%d')}.xlsx"
    )

    df.to_excel(output_path, index=False)

    save_backtest_summary(output_path, results)