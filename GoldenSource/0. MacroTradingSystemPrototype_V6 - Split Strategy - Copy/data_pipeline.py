import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(BASE_DIR, "marketData_transformed")
RAW_CACHE_PATH = os.path.join(BASE_PATH, "raw_cache")

START_DATE = "2015-01-01"

ASSETS = {
    "SPY": "price",
    "^VIX": "VIX",
    "^TNX": "TNX"
}

def create_output_path():
    today = datetime.today()
    year = str(today.year)
    month = f"{today.month:02d}"
    date_str = today.strftime("%Y-%m-%d")

    folder_path = os.path.join(BASE_PATH, year, month)
    os.makedirs(folder_path, exist_ok=True)

    file_path = os.path.join(folder_path, f"market_data_{date_str}.csv")
    return file_path


def download_data():
    os.makedirs(RAW_CACHE_PATH, exist_ok=True)
    data = {}

    for ticker, name in ASSETS.items():

        cache_file = os.path.join(RAW_CACHE_PATH, f"{name}.csv")

        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        else:
            df = yf.download(ticker, start=START_DATE)

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            series = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
            series.to_csv(cache_file)
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)

        data[name] = df.iloc[:, 0]

    return pd.DataFrame(data)


def clean_data(df):
    df = df.sort_index()
    df = df.ffill()
    df = df.dropna()
    return df


def build_features(df):

    df["returns"] = df["price"].pct_change()
    df["momentum_20"] = df["price"].pct_change(20)
    df["realised_vol_20"] = df["returns"].rolling(20).std()

    df["ma_50"] = df["price"].rolling(50).mean()
    df["trend"] = (df["price"] - df["ma_50"]) / df["ma_50"]

    df["vol_adj_mom"] = df["momentum_20"] / (df["realised_vol_20"] + 1e-6)
    df["yield_change"] = df["TNX"].diff()

    # ============================================
    # 🔥 STRONG REGIME FEATURES
    # ============================================

    df["trend_strength"] = df["trend"].abs()

    df["market_regime"] = np.where(
        df["trend_strength"] > 0.035,
        "trend",
        "chop"
    )

    df["trend_direction"] = np.sign(df["trend"])

    return df


def validate_data(df):

    if df.isna().sum().sum() > 0:
        print("WARNING: Missing values detected")

    if not df.index.is_monotonic_increasing:
        raise Exception("Index not sorted")

    print("Data validation passed")


def run_pipeline():

    print("\n--- RUNNING DATA PIPELINE ---\n")

    df = download_data()
    df = clean_data(df)
    df = build_features(df)
    df = df.dropna()

    validate_data(df)

    output_path = create_output_path()
    df.to_csv(output_path)

    return df


if __name__ == "__main__":
    run_pipeline()