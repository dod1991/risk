import numpy as np

TARGET_VOL = 0.15
MIN_VOL = 0.01
MAX_LEVERAGE = 3.0
MAX_POSITION = 1.0


def apply_trade_engine(df):

    print("\n--- RUNNING TRADE ENGINE (PRE-SPLIT: FIX 7) ---\n")

    df["adj_vol"] = df["realised_vol_20"].clip(lower=MIN_VOL)
    df["adj_vol_smooth"] = df["adj_vol"].ewm(span=10).mean()

    df["vol_scaler"] = TARGET_VOL / df["adj_vol_smooth"]
    df["vol_scaler"] = df["vol_scaler"].clip(upper=MAX_LEVERAGE)

    # ✅ SINGLE EXPOSURE ONLY
    df["raw_position"] = df["exposure"].shift(1)

    signal_strength = df["quant_score_smooth"].abs()

    signal_boost = 0.5 + 1.5 * (signal_strength ** 2)
    signal_boost = np.clip(signal_boost, 0.5, 1.5)

    df["position"] = (
        df["raw_position"] *
        df["vol_scaler"] *
        signal_boost
    )

    df["position"] = df["position"].clip(-MAX_POSITION, MAX_POSITION)

    df["position"] = df["position"].ewm(span=10).mean()

    prev_position = df["position"].shift(1)

    df["position"] = np.where(
        (df["position"] * prev_position < 0),
        prev_position * 0.5,
        df["position"]
    )

    trend_strength = df["trend_strength"]

    regime_scaler = 0.8 + 0.6 * (trend_strength / 0.05)
    regime_scaler = np.clip(regime_scaler, 0.7, 1.3)

    df["position"] = df["position"] * regime_scaler

    prev_position = df["position"].shift(1)
    change = df["position"] - prev_position

    vol = df["realised_vol_20"]

    df["trade_threshold"] = np.where(vol > 0.02, 0.05, 0.10)

    df["position"] = np.where(
        change.abs() < df["trade_threshold"],
        prev_position,
        df["position"]
    )

    df["temp_returns"] = df["position"] * df["returns"]
    df["temp_equity"] = (1 + df["temp_returns"]).cumprod()

    df["temp_peak"] = df["temp_equity"].cummax()
    df["drawdown"] = (df["temp_equity"] - df["temp_peak"]) / df["temp_peak"]

    df["dd_scaler"] = np.where(
        df["drawdown"] < -0.10, 0.5,
        np.where(df["drawdown"] < -0.05, 0.75, 1.0)
    )

    df["position"] = df["position"] * df["dd_scaler"]

    return df