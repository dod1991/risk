import numpy as np

# ============================================
# CONFIG
# ============================================

TARGET_VOL = 0.15
MIN_VOL = 0.01
MAX_LEVERAGE = 3.0
MAX_POSITION = 1.0


# ============================================
# TRADE ENGINE (BEST VERSION)
# ============================================

def apply_trade_engine(df):

    print("\n--- RUNNING TRADE ENGINE (BEST VERSION: STATIC PORTFOLIO) ---\n")

    # ========================================
    # VOL TARGETING
    # ========================================

    df["adj_vol"] = df["realised_vol_20"].clip(lower=MIN_VOL)
    df["adj_vol_smooth"] = df["adj_vol"].ewm(span=10).mean()

    df["vol_scaler"] = TARGET_VOL / df["adj_vol_smooth"]
    df["vol_scaler"] = df["vol_scaler"].clip(upper=MAX_LEVERAGE)

    # ========================================
    # BASE POSITIONS
    # ========================================

    df["raw_position_trend"] = df["exposure_trend"].shift(1)
    df["raw_position_mr"] = df["exposure_mr"].shift(1)

    # ========================================
    # SIGNAL SIZING
    # ========================================

    signal_strength = df["quant_score_smooth"].abs()

    signal_boost = 0.5 + 1.5 * (signal_strength ** 2)
    signal_boost = np.clip(signal_boost, 0.5, 1.5)

    # ========================================
    # POSITION FUNCTION
    # ========================================

    def compute_position(raw_position):
        pos = raw_position * df["vol_scaler"] * signal_boost
        return np.clip(pos, -MAX_POSITION, MAX_POSITION)

    df["position_trend"] = compute_position(df["raw_position_trend"])
    df["position_mr"] = compute_position(df["raw_position_mr"])

    # ========================================
    # SMOOTHING
    # ========================================

    df["position_trend"] = df["position_trend"].ewm(span=10).mean()
    df["position_mr"] = df["position_mr"].ewm(span=10).mean()

    # ========================================
    # ANTI-FLIP
    # ========================================

    for col in ["position_trend", "position_mr"]:
        prev = df[col].shift(1)
        df[col] = np.where(
            (df[col] * prev < 0),
            prev * 0.5,
            df[col]
        )

    # ========================================
    # STATIC PORTFOLIO COMBINATION (BEST)
    # ========================================

    df["position"] = (
        0.6 * df["position_trend"] +
        0.4 * df["position_mr"]
    )

    # ========================================
    # TRADE FILTER
    # ========================================

    prev_position = df["position"].shift(1)
    position_change = df["position"] - prev_position

    vol = df["realised_vol_20"]

    df["trade_threshold"] = np.where(
        vol > 0.02,
        0.05,
        0.10
    )

    df["position"] = np.where(
        position_change.abs() < df["trade_threshold"],
        prev_position,
        df["position"]
    )

    # ========================================
    # DRAWDOWN CONTROL
    # ========================================

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