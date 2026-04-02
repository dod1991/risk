import numpy as np
import pandas as pd

# ============================================
# CONFIG
# ============================================

TARGET_VOL = 0.18
MIN_VOL = 0.01
MAX_LEVERAGE = 3.0
MAX_POSITION = 1.0

# ============================================
# STEP 1 LOCKED-IN TRADE CONTROL
# ============================================

MIN_HOLD_DAYS = 2
HIGH_VOL_THRESHOLD = 0.02
TRADE_THRESHOLD_HIGH_VOL = 0.07
TRADE_THRESHOLD_LOW_VOL = 0.12

# ============================================
# STEP 2 + OPTION 2 + STEP 5:
# REGIME-BASED PORTFOLIO WEIGHTS
# ============================================

TREND_REGIME_TREND_WEIGHT = 0.85
TREND_REGIME_MR_WEIGHT = 0.15

CHOP_REGIME_TREND_WEIGHT = 0.20
CHOP_REGIME_MR_WEIGHT = 0.80


# ============================================
# TRADE ENGINE (STEP 5: STRONGER CHOP MR TILT)
# ============================================

def apply_trade_engine(df):

    print("\n--- RUNNING TRADE ENGINE (STEP 5: STRONGER CHOP MR TILT) ---\n")

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
    # REGIME-BASED PORTFOLIO COMBINATION
    # ========================================

    df["trend_weight"] = np.where(
        df["market_regime"] == "trend",
        TREND_REGIME_TREND_WEIGHT,
        CHOP_REGIME_TREND_WEIGHT
    )

    df["mr_weight"] = np.where(
        df["market_regime"] == "trend",
        TREND_REGIME_MR_WEIGHT,
        CHOP_REGIME_MR_WEIGHT
    )

    df["position"] = (
        df["trend_weight"] * df["position_trend"] +
        df["mr_weight"] * df["position_mr"]
    )

    # ========================================
    # TRADE FILTER + MIN HOLD
    # ========================================

    vol = df["realised_vol_20"]

    df["trade_threshold"] = np.where(
        vol > HIGH_VOL_THRESHOLD,
        TRADE_THRESHOLD_HIGH_VOL,
        TRADE_THRESHOLD_LOW_VOL
    )

    filtered_position = []
    trade_flag = []
    days_since_trade_list = []

    days_since_trade = MIN_HOLD_DAYS

    for i in range(len(df)):
        target = df["position"].iloc[i]

        if i == 0 or pd.isna(target):
            initial_position = 0.0 if pd.isna(target) else target
            filtered_position.append(initial_position)
            trade_flag.append(1)
            days_since_trade = 0
            days_since_trade_list.append(days_since_trade)
            continue

        prev_live_position = filtered_position[-1]
        threshold = df["trade_threshold"].iloc[i]

        if pd.isna(threshold):
            filtered_position.append(prev_live_position)
            trade_flag.append(0)
            days_since_trade += 1
            days_since_trade_list.append(days_since_trade)
            continue

        change = abs(target - prev_live_position)

        should_trade = (
            not pd.isna(target)
            and change >= threshold
            and days_since_trade >= MIN_HOLD_DAYS
        )

        if should_trade:
            filtered_position.append(target)
            trade_flag.append(1)
            days_since_trade = 0
        else:
            filtered_position.append(prev_live_position)
            trade_flag.append(0)
            days_since_trade += 1

        days_since_trade_list.append(days_since_trade)

    df["position_target"] = df["position"]
    df["position"] = filtered_position
    df["trade_executed"] = trade_flag
    df["days_since_trade"] = days_since_trade_list

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