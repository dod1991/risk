import numpy as np
import pandas as pd

# ============================================
# RISK CAP
# ============================================

def compute_risk_cap(risk_score):

    if risk_score <= -2:
        return 0.0
    elif risk_score == -1:
        return 0.3
    elif risk_score == 0:
        return 0.7
    else:
        return 1.0


# ============================================
# THRESHOLD
# ============================================

def compute_threshold(risk_score):

    if risk_score <= -1:
        return 0.5
    elif risk_score == 0:
        return 0.4
    else:
        return 0.3


# ============================================
# CRASH EVENT DETECTION (RARE / STRICT)
# ============================================

def compute_crash_event_flag(row):

    vix = row["VIX"]
    ret = row["returns"]
    trend = row["trend"]
    vol = row["realised_vol_20"]

    # True crisis-style trigger:
    # elevated fear + sharp selloff + weak structure
    hard_crash = (
        (vix >= 35) and
        (ret <= -0.03)
    )

    structural_break = (
        (vix >= 30) and
        (ret <= -0.02) and
        (trend < -0.02) and
        (vol >= 0.025)
    )

    return int(hard_crash or structural_break)


def build_crash_state_series(df, crash_duration_days=5):

    crash_state = np.zeros(len(df), dtype=int)
    countdown = 0

    for i in range(len(df)):
        event_flag = int(df["crash_event_flag"].iloc[i])

        if event_flag == 1:
            countdown = crash_duration_days

        if countdown > 0:
            crash_state[i] = 1
            countdown -= 1

    return pd.Series(crash_state, index=df.index)


# ============================================
# CRASH OVERLAY (BOUNDED / TEMPORARY)
# ============================================

def compute_crash_overlay(row):

    if row["crash_state"] == 0:
        return 0.0

    risk = row["risk_score"]
    regime = row["market_regime"]
    vix = row["VIX"]
    ret = row["returns"]

    # If the core risk engine has fully shut down, do nothing
    if risk <= -2:
        return 0.0

    # Severity
    if vix >= 40 and ret <= -0.035:
        overlay = -0.30
    elif vix >= 35 and ret <= -0.03:
        overlay = -0.22
    else:
        overlay = -0.15

    # Keep crash alpha more directional in trend regimes,
    # lighter in chop so we don't destroy MR behavior.
    regime_multiplier = 1.0 if regime == "trend" else 0.7

    return overlay * regime_multiplier


# ============================================
# ASYMMETRIC EXPOSURE ENGINE
# ============================================

def compute_exposure(row):

    risk = row["risk_score"]
    quant = row["quant_score_smooth"]
    macro = row["macro_scaler"]
    regime = row["market_regime"]
    trend_strength = row["trend_strength"]
    crash_overlay = row["crash_overlay"]

    cap = compute_risk_cap(risk)
    threshold = compute_threshold(risk)

    trend_multiplier = min(trend_strength / 0.03, 1.0)

    # ========================================
    # ASYMMETRIC LONG / SHORT FILTER
    # ========================================

    long_threshold = threshold
    short_threshold = threshold + 0.15

    if quant > long_threshold:
        direction = "long"
    elif quant < -short_threshold:
        direction = "short"
    else:
        direction = "flat"

    # ========================================
    # BASE REGIME WEIGHTS
    # ========================================

    trend_weight = 1.0 if regime == "trend" else 0.5
    mr_weight = 1.0 if regime == "chop" else 0.5

    # ========================================
    # LONG SIDE
    # ========================================

    if direction == "long":

        exposure_trend = cap * quant * macro * trend_multiplier * trend_weight
        exposure_mr = cap * quant * macro * 0.5 * mr_weight

    # ========================================
    # SHORT SIDE (ASYMMETRIC BOOST)
    # ========================================

    elif direction == "short":

        short_boost = 1.25
        regime_adjust = 1.2 if regime == "chop" else 0.8

        exposure_trend = (
            cap
            * quant
            * macro
            * trend_multiplier
            * trend_weight
            * short_boost
            * regime_adjust
        )

        exposure_mr = (
            cap
            * quant
            * macro
            * 0.5
            * mr_weight
            * short_boost
            * regime_adjust
        )

    # ========================================
    # FLAT
    # ========================================

    else:
        exposure_trend = 0.0
        exposure_mr = 0.0

    # ========================================
    # ADVANCED CRASH ALPHA OVERLAY
    # ========================================

    # Apply only to trend sleeve, because crash behavior is directional.
    # Keep bounded so it complements the strategy rather than dominates it.
    exposure_trend += crash_overlay

    exposure_trend = np.clip(exposure_trend, -1.0, 1.0)
    exposure_mr = np.clip(exposure_mr, -1.0, 1.0)

    return exposure_trend, exposure_mr


# ============================================
# EXPLANATION
# ============================================

def generate_decision_reason(row):

    quant = row["quant_score_smooth"]
    threshold = compute_threshold(row["risk_score"])
    short_threshold = threshold + 0.15

    if quant > threshold:
        side = "LONG"
    elif quant < -short_threshold:
        side = "SHORT"
    else:
        side = "FLAT"

    return (
        f"side={side}, "
        f"quant={round(row['quant_score_smooth'], 2)}, "
        f"risk={row['risk_score']}, "
        f"regime={row['market_regime']}, "
        f"trend_strength={round(row['trend_strength'], 3)}, "
        f"crash_event={int(row['crash_event_flag'])}, "
        f"crash_state={int(row['crash_state'])}, "
        f"crash_overlay={round(row['crash_overlay'], 2)}, "
        f"thr_long={round(threshold, 2)}, "
        f"thr_short={round(short_threshold, 2)}, "
        f"exp_t={round(row.get('exposure_trend', 0), 2)}, "
        f"exp_mr={round(row.get('exposure_mr', 0), 2)}"
    )


# ============================================
# RUN ENGINE
# ============================================

def run_decision_engine(df):

    print("\n--- RUNNING DECISION ENGINE (STEP 7 ADVANCED: EVENT-DRIVEN CRASH ALPHA) ---\n")

    df["crash_event_flag"] = df.apply(compute_crash_event_flag, axis=1)
    df["crash_state"] = build_crash_state_series(df, crash_duration_days=5)
    df["crash_overlay"] = df.apply(compute_crash_overlay, axis=1)

    exposures = df.apply(compute_exposure, axis=1)

    df["exposure_trend"] = exposures.apply(lambda x: x[0])
    df["exposure_mr"] = exposures.apply(lambda x: x[1])

    # Compatibility column
    df["exposure"] = df["exposure_trend"] + df["exposure_mr"]

    df["decision_reason"] = df.apply(generate_decision_reason, axis=1)

    return df