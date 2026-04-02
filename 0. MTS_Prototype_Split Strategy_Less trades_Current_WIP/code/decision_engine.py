import numpy as np

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
# CRASH ALPHA DETECTOR
# ============================================

def compute_crash_score(row):

    vix = row["VIX"]
    ret = row["returns"]
    trend = row["trend"]
    vol = row["realised_vol_20"]
    trend_strength = row["trend_strength"]

    score = 0

    # Panic vol regime
    if vix >= 30:
        score += 2
    elif vix >= 25:
        score += 1

    # Sharp down day
    if ret <= -0.025:
        score += 2
    elif ret <= -0.015:
        score += 1

    # Negative market structure
    if trend < -0.02:
        score += 1

    # Elevated realized vol
    if vol >= 0.025:
        score += 1
    elif vol >= 0.02:
        score += 0.5

    # Strong downside trend gets extra confirmation
    if trend < 0 and trend_strength > 0.03:
        score += 0.5

    return score


def compute_crash_overlay(row):

    crash_score = row["crash_score"]
    regime = row["market_regime"]
    risk = row["risk_score"]

    # If risk engine has fully shut risk down, do nothing
    if risk <= -2:
        return 0.0

    # Scale overlay by severity
    if crash_score >= 4:
        base_overlay = -0.35
    elif crash_score >= 3:
        base_overlay = -0.20
    elif crash_score >= 2:
        base_overlay = -0.10
    else:
        base_overlay = 0.0

    # Slightly stronger in trend regime when the trend is already down,
    # but not too large in chop to avoid over-trading noise.
    if base_overlay != 0:
        regime_multiplier = 1.1 if regime == "trend" else 0.9
        return base_overlay * regime_multiplier

    return 0.0


# ============================================
# ASYMMETRIC EXPOSURE ENGINE + CRASH ALPHA
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

    # Trend multiplier from existing logic
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
    # FLAT BASE CASE
    # ========================================

    else:
        exposure_trend = 0.0
        exposure_mr = 0.0

    # ========================================
    # CRASH ALPHA OVERLAY
    # ========================================

    # Add overlay mainly to trend sleeve, because crashes are directional.
    # Keep it bounded so it doesn't dominate the whole system.
    exposure_trend += crash_overlay

    # Final clipping for safety
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
        f"crash_score={round(row['crash_score'], 2)}, "
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

    print("\n--- RUNNING DECISION ENGINE (STEP 7: CRASH ALPHA OVERLAY) ---\n")

    df["crash_score"] = df.apply(compute_crash_score, axis=1)
    df["crash_overlay"] = df.apply(compute_crash_overlay, axis=1)

    exposures = df.apply(compute_exposure, axis=1)

    df["exposure_trend"] = exposures.apply(lambda x: x[0])
    df["exposure_mr"] = exposures.apply(lambda x: x[1])

    # Compatibility column
    df["exposure"] = df["exposure_trend"] + df["exposure_mr"]

    df["decision_reason"] = df.apply(generate_decision_reason, axis=1)

    return df