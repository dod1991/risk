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
# ASYMMETRIC EXPOSURE ENGINE
# ============================================

def compute_exposure(row):

    risk = row["risk_score"]
    quant = row["quant_score_smooth"]
    macro = row["macro_scaler"]
    regime = row["market_regime"]
    trend_strength = row["trend_strength"]

    cap = compute_risk_cap(risk)
    threshold = compute_threshold(risk)

    # Trend multiplier from existing logic
    trend_multiplier = min(trend_strength / 0.03, 1.0)

    # ========================================
    # ASYMMETRIC LONG / SHORT FILTER
    # ========================================

    # Long side unchanged
    long_threshold = threshold

    # Shorts require stronger confirmation
    short_threshold = threshold + 0.15

    if quant > long_threshold:
        direction = "long"
    elif quant < -short_threshold:
        direction = "short"
    else:
        return 0, 0

    # ========================================
    # BASE REGIME WEIGHTS
    # ========================================

    trend_weight = 1.0 if regime == "trend" else 0.5
    mr_weight = 1.0 if regime == "chop" else 0.5

    # ========================================
    # LONG SIDE (UNCHANGED)
    # ========================================

    if direction == "long":

        exposure_trend = cap * quant * macro * trend_multiplier * trend_weight
        exposure_mr = cap * quant * macro * 0.5 * mr_weight

    # ========================================
    # SHORT SIDE (ASYMMETRIC BOOST)
    # ========================================

    else:

        # Slight short boost so shorts are meaningful
        short_boost = 1.25

        # Prefer shorts more in chop, less in trend
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
        f"thr_long={round(threshold, 2)}, "
        f"thr_short={round(short_threshold, 2)}, "
        f"exp_t={round(row.get('exposure_trend', 0), 2)}, "
        f"exp_mr={round(row.get('exposure_mr', 0), 2)}"
    )


# ============================================
# RUN ENGINE
# ============================================

def run_decision_engine(df):

    print("\n--- RUNNING DECISION ENGINE (STEP 6: ASYMMETRIC SHORT LOGIC) ---\n")

    exposures = df.apply(compute_exposure, axis=1)

    df["exposure_trend"] = exposures.apply(lambda x: x[0])
    df["exposure_mr"] = exposures.apply(lambda x: x[1])

    # Compatibility column
    df["exposure"] = df["exposure_trend"] + df["exposure_mr"]

    df["decision_reason"] = df.apply(generate_decision_reason, axis=1)

    return df