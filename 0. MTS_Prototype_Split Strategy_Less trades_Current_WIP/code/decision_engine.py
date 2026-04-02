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
        return 0.7   # was 0.6
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
# EXPOSURE ENGINE
# ============================================

def compute_exposure(row):

    risk = row["risk_score"]
    quant = row["quant_score_smooth"]
    macro = row["macro_scaler"]
    regime = row["market_regime"]
    trend_strength = row["trend_strength"]

    cap = compute_risk_cap(risk)
    threshold = compute_threshold(risk)

    trend_multiplier = min(trend_strength / 0.03, 1.0)

    if abs(quant) < threshold:
        return 0, 0

    # Soft regime weights
    trend_weight = 1.0 if regime == "trend" else 0.5
    mr_weight = 1.0 if regime == "chop" else 0.5

    # Trend strategy
    exposure_trend = cap * quant * macro * trend_multiplier * trend_weight

    # Mean reversion strategy
    exposure_mr = cap * quant * macro * 0.5 * mr_weight

    return exposure_trend, exposure_mr


# ============================================
# EXPLANATION
# ============================================

def generate_decision_reason(row):

    return (
        f"quant={round(row['quant_score_smooth'],2)}, "
        f"risk={row['risk_score']}, "
        f"regime={row['market_regime']}, "
        f"trend_strength={round(row['trend_strength'],3)}, "
        f"exp_t={round(row.get('exposure_trend',0),2)}, "
        f"exp_mr={round(row.get('exposure_mr',0),2)}"
    )


# ============================================
# RUN ENGINE
# ============================================

def run_decision_engine(df):

    print("\n--- RUNNING DECISION ENGINE (OPTION 2: HIGHER NEUTRAL RISK CAP) ---\n")

    exposures = df.apply(compute_exposure, axis=1)

    df["exposure_trend"] = exposures.apply(lambda x: x[0])
    df["exposure_mr"] = exposures.apply(lambda x: x[1])

    # Compatibility column
    df["exposure"] = df["exposure_trend"] + df["exposure_mr"]

    df["decision_reason"] = df.apply(generate_decision_reason, axis=1)

    return df