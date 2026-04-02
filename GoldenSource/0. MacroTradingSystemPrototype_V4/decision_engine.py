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
        return 0.6
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
# SINGLE EXPOSURE (PRE-SPLIT)
# ============================================

def compute_exposure(row):

    risk = row["risk_score"]
    quant = row["quant_score_smooth"]
    macro = row["macro_scaler"]
    trend_strength = row["trend_strength"]

    cap = compute_risk_cap(risk)
    threshold = compute_threshold(risk)

    # Trend multiplier (same logic as before split)
    trend_multiplier = min(trend_strength / 0.03, 1.0)

    # Signal filter
    if abs(quant) < threshold:
        return 0

    # 🔥 SINGLE EXPOSURE (KEY)
    exposure = cap * quant * macro * trend_multiplier

    return exposure


# ============================================
# EXPLANATION
# ============================================

def generate_decision_reason(row):

    return (
        f"quant={round(row['quant_score_smooth'],2)}, "
        f"risk={row['risk_score']}, "
        f"trend_strength={round(row['trend_strength'],3)}, "
        f"exposure={round(row.get('exposure',0),2)}"
    )


# ============================================
# RUN ENGINE
# ============================================

def run_decision_engine(df):

    print("\n--- RUNNING DECISION ENGINE (PRE-SPLIT) ---\n")

    df["exposure"] = df.apply(compute_exposure, axis=1)

    df["decision_reason"] = df.apply(generate_decision_reason, axis=1)

    return df