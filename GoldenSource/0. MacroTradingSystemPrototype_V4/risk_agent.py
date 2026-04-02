import numpy as np

# ============================================
# SHOCK (FAST RISK)
# ============================================

def compute_shock(row):

    vix = row["VIX"]
    ret = row["returns"]

    if vix > 25 or ret < -0.02:
        return -1
    elif vix < 15 and ret > 0:
        return +1
    else:
        return 0


# ============================================
# STRUCTURE (MARKET CONDITION)
# ============================================

def compute_structure(row):

    vol = row["realised_vol_20"]
    trend = row["trend"]

    if vol > 0.02 and trend < 0:
        return -1
    elif vol < 0.015 and trend > 0:
        return +1
    else:
        return 0


# ============================================
# REGIME (CONFIRMATION)
# ============================================

def compute_regime(row):

    vix = row["VIX"]
    vol = row["realised_vol_20"]

    if vix > 20 and vol > 0.02:
        return -1
    elif vix < 15 and vol < 0.015:
        return +1
    else:
        return 0


# ============================================
# COMBINE (CORE LOGIC)
# ============================================

def compute_risk_score(row):

    shock = compute_shock(row)
    structure = compute_structure(row)
    regime = compute_regime(row)

    # --- LAYERED LOGIC ---

    # Shock dominates direction
    if shock == -1:
        base = -1
    elif shock == +1:
        base = +1
    else:
        base = structure

    # Structure escalates risk
    if shock == -1 and structure == -1:
        base = -2

    # Regime confirms or dampens
    if regime == -1:
        base = min(base, 0)  # cap upside
    elif regime == +1:
        base = max(base, 0)  # allow upside

    return base


# ============================================
# APPLY TO DATAFRAME
# ============================================

def apply_risk_agent(df):

    df["shock"] = df.apply(compute_shock, axis=1)
    df["structure"] = df.apply(compute_structure, axis=1)
    df["regime"] = df.apply(compute_regime, axis=1)

    df["risk_score"] = df.apply(compute_risk_score, axis=1)

    return df
