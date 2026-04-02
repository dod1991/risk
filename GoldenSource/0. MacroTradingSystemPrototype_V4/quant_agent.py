import numpy as np

# ============================================
# NORMALISATION
# ============================================

def zscore(series, window=100):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / (std + 1e-6)


# ============================================
# FACTORS
# ============================================

def compute_quant_factors(df):

    # Existing signals
    df["mom_z"] = zscore(df["momentum_20"])
    df["trend_z"] = zscore(df["trend"])

    # ========================================
    # NEW SIGNALS (STEP 4)
    # ========================================

    # Mean reversion (short-term)
    df["reversal_5"] = -df["price"].pct_change(5)
    df["rev_z"] = zscore(df["reversal_5"])

    # Volatility breakout
    df["vol_breakout"] = df["returns"] / (df["realised_vol_20"] + 1e-6)
    df["vol_signal"] = zscore(df["vol_breakout"])

    return df


# ============================================
# SIGNALS
# ============================================

def compute_individual_signals(df):

    df["momentum_signal"] = df["mom_z"]
    df["trend_signal"] = df["trend_z"]

    return df


# ============================================
# REGIME-AWARE MULTI-FACTOR (FIXED STEP 4)
# ============================================

def compute_quant_score(df):

    momentum = df["momentum_signal"]
    trend = df["trend_signal"]
    rev = df["rev_z"]
    vol = df["vol_signal"]

    regime = df["market_regime"]
    trend_dir = df["trend_direction"]
    trend_strength = df["trend_strength"]

    # ========================================
    # BASE TREND COMPONENT (DOMINANT)
    # ========================================

    trend_component = 0.6 * momentum + 0.3 * trend

    # ========================================
    # REGIME-AWARE SIGNAL COMBINATION
    # ========================================

    raw_signal = trend_component.copy()

    # Mean reversion ONLY in chop regimes
    raw_signal += np.where(regime == "chop", 0.2 * rev, 0)

    # Vol breakout ONLY in strong trends
    raw_signal += np.where(trend_strength > 0.04, 0.1 * vol, 0)

    # Align with trend direction
    raw_signal = raw_signal * trend_dir

    # ========================================
    # FINAL SIGNAL
    # ========================================

    df["quant_raw"] = raw_signal
    df["quant_score"] = np.tanh(df["quant_raw"])

    # ========================================
    # SIGNAL PERSISTENCE (STEP 3)
    # ========================================

    df["quant_score_smooth"] = df["quant_score"].rolling(3).mean()

    return df


# ============================================
# CONFIDENCE
# ============================================

def compute_confidence(df):

    df["quant_confidence"] = df["quant_score_smooth"].abs()

    return df


# ============================================
# MAIN
# ============================================

def apply_quant_agent(df):

    print("\n--- RUNNING QUANT AGENT (STEP 4 FIXED: REGIME-AWARE ALPHA) ---\n")

    df = compute_quant_factors(df)
    df = compute_individual_signals(df)
    df = compute_quant_score(df)
    df = compute_confidence(df)

    return df