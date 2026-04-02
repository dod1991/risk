import numpy as np

# ============================================
# NORMALISATION
# ============================================

def zscore(series, window=252):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / (std + 1e-6)


# ============================================
# MACRO SIGNALS (ADAPTIVE)
# ============================================

def compute_rate_signal(df):

    # Z-score of yields (relative positioning)
    df["tnx_z"] = zscore(df["TNX"], 252)

    # Rate level signal (relative)
    df["rate_level_signal"] = np.where(
        df["tnx_z"] > 1, -1,
        np.where(df["tnx_z"] < -1, +1, 0)
    )

    # Rate trend signal (directional)
    df["rate_trend_signal"] = np.sign(df["yield_change"])

    return df


# ============================================
# COMBINE MACRO SIGNALS
# ============================================

def compute_macro_score(df):

    score = df["rate_level_signal"] + df["rate_trend_signal"]

    # squash into [-1, 0, +1]
    df["macro_score"] = np.where(
        score > 0, +1,
        np.where(score < 0, -1, 0)
    )

    return df


# ============================================
# MACRO SCALER (FIXED)
# ============================================

def compute_macro_scaler(df):

    # ========================================
    # 🔥 CONDITIONAL MACRO (FIX)
    # ========================================

    df["macro_scaler"] = np.where(
        (df["macro_score"] == -1) & (df["trend_strength"] < 0.03),
        0.7,   # only reduce exposure in weak + risk-off environments
        1.0    # otherwise no interference
    )

    return df


# ============================================
# APPLY TO DATAFRAME
# ============================================

def apply_macro_agent(df):

    print("\n--- RUNNING MACRO AGENT (STEP 5 FIXED: CONDITIONAL MACRO) ---\n")

    df = compute_rate_signal(df)
    df = compute_macro_score(df)
    df = compute_macro_scaler(df)

    return df