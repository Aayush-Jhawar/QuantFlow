"""
Technical indicators module for the QuantFlow trading system.

Contains SuperTrend (multi-timeframe), VPIN (Volume-Synchronized Probability
of Informed Trading), ADX, and Kalman filter implementations.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


# ======================================================================
# SuperTrend variants
# ======================================================================

def calc_supertrend(df, i, bool_var_prev, period=230, atr_multiplier=4, window=8,
                    high_col="BB11_T8", low_col="BB12_T8"):
    """
    Standard SuperTrend calculation on a given timeframe.

    Parameters
    ----------
    df : pd.DataFrame
        Price dataframe (modified in-place with band columns).
    i : int
        Current bar index.
    bool_var_prev : bool
        Previous uptrend state.
    period : int
        ATR lookback period.
    atr_multiplier : float
        Band width multiplier.
    window : int
        Identifier tag for column names.
    high_col, low_col : str
        Column names for high/low prices at this timeframe.

    Returns
    -------
    bool
        Current uptrend state.
    """
    for col in [f"Upperband{window}", f"Lowerband{window}",
                f"H-L{window}", f"H-PC{window}", f"L-PC{window}",
                f"TR{window}", f"ATR{window}"]:
        if col not in df.columns:
            df[col] = 0

    df.loc[i, f"H-L{window}"] = df[high_col].iloc[i] - df[low_col].iloc[i]
    df.loc[i, f"H-PC{window}"] = abs(df[high_col].iloc[i] - df["Price"].iloc[i])
    df.loc[i, f"L-PC{window}"] = abs(df[low_col].iloc[i] - df["Price"].iloc[i - 1])
    df.loc[i, f"TR{window}"] = df[[f"H-L{window}", f"H-PC{window}", f"L-PC{window}"]].iloc[i].max()
    df.loc[i, f"ATR{window}"] = (
        df[f"TR{window}"][max(0, i - period + 1) : i + 1].mean()
        if (i + 1 - period >= 0)
        else np.nan
    )

    mid = (df[high_col].iloc[i] + df[low_col].iloc[i]) / 2
    df.loc[i, f"Upperband{window}"] = mid + atr_multiplier * df[f"ATR{window}"].iloc[i]
    df.loc[i, f"Lowerband{window}"] = mid - atr_multiplier * df[f"ATR{window}"].iloc[i]

    current_state = bool_var_prev
    if df["Price"].iloc[i] > df[f"Upperband{window}"].iloc[i - 1]:
        current_state = True
    elif df["Price"].iloc[i] < df[f"Lowerband{window}"].iloc[i - 1]:
        current_state = False
    else:
        if current_state and df[f"Lowerband{window}"].iloc[i] < df[f"Lowerband{window}"].iloc[i - 1]:
            df.loc[df.index[i], f"Lowerband{window}"] = df[f"Lowerband{window}"].iloc[i - 1]
        if not current_state and df[f"Upperband{window}"].iloc[i] > df[f"Upperband{window}"].iloc[i - 1]:
            df.loc[df.index[i], f"Upperband{window}"] = df[f"Upperband{window}"].iloc[i - 1]
    return current_state


def calc_supertrend_long(df, i, bool_var_prev):
    """SuperTrend with long-period parameters (period=1620, mult=2.77, window=12)."""
    return calc_supertrend(
        df, i, bool_var_prev,
        period=1620, atr_multiplier=2.77, window=12,
        high_col="BB11_T12", low_col="BB12_T12",
    )


def calc_supertrend_short(df, i, bool_var_prev):
    """SuperTrend with short-period parameters (period=15, mult=2.7, window=5)."""
    return calc_supertrend(
        df, i, bool_var_prev,
        period=15, atr_multiplier=2.7, window=5,
        high_col="BB11_T5", low_col="BB12_T5",
    )


def calc_supertrend_composite(df, i, bool_var_prev):
    """
    Composite SuperTrend using mixed high/low from different timeframes.

    Uses BB11_T3 as high and BB12_T5 as low with EWM-based ATR (period=7, mult=2).
    """
    HIGH = "BB11_T3"
    LOW = "BB12_T5"
    window = "new"
    period = 7
    atr_multiplier = 2

    if "Upperband_super" not in df.columns:
        df["Upperband_super"] = np.nan
    if "Lowerband_super" not in df.columns:
        df["Lowerband_super"] = np.nan

    for col in [f"H-L{window}", f"H-PC{window}", f"L-PC{window}",
                f"TR{window}", f"ATR{window}"]:
        if col not in df.columns:
            df[col] = 0

    df.loc[i, f"H-L{window}"] = df[HIGH].iloc[i] - df[LOW].iloc[i]
    df.loc[i, f"H-PC{window}"] = abs(df[HIGH].iloc[i] - df["Price"].iloc[i - 1])
    df.loc[i, f"L-PC{window}"] = abs(df[LOW].iloc[i] - df["Price"].iloc[i - 1])
    df.loc[i, f"TR{window}"] = df[[f"H-L{window}", f"H-PC{window}", f"L-PC{window}"]].iloc[i].max()
    df[f"ATR{window}"] = df[f"TR{window}"].ewm(alpha=1 / period, adjust=False).mean()

    mid = (df[HIGH].iloc[i] + df[LOW].iloc[i]) / 2
    df["Upperband_super"].iloc[i] = mid + atr_multiplier * df[f"ATR{window}"].iloc[i]
    df["Lowerband_super"].iloc[i] = mid - atr_multiplier * df[f"ATR{window}"].iloc[i]

    current_state = bool_var_prev
    if df["Price"].iloc[i] > df["Upperband_super"].iloc[i - 1]:
        current_state = True
    elif df["Price"].iloc[i] < df["Lowerband_super"].iloc[i - 1]:
        current_state = False
    else:
        if current_state and df["Lowerband_super"].iloc[i] < df["Lowerband_super"].iloc[i - 1]:
            df.loc[df.index[i], "Lowerband_super"] = df["Lowerband_super"].iloc[i - 1]
        if not current_state and df["Upperband_super"].iloc[i] > df["Upperband_super"].iloc[i - 1]:
            df.loc[df.index[i], "Upperband_super"] = df["Upperband_super"].iloc[i - 1]
    return current_state


# ======================================================================
# VPIN (Volume-Synchronized Probability of Informed Trading)
# ======================================================================

def vpin_fractional_ema(prices, volumes, V, n, buckets, sigma_dp, sigma_window=600):
    """
    Compute VPIN using fractional volume buckets and EMA smoothing.

    Parameters
    ----------
    prices, volumes : array-like
        Raw tick prices and volumes.
    V : float
        Target volume per bucket.
    n : int
        Number of buckets for EMA window.
    buckets : list of (prices, volumes) tuples
        Completed volume buckets.
    sigma_dp : array-like
        Rolling standard deviation of price changes.
    sigma_window : int
        Window for sigma calculation (unused here, kept for API compat).

    Returns
    -------
    float
        Current VPIN estimate.
    """
    prices = np.asarray(prices)
    volumes = np.asarray(volumes)

    OI_list = []
    for p_b, v_b in buckets:
        if len(p_b) < 2:
            OI_list.append(0.0)
            continue

        dP = np.diff(p_b)
        end_idx = len(p_b) - 1
        sigma_local = sigma_dp[end_idx]

        if np.isnan(sigma_local) or sigma_local <= 0:
            sigma_local = 1e-8

        Z = norm.cdf(dP / sigma_local)
        v_eff = v_b[1:]

        VB = np.sum(v_eff * Z)
        VS = np.sum(v_eff) - VB

        OI_list.append(abs(VB - VS))

    alpha = 2 / (n + 1)
    vpin = np.full(len(OI_list), np.nan)

    ema = None
    for i, oi in enumerate(OI_list):
        if ema is None:
            ema = oi
        else:
            ema = alpha * oi + (1 - alpha) * ema
        vpin[i] = ema / (n * V)

    return vpin[-1]


def get_zvpin(vpin_values, W):
    """
    Compute Z-VPIN: normalised VPIN in [0, 1] using quantile scaling.

    Parameters
    ----------
    vpin_values : array-like
        Historical VPIN values.
    W : int
        Lookback window for quantile computation.

    Returns
    -------
    float
        Z-VPIN value clipped to [0, 1].
    """
    lo, hi = np.quantile(vpin_values[max(0, len(vpin_values) - W) :], [0.05, 0.95])
    out = (vpin_values[-1] - lo) / (hi - lo + 1e-12)
    return float(np.clip(out, 0, 1))
