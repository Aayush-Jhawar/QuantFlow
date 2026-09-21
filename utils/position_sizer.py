"""
Position sizing module for the QuantFlow trading system.

Implements entropy-based position sizing with TII (Trend Intensity Index)
factor modulation and sigmoid time-decay for session-aware risk management.
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


class PositionSizer:
    """
    Adaptive position sizer that combines:
      - Rolling Shannon entropy of log-returns (uncertainty measure)
      - Trend Intensity Index (TII) factor for trend-following amplification
      - Sigmoid time-decay to reduce exposure near session close

    Parameters
    ----------
    base_size : float
        Minimum position size floor.
    max_size : float
        Hard cap on position size.
    entropy_window : int
        Lookback window for rolling entropy calculation.
    entropy_bins : int
        Number of histogram bins for entropy estimation.
    tii_period : int
        Lookback period for TII calculation.
    tii_smoothing : int
        EWM span for smoothing the TII factor.
    sigmoid_beta : float
        Steepness of the sigmoid decay near session close.
    smoothing_window : int
        EWM span for final position size smoothing.
    """

    def __init__(
        self,
        base_size: float = 1.0,
        max_size: float = 100.0,
        entropy_window: int = 200,
        entropy_bins: int = 20,
        tii_period: int = 60,
        tii_smoothing: int = 50,
        sigmoid_beta: float = 0.001,
        smoothing_window: int = 100,
    ):
        self.base_size = base_size
        self.max_size = max_size
        self.entropy_window = entropy_window
        self.entropy_bins = entropy_bins
        self.tii_period = tii_period
        self.tii_smoothing = tii_smoothing
        self.sigmoid_beta = sigmoid_beta
        self.smoothing_window = smoothing_window

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_time_to_seconds(time_str: str) -> int:
        """Convert HH:MM:SS string to total seconds since midnight."""
        parts = str(time_str).split(":")
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

    def _calculate_rolling_entropy(self, price_series: pd.Series) -> pd.Series:
        """Compute rolling Shannon entropy of log-returns."""
        log_returns = np.log(price_series / price_series.shift(1)).fillna(0).values

        if len(log_returns) < self.entropy_window:
            return pd.Series(np.nan, index=price_series.index)

        windows = sliding_window_view(log_returns, window_shape=self.entropy_window)

        entropy_values = []
        for window in windows:
            counts, _ = np.histogram(window, bins=self.entropy_bins, density=False)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            ent = -np.sum(probs * np.log(probs))
            entropy_values.append(ent)

        pad_size = self.entropy_window - 1
        entropy_values = [np.nan] * pad_size + entropy_values

        return pd.Series(entropy_values, index=price_series.index)

    def _calculate_tii_factor(self, df: pd.DataFrame) -> pd.Series:
        """
        Trend Intensity Index factor.

        Maps the fraction of bars above the SMA to a [0.6, 1.3] multiplier,
        then smooths with an EWM.
        """
        sma = df["Price"].rolling(window=self.tii_period).mean()
        above_sma = (df["Price"] > sma).rolling(window=self.tii_period).sum()
        tii = 100 * (above_sma / self.tii_period)

        # Distance from 50 -> trend intensity (0-50 range)
        tii_distance = np.abs(tii - 50)
        tii_factor = 0.6 + (0.7 * (tii_distance / 50))
        tii_smooth = tii_factor.ewm(span=self.tii_smoothing, adjust=False).mean()
        return tii_smooth

    def _calculate_sigmoid_decay(self, time_series: pd.Series) -> np.ndarray:
        """
        Sigmoid multiplier that decays toward zero near the session close.

        The inflection point is placed at 85% of session duration.
        """
        seconds = np.array([self._parse_time_to_seconds(t) for t in time_series])
        open_sec = seconds[0]
        close_sec = seconds[-1]
        session_duration = close_sec - open_sec
        t_threshold = open_sec + (0.85 * session_duration)
        sigmoid_multiplier = 1.0 / (1.0 + np.exp(self.sigmoid_beta * (seconds - t_threshold)))
        return sigmoid_multiplier

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_position_sizing_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute position sizes for every row.

        Returns a copy of *df* with additional diagnostic columns:
          ``entropy_raw``, ``factor_entropy``, ``factor_tii``,
          ``factor_decay``, ``raw_position_size``, ``position_size``.
        """
        # Entropy factor -- high entropy => lower size
        entropy_s = self._calculate_rolling_entropy(df["Price"])
        entropy_smooth = entropy_s.ewm(span=self.smoothing_window * 4, adjust=False).mean()
        entropy_pct = entropy_smooth.expanding().rank(pct=True).fillna(0.5)
        entropy_factor = 1.0 - (0.7 * entropy_pct)

        # TII factor -- strong trend => larger size
        tii_factor = self._calculate_tii_factor(df).fillna(1.0)

        # Time decay -- reduce near close
        time_factor = self._calculate_sigmoid_decay(df["Time"])

        # Combine
        raw_size = self.max_size * entropy_factor * tii_factor * time_factor
        raw_size = np.maximum(raw_size, self.base_size * time_factor)

        # Final smoothing & clamping
        raw_size_smooth = pd.Series(raw_size).ewm(span=self.smoothing_window, adjust=False).mean()
        final_size = np.clip(raw_size_smooth, 0, self.max_size)
        final_size = pd.Series(final_size).fillna(self.base_size)
        final_size_int = np.round(final_size).astype(int)

        output_df = df.copy()
        output_df["entropy_raw"] = entropy_s
        output_df["factor_entropy"] = entropy_factor
        output_df["factor_tii"] = tii_factor
        output_df["factor_decay"] = time_factor
        output_df["raw_position_size"] = raw_size
        output_df["position_size"] = final_size_int

        return output_df
