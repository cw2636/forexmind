"""
ForexMind — Feature Engineering Pipeline
==========================================
Transforms raw OHLCV + indicator-augmented DataFrames into ML-ready feature matrices.

Features created:
  - Lag features (past N bars of each indicator)
  - Rolling statistics (mean, std, skew over windows)
  - Session flags (London, NY, Tokyo, overlap)
  - Candle pattern binary flags
  - Rate-of-change normalised indicators
  - Target label: next-bar direction (for supervised training)

Advanced Python concepts:
  - Pipeline pattern (list of transform functions applied sequentially)
  - functools.reduce for composing transforms
  - numpy vectorisation over pandas loops
  - Type aliases
"""

from __future__ import annotations

from functools import reduce
from typing import Callable

import numpy as np
import pandas as pd

from forexmind.utils.session_times import get_session_status
from forexmind.utils.logger import get_logger

log = get_logger(__name__)

# Type alias for a transform function: DataFrame → DataFrame
Transform = Callable[[pd.DataFrame], pd.DataFrame]


# ── Individual transform functions ───────────────────────────────────────────

def add_lag_features(df: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
    """
    Add lagged versions of key indicators.
    E.g. rsi_lag1, rsi_lag2, ... rsi_lag5
    """
    lag_cols = ["rsi", "macd", "macd_hist", "adx", "atr_pct", "bb_pct", "stoch_k", "cci"]
    for col in lag_cols:
        if col in df.columns:
            for lag in range(1, lags + 1):
                df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_rolling_stats(
    df: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Rolling mean and std of close price over multiple windows."""
    windows = windows or [5, 10, 20]
    for w in windows:
        df[f"close_roll_mean_{w}"] = df["close"].rolling(w).mean()
        df[f"close_roll_std_{w}"] = df["close"].rolling(w).std()
        df[f"high_roll_max_{w}"] = df["high"].rolling(w).max()
        df[f"low_roll_min_{w}"] = df["low"].rolling(w).min()
    return df


def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Candle pattern binary flags for ML input.
    These are simpler versions of candlestick pattern recognition.
    """
    body = (df["close"] - df["open"]).abs()
    full_range = df["high"] - df["low"]
    upper_wick = df["high"] - df[["close", "open"]].max(axis=1)
    lower_wick = df[["close", "open"]].min(axis=1) - df["low"]

    df["candle_body_pct"] = body / full_range.replace(0, np.nan)       # Body as % of full range
    df["candle_is_bullish"] = (df["close"] > df["open"]).astype(int)
    df["candle_is_doji"] = (df["candle_body_pct"] < 0.1).astype(int)
    df["candle_upper_wick_pct"] = upper_wick / full_range.replace(0, np.nan)
    df["candle_lower_wick_pct"] = lower_wick / full_range.replace(0, np.nan)
    # Engulfing: current body is larger than previous and opposite colour
    prev_body = body.shift(1)
    prev_bullish = df["candle_is_bullish"].shift(1)
    df["candle_bullish_engulf"] = (
        (df["candle_is_bullish"] == 1)
        & (prev_bullish == 0)
        & (body > prev_body)
    ).astype(int)
    df["candle_bearish_engulf"] = (
        (df["candle_is_bullish"] == 0)
        & (prev_bullish == 1)
        & (body > prev_body)
    ).astype(int)
    return df


def add_session_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary session flags as features.
    These help the ML model learn that patterns differ by session.
    """
    def _flags(ts: pd.Timestamp) -> dict[str, int]:
        status = get_session_status(ts.to_pydatetime())
        return {
            "session_london": int("London" in status.active_sessions),
            "session_ny": int("New York" in status.active_sessions),
            "session_tokyo": int("Tokyo" in status.active_sessions),
            "session_overlap": int(status.is_overlap),
            "session_score": status.session_score,
        }

    flags_df = pd.DataFrame(
        [_flags(ts) for ts in df.index],
        index=df.index,
    )
    return pd.concat([df, flags_df], axis=1)


def add_normalised_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise price-level indicators to be scale-independent."""
    if "atr" in df.columns and "close" in df.columns:
        df["norm_atr"] = df["atr"] / df["close"]
    if "bb_upper" in df.columns and "bb_lower" in df.columns:
        mid = (df["bb_upper"] + df["bb_lower"]) / 2
        width = df["bb_upper"] - df["bb_lower"]
        df["norm_bb_width"] = width / mid.replace(0, np.nan)
    if "ema_21" in df.columns and "ema_50" in df.columns:
        df["ema_cross_dist"] = (df["ema_21"] - df["ema_50"]) / df["close"].replace(0, np.nan)
    return df


def add_target_label(df: pd.DataFrame, forward_bars: int = 12, threshold_pct: float = 0.05) -> pd.DataFrame:
    """
    Add a classification target: did price go UP, DOWN, or stay FLAT
    over the next `forward_bars` bars?

    target: 1 = UP (bullish), -1 = DOWN (bearish), 0 = flat
    Used for training supervised ML models.

    Defaults: 12 M5 bars = 1 hour ahead; 0.05% threshold ≈ 5 pips (EUR/USD).
    These give a much cleaner signal than 1-bar-ahead on M5.
    threshold_pct: Minimum % move to count as directional (filters noise).
    """
    future_return = df["close"].shift(-forward_bars) / df["close"] - 1.0
    df["target"] = np.where(
        future_return > threshold_pct / 100.0, 1,
        np.where(future_return < -threshold_pct / 100.0, -1, 0)
    )
    df["future_return"] = future_return
    return df


# ── Pipeline orchestrator ─────────────────────────────────────────────────────

def build_feature_matrix(
    df: pd.DataFrame,
    add_target: bool = True,
    lags: int = 5,
    include_sessions: bool = True,
) -> pd.DataFrame:
    """
    Apply the full feature engineering pipeline to a DataFrame.

    Args:
        df: OHLCV + indicator-augmented DataFrame (output of IndicatorEngine.compute())
        add_target: Whether to add the target label column (for training)
        lags: Number of lag features per indicator
        include_sessions: Whether to add Forex session flags

    Returns:
        Feature-rich DataFrame ready for ML training/inference.
    """
    transforms: list[Transform] = [
        lambda d: add_lag_features(d, lags=lags),
        add_rolling_stats,
        add_candle_features,
        add_normalised_indicators,
    ]
    if include_sessions:
        transforms.append(add_session_flags)
    if add_target:
        transforms.append(add_target_label)

    # Apply all transforms in sequence using functools.reduce
    result = reduce(lambda d, fn: fn(d), transforms, df.copy())

    # Drop rows with NaN (from lags and rolling windows)
    initial_len = len(result)
    result = result.dropna(subset=["rsi", "macd", "adx"]).copy()
    log.debug(f"Feature engineering: {initial_len} → {len(result)} rows after dropna")
    return result


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the list of feature column names to pass to the ML model.
    Excludes raw OHLCV columns, target, and metadata columns.
    """
    exclude = {
        "open", "high", "low", "close", "volume",
        "target", "future_return",
        # Raw price-level pivot columns (not normalised)
        "pivot", "r1", "r2", "s1", "s2",
        "psar_long", "psar_short",
        # These hold actual prices, not ratios
        "ema_9", "ema_21", "ema_50", "ema_200",
        "bb_upper", "bb_lower", "bb_mid",
        "swing_high", "swing_low",
    }
    return [col for col in df.columns if col not in exclude and not col.startswith("_")]
