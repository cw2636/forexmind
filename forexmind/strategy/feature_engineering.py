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
    """
    Rolling statistics expressed as price-scale-independent ratios.
    Raw high/low rolling max/min are excluded — they carry price-level
    information that generalises poorly across instruments and time periods.
    """
    windows = windows or [5, 10, 20]
    for w in windows:
        roll_mean = df["close"].rolling(w).mean()
        roll_std  = df["close"].rolling(w).std()
        # Normalised distance from rolling mean (z-score style)
        df[f"close_dist_mean_{w}"]  = (df["close"] - roll_mean) / roll_mean.replace(0, np.nan)
        # Normalised volatility (coefficient of variation)
        df[f"close_roll_cv_{w}"]    = roll_std / roll_mean.replace(0, np.nan)
        # Rolling high/low expressed as % range above/below close — scale-free
        roll_high = df["high"].rolling(w).max()
        roll_low  = df["low"].rolling(w).min()
        df[f"roll_high_pct_{w}"]   = (roll_high - df["close"]) / df["close"].replace(0, np.nan)
        df[f"roll_low_pct_{w}"]    = (df["close"] - roll_low)  / df["close"].replace(0, np.nan)
        # Position of current close within the rolling range (0=at low, 1=at high)
        roll_range = (roll_high - roll_low).replace(0, np.nan)
        df[f"roll_pos_{w}"]        = (df["close"] - roll_low) / roll_range
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


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cyclically-encoded time features.

    Forex has extremely strong time-of-day and day-of-week patterns:
      - London open (7-9 UTC): high volatility, breakout moves
      - NY open (12-14 UTC): highest liquidity, trend continuation
      - London-NY overlap (12-16 UTC): prime scalping window
      - Asian session (0-7 UTC): range-bound, lower volatility
      - Friday 18-20 UTC: liquidity drop, wider spreads

    Sine/cosine encoding preserves the circular nature of time
    (e.g. hour 23 is close to hour 0, not far away).
    """
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        return df

    hour = idx.hour + idx.minute / 60.0
    dow  = idx.dayofweek.astype(float)   # 0=Monday … 4=Friday

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["dow_sin"]  = np.sin(2 * np.pi * dow  / 5.0)
    df["dow_cos"]  = np.cos(2 * np.pi * dow  / 5.0)

    # Session binary flags — cleaner signal than session_score alone
    df["is_london"]    = ((hour >= 7)  & (hour < 16)).astype(np.float32)
    df["is_ny"]        = ((hour >= 12) & (hour < 21)).astype(np.float32)
    df["is_tokyo"]     = ((hour >= 0)  & (hour < 9)).astype(np.float32)
    df["is_overlap"]   = ((hour >= 12) & (hour < 16)).astype(np.float32)
    df["is_friday_pm"] = ((dow == 4)   & (hour >= 16)).astype(np.float32)  # Low liquidity

    # Minutes since London open / NY open (momentum proxy for session age)
    london_open_h = 7.0
    ny_open_h     = 12.0
    df["mins_since_london"] = np.clip((hour - london_open_h) * 60, 0, 540)   # 0-9h
    df["mins_since_ny"]     = np.clip((hour - ny_open_h)     * 60, 0, 540)

    return df


def add_htf_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Higher-timeframe trend direction encoded as ML features.

    Constructs H1-equivalent features by resampling the existing bar data.
    This gives the model explicit multi-timeframe information instead of
    relying on the rule-based HTF filter to handle it implicitly.

    Features added:
      - htf_ema_trend:    +1 bullish, -1 bearish, 0 choppy
      - htf_rsi:          RSI on the H1 resampled close (normalised to -1…+1)
      - htf_macd_hist:    MACD histogram sign on H1 (-1/0/+1)
      - htf_adx:          ADX on H1 (normalised)

    NOTE: These columns are ALWAYS added (neutral 0.0 if computation fails)
    so that ML models trained with these features never crash at inference time.
    """
    htf_cols = ["htf_ema_trend", "htf_rsi", "htf_macd_hist", "htf_adx"]
    try:
        import pandas_ta as ta

        # Resample to H1 — use only past bars, align to bar close
        h1 = df[["open", "high", "low", "close", "volume"]].resample("1h").agg(
            {"open": "first", "high": "max", "low": "min",
             "close": "last", "volume": "sum"}
        ).dropna()

        if len(h1) < 50:
            log.debug("HTF feature computation skipped — fewer than 50 H1 bars available")
            for col in htf_cols:
                df[col] = 0.0
            return df

        # Compute HTF indicators on H1
        ema_9  = ta.ema(h1["close"], length=9)
        ema_21 = ta.ema(h1["close"], length=21)
        ema_50 = ta.ema(h1["close"], length=50)
        h1_rsi = ta.rsi(h1["close"], length=14)
        macd   = ta.macd(h1["close"])
        h1_adx = ta.adx(h1["high"], h1["low"], h1["close"], length=14)

        # EMA trend: +1 bullish stack, -1 bearish, 0 choppy
        ema_trend = pd.Series(0.0, index=h1.index)
        if ema_9 is not None and ema_21 is not None and ema_50 is not None:
            ema_trend = np.where(ema_9 > ema_21, 1.0, np.where(ema_9 < ema_21, -1.0, 0.0))
            ema_trend = pd.Series(ema_trend, index=h1.index)

        h1["htf_ema_trend"] = ema_trend
        h1["htf_rsi"]       = (h1_rsi - 50.0) / 50.0 if h1_rsi is not None else 0.0
        if macd is not None:
            # Use column name search — pandas-ta names the histogram "MACDh_12_26_9"
            macdh_col = next((c for c in macd.columns if c.startswith("MACDh_")), None)
            h1["htf_macd_hist"] = np.sign(macd[macdh_col]) if macdh_col else 0.0
        else:
            h1["htf_macd_hist"] = 0.0
        if h1_adx is not None:
            # pandas-ta names the ADX column "ADX_14"
            adx_col = next((c for c in h1_adx.columns if c.startswith("ADX_")), None)
            h1["htf_adx"] = h1_adx[adx_col] / 100.0 if adx_col else 0.0
        else:
            h1["htf_adx"] = 0.0

        # Forward-fill HTF values onto the M5 index (each M5 bar gets its H1 context)
        for col in htf_cols:
            filled = h1[col].reindex(df.index, method="ffill")
            df[col] = filled.fillna(0.0)  # fill any leading NaN with neutral

    except Exception as e:
        log.warning(f"HTF feature computation failed: {e} — filling with neutral zeros")
        for col in htf_cols:
            df[col] = 0.0

    return df


def add_volatility_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Volatility regime features that help the model understand market state.

    - atr_pct_rank: rolling percentile of ATR% over past 100 bars (0=calm, 1=volatile)
    - bb_squeeze:   flag when Bollinger Bands are at their narrowest in 50 bars
    - return_5:     5-bar log return (momentum)
    - return_20:    20-bar log return (trend)
    - vol_change:   ATR change rate (expanding=1, contracting=-1)
    """
    if "atr_pct" in df.columns:
        # Percentile rank of current ATR within recent history
        df["atr_pct_rank"] = df["atr_pct"].rolling(100, min_periods=20).rank(pct=True)
        # Volatility direction: is vol expanding or contracting?
        df["vol_change"] = np.sign(df["atr_pct"].diff(5))

    if "bb_width" in df.columns:
        min_width = df["bb_width"].rolling(50, min_periods=10).min()
        df["bb_squeeze"] = (df["bb_width"] <= min_width * 1.05).astype(np.float32)

    # Log returns at multiple horizons — direction-agnostic momentum
    close = df["close"]
    df["log_ret_1"]  = np.log(close / close.shift(1))
    df["log_ret_5"]  = np.log(close / close.shift(5))
    df["log_ret_20"] = np.log(close / close.shift(20))
    # Normalise by ATR so the signal is volatility-adjusted
    if "atr" in df.columns:
        atr = df["atr"].replace(0, np.nan)
        df["mom_1_atr"]  = df["log_ret_1"]  / atr
        df["mom_5_atr"]  = df["log_ret_5"]  / atr
        df["mom_20_atr"] = df["log_ret_20"] / atr

    return df


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


def add_target_label(
    df: pd.DataFrame,
    forward_bars: int = 12,
    threshold_pct: float | None = None,
    atr_threshold_mult: float = 0.5,
) -> pd.DataFrame:
    """
    Add a classification target: did price go UP, DOWN, or stay FLAT
    over the next `forward_bars` bars?

    target: 1 = UP (bullish), -1 = DOWN (bearish), 0 = flat

    ATR-adaptive threshold (default):
      threshold = atr_threshold_mult × current_atr_pct
      This adapts to volatility — in a fast market a 10-pip move is noise,
      in a quiet market it's a real signal. Produces cleaner, more consistent labels.

    Falls back to fixed threshold_pct if ATR is not available or threshold_pct
    is explicitly provided.

    Critical: the last `forward_bars` rows are set to NaN (future unknown).
    """
    future_return = df["close"].shift(-forward_bars) / df["close"] - 1.0

    if threshold_pct is None and "atr_pct" in df.columns:
        # Adaptive: threshold = half the current ATR%, rolling-smoothed
        # Clip to reasonable range [0.02%, 0.15%] to handle extreme regimes
        thresh_series = (df["atr_pct"] * atr_threshold_mult / 100.0).clip(0.0002, 0.0015)
        target = np.where(
            future_return > thresh_series, 1,
            np.where(future_return < -thresh_series, -1, 0)
        ).astype(float)
    else:
        thr = (threshold_pct or 0.05) / 100.0
        target = np.where(
            future_return > thr, 1,
            np.where(future_return < -thr, -1, 0)
        ).astype(float)

    # Mark the final `forward_bars` rows as NaN — their future is unknown
    target[-forward_bars:] = np.nan
    df["target"] = target
    df["future_return"] = future_return
    return df


# ── Pipeline orchestrator ─────────────────────────────────────────────────────

def build_feature_matrix(
    df: pd.DataFrame,
    add_target: bool = True,
    lags: int = 5,
    include_sessions: bool = True,
    forward_bars: int = 12,
) -> pd.DataFrame:
    """
    Apply the full feature engineering pipeline to a DataFrame.

    Args:
        df: OHLCV + indicator-augmented DataFrame (output of IndicatorEngine.compute())
        add_target: Whether to add the target label column (for training)
        lags: Number of lag features per indicator
        include_sessions: Whether to add Forex session flags
        forward_bars: Number of bars ahead the target looks (used to trim NaN tail)

    Returns:
        Feature-rich DataFrame ready for ML training/inference.
    """
    transforms: list[Transform] = [
        lambda d: add_lag_features(d, lags=lags),
        add_rolling_stats,
        add_candle_features,
        add_normalised_indicators,
        add_time_features,
        add_htf_trend_features,
        add_volatility_regime,
    ]
    if include_sessions:
        transforms.append(add_session_flags)
    if add_target:
        transforms.append(lambda d: add_target_label(d, forward_bars=forward_bars))

    # Apply all transforms in sequence using functools.reduce
    result = reduce(lambda d, fn: fn(d), transforms, df.copy())

    # Drop rows with NaN indicators (from lags and rolling windows)
    initial_len = len(result)
    dropna_cols = ["rsi", "macd", "adx"]
    if add_target:
        # Also drop rows where target is NaN — these are the final `forward_bars`
        # rows whose future outcome is unknown. Assigning them label 0 would corrupt training.
        dropna_cols.append("target")
    result = result.dropna(subset=dropna_cols).copy()
    log.debug(f"Feature engineering: {initial_len} → {len(result)} rows after dropna")
    return result


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the list of feature column names to pass to the ML model.
    Excludes raw OHLCV and any price-level columns that don't generalise
    across instruments or time periods.
    """
    exclude = {
        # Raw OHLCV
        "open", "high", "low", "close", "volume",
        # Target and auxiliary labels
        "target", "future_return",
        # Raw price-level pivot columns (not normalised)
        "pivot", "r1", "r2", "s1", "s2",
        "psar_long", "psar_short",
        # Absolute price columns — these encode the price level, not momentum
        "ema_9", "ema_21", "ema_50", "ema_200",
        "bb_upper", "bb_lower", "bb_mid",
        "swing_high", "swing_low",
        "macd", "macd_signal",   # Use macd_hist (difference) instead
        "psar",                  # Use psar_signal categorical or log-distance
        # Old-style rolling absolute price stats (replaced by normalised versions)
        "close_roll_mean_5", "close_roll_mean_10", "close_roll_mean_20",
        "close_roll_std_5", "close_roll_std_10", "close_roll_std_20",
        # Log returns are included; raw close changes would be price-level
        "log_ret_1", "log_ret_5", "log_ret_20",  # use atr-normalised versions instead
    }
    return [col for col in df.columns if col not in exclude and not col.startswith("_")]
