"""
ForexMind — Technical Indicators Engine
=========================================
Computes 25+ technical indicators on OHLCV DataFrames using pandas-ta.
All results are attached directly to the DataFrame as new columns.

Indicators implemented:
  TREND:      EMA(9,21,50,200), MACD, ADX, Parabolic SAR
  MOMENTUM:   RSI, Stochastic, Williams %R, CCI, MFI, ROC
  VOLATILITY: Bollinger Bands, ATR, Keltner Channels
  VOLUME:     OBV, VWAP (intraday)
  STRUCTURE:  Pivot Points, Swing Highs/Lows, Higher-High / Lower-Low detection

Advanced Python concepts:
  - pandas-ta Strategy object for efficient bulk computation
  - @dataclass for structured indicator snapshots
  - Cached property for lazy evaluation
  - TypedDict for typed dictionary returns
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False

from forexmind.config.settings import get_settings
from forexmind.utils.logger import get_logger

log = get_logger(__name__)


# ── Typed return structures ───────────────────────────────────────────────────

class IndicatorSnapshot(TypedDict):
    """Latest values of all indicators for one candle (the most recent bar)."""
    instrument: str
    timeframe: str
    timestamp: str

    # Trend
    ema_9: float
    ema_21: float
    ema_50: float
    ema_200: float
    ema_trend: str        # "bullish" | "weak_bullish" | "bearish" | "weak_bearish" | "choppy"
    macd: float
    macd_signal: float
    macd_hist: float
    macd_cross: str       # "bull_cross" | "bear_cross" | "none"
    adx: float
    adx_trend_strength: str  # "trending" | "ranging"
    dmp: float             # +DI (bullish directional strength)
    dmn: float             # -DI (bearish directional strength)
    psar: float
    psar_signal: str      # "bullish" | "bearish" | "neutral"

    # Momentum
    rsi: float
    rsi_zone: str         # "overbought" | "oversold" | "neutral"
    stoch_k: float
    stoch_d: float
    stoch_cross: str      # "bull_cross" | "bear_cross" | "none"
    cci: float
    williams_r: float
    mfi: float

    # Volatility
    bb_upper: float
    bb_mid: float
    bb_lower: float
    bb_width: float       # Normalized bandwidth
    bb_position: float    # 0=at lower, 0.5=at mid, 1.0=at upper
    atr: float
    atr_pct: float        # ATR as % of price

    # Volume / Price Structure
    obv: float
    pivot_high: float
    pivot_low: float
    support: float
    resistance: float


@dataclass
class IndicatorConfig:
    """Holds all configurable indicator parameters."""
    ema_periods: list[int] = field(default_factory=lambda: [9, 21, 50, 200])
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    stoch_k: int = 14
    stoch_d: int = 3
    adx_period: int = 14
    adx_trend_threshold: float = 25.0
    cci_period: int = 20
    williams_r_period: int = 14
    mfi_period: int = 14

    @classmethod
    def from_settings(cls) -> "IndicatorConfig":
        """Load config from config.yaml."""
        yaml_ind = get_settings().indicator_config
        cfg = cls()
        if yaml_ind:
            cfg.ema_periods = yaml_ind.get("ema_periods", cfg.ema_periods)
            cfg.rsi_period = yaml_ind.get("rsi_period", cfg.rsi_period)
            cfg.rsi_overbought = yaml_ind.get("rsi_overbought", cfg.rsi_overbought)
            cfg.rsi_oversold = yaml_ind.get("rsi_oversold", cfg.rsi_oversold)
            cfg.macd_fast = yaml_ind.get("macd_fast", cfg.macd_fast)
            cfg.macd_slow = yaml_ind.get("macd_slow", cfg.macd_slow)
            cfg.macd_signal = yaml_ind.get("macd_signal", cfg.macd_signal)
            cfg.bb_period = yaml_ind.get("bb_period", cfg.bb_period)
            cfg.bb_std = yaml_ind.get("bb_std", cfg.bb_std)
            cfg.atr_period = yaml_ind.get("atr_period", cfg.atr_period)
            cfg.adx_period = yaml_ind.get("adx_period", cfg.adx_period)
            cfg.adx_trend_threshold = yaml_ind.get("adx_trend_threshold", cfg.adx_trend_threshold)
        return cfg


# ── Core Indicator Engine ────────────────────────────────────────────────────

class IndicatorEngine:
    """
    Computes all technical indicators and returns a clean snapshot dict.

    Usage:
        engine = IndicatorEngine()
        df_with_indicators = engine.compute(df_ohlcv)
        snapshot = engine.snapshot(df_with_indicators, "EUR_USD", "M5")
    """

    def __init__(self, config: IndicatorConfig | None = None) -> None:
        self.cfg = config or IndicatorConfig.from_settings()
        if not PANDAS_TA_AVAILABLE:
            log.warning("pandas-ta not installed — indicators will be empty. Run: pip install pandas-ta")

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all indicator columns to df (in-place copy).
        Input: DataFrame with columns open, high, low, close, volume
               and a DatetimeIndex (UTC).
        Returns: augmented DataFrame (new object, original unchanged).
        """
        df = df.copy()
        if not PANDAS_TA_AVAILABLE or len(df) < 50:
            return df   # Not enough data — return as-is

        # ── EMAs ──────────────────────────────────────────────────────────────
        for period in self.cfg.ema_periods:
            df[f"ema_{period}"] = ta.ema(df["close"], length=period)

        # ── MACD ──────────────────────────────────────────────────────────────
        macd = ta.macd(
            df["close"],
            fast=self.cfg.macd_fast,
            slow=self.cfg.macd_slow,
            signal=self.cfg.macd_signal,
        )
        if macd is not None:
            # Use column name prefix search — avoids fragile integer indexing
            # pandas-ta returns: MACD_{f}_{s}_{sig}, MACDh_{f}_{s}_{sig}, MACDs_{f}_{s}_{sig}
            macd_col  = next((c for c in macd.columns if c.startswith("MACD_")), None)
            macdh_col = next((c for c in macd.columns if c.startswith("MACDh_")), None)
            macds_col = next((c for c in macd.columns if c.startswith("MACDs_")), None)
            if macd_col:  df["macd"]        = macd[macd_col]
            if macdh_col: df["macd_hist"]   = macd[macdh_col]
            if macds_col: df["macd_signal"] = macd[macds_col]

        # ── ADX ───────────────────────────────────────────────────────────────
        adx = ta.adx(df["high"], df["low"], df["close"], length=self.cfg.adx_period)
        if adx is not None:
            # pandas-ta returns: ADX_{n}, DMP_{n} (+DI), DMN_{n} (-DI)
            adx_col = next((c for c in adx.columns if c.startswith("ADX_")), None)
            dmp_col = next((c for c in adx.columns if c.startswith("DMP_")), None)
            dmn_col = next((c for c in adx.columns if c.startswith("DMN_")), None)
            if adx_col: df["adx"] = adx[adx_col]
            if dmp_col: df["dmp"] = adx[dmp_col]   # +DI: bullish directional strength
            if dmn_col: df["dmn"] = adx[dmn_col]   # -DI: bearish directional strength

        # ── Parabolic SAR ─────────────────────────────────────────────────────
        psar = ta.psar(df["high"], df["low"], df["close"])
        if psar is not None:
            # pandas-ta returns: PSARl_{step}_{max} (long/bullish dots below price)
            #                    PSARs_{step}_{max} (short/bearish dots above price)
            # Exactly one is non-NaN per bar — whichever SAR is active.
            psar_long_col  = next((c for c in psar.columns if "PSARl" in c), None)
            psar_short_col = next((c for c in psar.columns if "PSARs" in c), None)
            if psar_long_col:  df["psar_long"]  = psar[psar_long_col]
            if psar_short_col: df["psar_short"] = psar[psar_short_col]
            df["psar"] = df["psar_long"].fillna(df["psar_short"])

        # ── RSI ───────────────────────────────────────────────────────────────
        df["rsi"] = ta.rsi(df["close"], length=self.cfg.rsi_period)

        # ── Stochastic ────────────────────────────────────────────────────────
        stoch = ta.stoch(
            df["high"], df["low"], df["close"],
            k=self.cfg.stoch_k, d=self.cfg.stoch_d,
        )
        if stoch is not None:
            df["stoch_k"] = stoch.iloc[:, 0]
            df["stoch_d"] = stoch.iloc[:, 1]

        # ── CCI ───────────────────────────────────────────────────────────────
        df["cci"] = ta.cci(df["high"], df["low"], df["close"], length=self.cfg.cci_period)

        # ── Williams %R ───────────────────────────────────────────────────────
        df["williams_r"] = ta.willr(
            df["high"], df["low"], df["close"], length=self.cfg.williams_r_period
        )

        # ── MFI ───────────────────────────────────────────────────────────────
        if df["volume"].sum() > 0:
            df["mfi"] = ta.mfi(
                df["high"], df["low"], df["close"], df["volume"], length=self.cfg.mfi_period
            )

        # ── Bollinger Bands ───────────────────────────────────────────────────
        bb = ta.bbands(df["close"], length=self.cfg.bb_period, std=self.cfg.bb_std)
        if bb is not None:
            df["bb_lower"] = bb.iloc[:, 0]
            df["bb_mid"] = bb.iloc[:, 1]
            df["bb_upper"] = bb.iloc[:, 2]
            df["bb_width"] = (bb.iloc[:, 2] - bb.iloc[:, 0]) / bb.iloc[:, 1]
            df["bb_pct"] = (df["close"] - bb.iloc[:, 0]) / (bb.iloc[:, 2] - bb.iloc[:, 0])

        # ── ATR ───────────────────────────────────────────────────────────────
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=self.cfg.atr_period)
        df["atr_pct"] = df["atr"] / df["close"] * 100.0

        # ── OBV ───────────────────────────────────────────────────────────────
        if df["volume"].sum() > 0:
            df["obv"] = ta.obv(df["close"], df["volume"])

        # ── Rate of Change ────────────────────────────────────────────────────
        df["roc_5"] = ta.roc(df["close"], length=5)
        df["roc_10"] = ta.roc(df["close"], length=10)

        # ── Pivot Points (last completed session) ─────────────────────────────
        df = self._add_pivot_points(df)

        # ── Swing Highs/Lows ─────────────────────────────────────────────────
        df = self._add_swing_levels(df)

        return df

    def _add_pivot_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classic floor pivot points based on the previous candle."""
        df["pivot"] = (df["high"].shift(1) + df["low"].shift(1) + df["close"].shift(1)) / 3
        df["r1"] = 2 * df["pivot"] - df["low"].shift(1)
        df["s1"] = 2 * df["pivot"] - df["high"].shift(1)
        df["r2"] = df["pivot"] + (df["high"].shift(1) - df["low"].shift(1))
        df["s2"] = df["pivot"] - (df["high"].shift(1) - df["low"].shift(1))
        return df

    def _add_swing_levels(self, df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
        """
        Mark swing highs and lows using only past bars (no center=True look-ahead).
        A bar is a swing high if its high was the highest of the previous `lookback` bars.
        """
        rolling_high = df["high"].rolling(lookback, center=False).max()
        rolling_low = df["low"].rolling(lookback, center=False).min()
        df["swing_high"] = np.where(df["high"] == rolling_high, df["high"], np.nan)
        df["swing_low"] = np.where(df["low"] == rolling_low, df["low"], np.nan)
        return df

    def snapshot(
        self,
        df: pd.DataFrame,
        instrument: str,
        timeframe: str,
    ) -> IndicatorSnapshot:
        """
        Extract the latest bar's indicator values into a structured dict.
        Call compute() first to populate the indicator columns.
        """
        if df.empty:
            return _empty_snapshot(instrument, timeframe)

        row = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else row

        def _f(col: str, default: float = 0.0) -> float:
            val = row.get(col, default)
            return float(val) if pd.notna(val) else default

        # ── Derived signals ────────────────────────────────────────────────────
        close = _f("close")
        ema_9 = _f("ema_9")
        ema_21 = _f("ema_21")
        ema_50 = _f("ema_50")
        ema_200 = _f("ema_200")

        # EMA trend alignment: full stack or partial (weak) alignment
        # bullish      — 9 > 21 > 50: all three fully aligned up
        # weak_bullish — 9 > 21, but 21 < 50: short-term bullish, trend turning
        # bearish      — 9 < 21 < 50: all three fully aligned down
        # weak_bearish — 9 < 21, but 21 > 50: short-term bearish, trend turning
        # choppy       — no clear short-term alignment
        if ema_9 > ema_21 > ema_50:
            ema_trend = "bullish"
        elif ema_9 < ema_21 < ema_50:
            ema_trend = "bearish"
        elif ema_9 > ema_21:
            ema_trend = "weak_bullish"
        elif ema_9 < ema_21:
            ema_trend = "weak_bearish"
        else:
            ema_trend = "choppy"

        # MACD cross detection
        macd_val = _f("macd")
        macd_sig = _f("macd_signal")
        prev_macd = float(prev.get("macd", 0) or 0)
        prev_macd_sig = float(prev.get("macd_signal", 0) or 0)
        if prev_macd <= prev_macd_sig and macd_val > macd_sig:
            macd_cross = "bull_cross"
        elif prev_macd >= prev_macd_sig and macd_val < macd_sig:
            macd_cross = "bear_cross"
        else:
            macd_cross = "none"

        # ADX trend strength + directional movement
        adx_val = _f("adx")
        adx_trend_strength = "trending" if adx_val > self.cfg.adx_trend_threshold else "ranging"
        dmp_val = _f("dmp")   # +DI
        dmn_val = _f("dmn")   # -DI

        # Parabolic SAR direction — only valid when psar is non-zero (NaN was replaced with 0 by _f)
        psar_raw = row.get("psar", None)
        if pd.notna(psar_raw) and float(psar_raw) != 0.0:
            psar_val = float(psar_raw)
            psar_signal = "bullish" if close > psar_val else "bearish"
        else:
            psar_val = 0.0
            psar_signal = "neutral"  # No active SAR — don't influence direction

        # RSI zone
        rsi_val = _f("rsi")
        rsi_zone = (
            "overbought" if rsi_val > self.cfg.rsi_overbought
            else "oversold" if rsi_val < self.cfg.rsi_oversold
            else "neutral"
        )

        # Stoch cross
        sk = _f("stoch_k")
        sd = _f("stoch_d")
        prev_sk = float(prev.get("stoch_k", 0) or 0)
        prev_sd = float(prev.get("stoch_d", 0) or 0)
        if prev_sk <= prev_sd and sk > sd:
            stoch_cross = "bull_cross"
        elif prev_sk >= prev_sd and sk < sd:
            stoch_cross = "bear_cross"
        else:
            stoch_cross = "none"

        # Nearest support / resistance from pivot points
        pivot = _f("pivot")
        r1 = _f("r1")
        r2 = _f("r2")
        s1 = _f("s1")
        s2 = _f("s2")

        if close > pivot:
            support = s1 if close > s1 else s2
            resistance = r1
        else:
            support = s1
            resistance = r1 if close < r1 else r2

        return IndicatorSnapshot(
            instrument=instrument,
            timeframe=timeframe,
            timestamp=str(df.index[-1]),
            ema_9=ema_9, ema_21=ema_21, ema_50=ema_50, ema_200=ema_200,
            ema_trend=ema_trend,
            macd=macd_val, macd_signal=_f("macd_signal"), macd_hist=_f("macd_hist"),
            macd_cross=macd_cross,
            adx=adx_val, adx_trend_strength=adx_trend_strength,
            dmp=dmp_val, dmn=dmn_val,
            psar=psar_val, psar_signal=psar_signal,
            rsi=rsi_val, rsi_zone=rsi_zone,
            stoch_k=sk, stoch_d=sd, stoch_cross=stoch_cross,
            cci=_f("cci"), williams_r=_f("williams_r"), mfi=_f("mfi"),
            bb_upper=_f("bb_upper"), bb_mid=_f("bb_mid"), bb_lower=_f("bb_lower"),
            bb_width=_f("bb_width"), bb_position=_f("bb_pct"),
            atr=_f("atr"), atr_pct=_f("atr_pct"),
            obv=_f("obv"),
            pivot_high=_f("swing_high"),
            pivot_low=_f("swing_low"),
            support=support,
            resistance=resistance,
        )


def _empty_snapshot(instrument: str, timeframe: str) -> IndicatorSnapshot:
    return IndicatorSnapshot(
        instrument=instrument, timeframe=timeframe, timestamp="",
        ema_9=0, ema_21=0, ema_50=0, ema_200=0, ema_trend="choppy",
        macd=0, macd_signal=0, macd_hist=0, macd_cross="none",
        adx=0, adx_trend_strength="ranging", dmp=0, dmn=0,
        psar=0, psar_signal="neutral",
        rsi=50, rsi_zone="neutral",
        stoch_k=50, stoch_d=50, stoch_cross="none",
        cci=0, williams_r=-50, mfi=50,
        bb_upper=0, bb_mid=0, bb_lower=0, bb_width=0, bb_position=0.5,
        atr=0, atr_pct=0, obv=0,
        pivot_high=0, pivot_low=0, support=0, resistance=0,
    )


# ── Singleton ─────────────────────────────────────────────────────────────────

_engine: IndicatorEngine | None = None


def get_indicator_engine() -> IndicatorEngine:
    global _engine
    if _engine is None:
        _engine = IndicatorEngine()
    return _engine
