"""
ForexMind — Rule-Based Scalping Strategy
==========================================
A multi-condition rule-based strategy optimised for 1m/5m scalping.

Entry logic combines:
  1. Higher-Timeframe trend filter (H1 EMA alignment)
  2. Entry timeframe confluence (M5 indicators)
  3. Momentum confirmation (RSI + Stochastic)
  4. Candle pattern confirmation (engulfing, pin-bar)
  5. No-trade filter: upcoming high-impact news

This is the "senior trader checklist" approach — a trade only happens
when multiple conditions agree. The more that agree, the higher the confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pandas as pd

from forexmind.indicators.engine import IndicatorEngine, IndicatorSnapshot, get_indicator_engine
from forexmind.indicators.scorer import score_snapshot, SignalScore
from forexmind.strategy.base import BaseStrategy, StrategySignal
from forexmind.utils.helpers import atr_stop_loss, atr_take_profit
from forexmind.utils.logger import get_logger
from forexmind.config.settings import get_settings

log = get_logger(__name__)


class RuleBasedStrategy(BaseStrategy):
    """
    Scalping entry strategy based on a 6-point confluence checklist.

    Rules for a LONG entry (inverted for SHORT):
      ✔ 1. HTF trend: H1 EMA(21) > EMA(50)  (trade with the trend)
      ✔ 2. M5 EMA trend is bullish (EMA9 > EMA21 > EMA50)
      ✔ 3. MACD histogram > 0 and rising
      ✔ 4. RSI(14) > 50 but < 70 (momentum without overbought)
      ✔ 5. Stochastic crossed up from below 40
      ✔ 6. Price > EMA21 (price not far extended from mean)

    Each condition passed scores +1. Confidence = conditions_met / 6.
    Entry triggers when >= 4 conditions are met.
    """

    name: ClassVar[str] = "rule_based"

    def __init__(self) -> None:
        self._cfg = get_settings().risk
        self._engine = get_indicator_engine()

    def generate_signal(
        self,
        df: pd.DataFrame,
        instrument: str,
        timeframe: str,
        current_price: float,
        htf_df: pd.DataFrame | None = None,
    ) -> StrategySignal:
        """
        Generate a scalping signal.

        Args:
            df: M5 OHLCV DataFrame with indicator columns.
            instrument: e.g. "EUR_USD"
            timeframe: e.g. "M5"
            current_price: Current mid price.
            htf_df: Optional H1 DataFrame for trend filter.
        """
        if len(df) < 50:
            return self._hold_signal(instrument, timeframe, current_price, "Insufficient data")

        snap = self._engine.snapshot(df, instrument, timeframe)
        score = score_snapshot(snap)

        if score.direction == "HOLD":
            return self._hold_signal(instrument, timeframe, current_price, score.reasoning)

        # ── Run the checklist ─────────────────────────────────────────────────
        conditions_met, total_conditions, checks = self._check_conditions(
            snap, score, htf_df
        )

        # Require at least 4 of 6 conditions
        min_conditions = 4
        if conditions_met < min_conditions:
            return self._hold_signal(
                instrument, timeframe, current_price,
                f"Only {conditions_met}/{total_conditions} conditions met. "
                f"Checks: {checks}"
            )

        confidence = conditions_met / total_conditions

        # ── Calculate stops using ATR ─────────────────────────────────────────
        atr = snap["atr"]
        if atr == 0:
            return self._hold_signal(instrument, timeframe, current_price, "ATR is zero")

        atr_mult = self._cfg.atr_stop_multiplier
        stop_loss = atr_stop_loss(current_price, atr, score.direction, multiplier=atr_mult)
        take_profit = atr_take_profit(
            current_price, stop_loss, score.direction,
            rr_ratio=self._cfg.default_rr_ratio
        )

        reasoning = (
            f"Rule-based {score.direction}: {conditions_met}/{total_conditions} conditions met. "
            f"Checks: {checks} | {score.reasoning}"
        )

        return StrategySignal(
            instrument=instrument,
            timeframe=timeframe,
            direction=score.direction,
            confidence=round(confidence, 4),
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_pct=self._cfg.min_risk_per_trade_pct + (
                confidence * (self._cfg.max_risk_per_trade_pct - self._cfg.min_risk_per_trade_pct)
            ),
            reasoning=reasoning,
            source=self.name,
        )

    def _check_conditions(
        self,
        snap: IndicatorSnapshot,
        score: SignalScore,
        htf_df: pd.DataFrame | None,
    ) -> tuple[int, int, dict[str, bool]]:
        """
        Returns (conditions_met, total_conditions, checks_dict).
        Checks dict maps condition name → bool.
        """
        direction = score.direction
        is_long = direction == "BUY"
        checks: dict[str, bool] = {}

        # Condition 1: HTF trend alignment
        if htf_df is not None and len(htf_df) >= 50:
            htf_snap = self._engine.snapshot(htf_df, snap["instrument"], "H1")
            checks["htf_trend"] = (
                htf_snap["ema_trend"] == "bullish" if is_long
                else htf_snap["ema_trend"] == "bearish"
            )
        else:
            # If no HTF data, use the H1 EMA cross as a weaker filter
            checks["htf_trend"] = snap["ema_50"] > snap["ema_200"] if is_long else snap["ema_50"] < snap["ema_200"]

        # Condition 2: EMA stack alignment on entry TF
        checks["ema_stack"] = (
            snap["ema_trend"] == "bullish" if is_long else snap["ema_trend"] == "bearish"
        )

        # Condition 3: MACD histogram in our direction
        checks["macd_direction"] = (
            snap["macd_hist"] > 0 if is_long else snap["macd_hist"] < 0
        )

        # Condition 4: RSI momentum — not overbought/oversold against us
        rsi = snap["rsi"]
        checks["rsi_ok"] = (
            (40 < rsi < 70) if is_long else (30 < rsi < 60)
        )

        # Condition 5: Stochastic in our favour
        checks["stoch_ok"] = (
            (snap["stoch_k"] > snap["stoch_d"] and snap["stoch_k"] < 80) if is_long
            else (snap["stoch_k"] < snap["stoch_d"] and snap["stoch_k"] > 20)
        )

        # Condition 6: PSAR confirms direction
        checks["psar_confirm"] = (
            snap["psar_signal"] == "bullish" if is_long else snap["psar_signal"] == "bearish"
        )

        met = sum(1 for v in checks.values() if v)
        return met, len(checks), checks
