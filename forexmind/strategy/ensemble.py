"""
ForexMind — Ensemble Strategy
================================
Combines all strategy outputs into a single high-confidence signal.

Voting algorithm:
  1. Each strategy generates a StrategySignal with direction + confidence.
  2. HOLD votes are weighted at 0 (but they reduce the overall confidence).
  3. Weighted vote: BUY_score = sum(weight * confidence for BUY signals)
                   SELL_score = sum(weight * confidence for SELL signals)
  4. Direction with higher score wins if it exceeds a minimum threshold.
  5. Final risk_pct is decided by the risk manager (not here).

Advanced Python:
  - asyncio.gather for parallel signal generation
  - Strategy registry pattern (dict of name → instance)
  - Type-safe weighted averaging
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from forexmind.strategy.base import BaseStrategy, StrategySignal
from forexmind.strategy.rule_based import RuleBasedStrategy
from forexmind.strategy.ml_strategy import LightGBMStrategy, LSTMStrategy
from forexmind.strategy.rl_strategy import RLStrategy
from forexmind.utils.helpers import atr_stop_loss, atr_take_profit
from forexmind.utils.logger import get_logger
from forexmind.config.settings import get_settings

log = get_logger(__name__)

# Minimum ensemble score to emit BUY/SELL.
# With RL weight cut to 0.05, 3 strategies agreeing at 70% confidence now produces:
# 0.35×0.70 + 0.35×0.70 + 0.25×0.70 + 0.05×0 = 0.665 — well above this threshold.
# Raised from 0.42 — we want fewer, better signals, not more weak ones.
MIN_ENSEMBLE_CONFIDENCE = 0.55
# Require 2 out of 3 main strategies (rule_based + one ML) to agree.
MIN_AGREEING_STRATEGIES = 2
# Meaningful margin required — prevents acting on coin-flip splits.
MIN_SCORE_MARGIN = 0.08


@dataclass
class EnsembleSignal:
    """Full ensemble result including per-strategy breakdown."""
    instrument: str
    timeframe: str
    direction: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_pct: float
    buy_score: float
    sell_score: float
    agreeing_count: int
    total_strategies: int
    component_signals: list[StrategySignal] = field(default_factory=list)
    reasoning: str = ""

    @property
    def is_actionable(self) -> bool:
        return (
            self.direction != "HOLD"
            and self.confidence >= MIN_ENSEMBLE_CONFIDENCE
            and self.agreeing_count >= MIN_AGREEING_STRATEGIES
        )


class EnsembleStrategy:
    """
    Combines RuleBasedStrategy, LightGBM, LSTM, and RL signals.
    Weights start from config.yaml but are dynamically recalibrated
    based on each strategy's live directional precision.
    """

    def __init__(self) -> None:
        cfg = get_settings()
        weights = cfg.ensemble_weights

        # Registry: strategy name → (instance, weight)
        self._strategies: dict[str, tuple[BaseStrategy, float]] = {
            "rule_based": (RuleBasedStrategy(), weights.get("rule_based", 0.30)),
            "lightgbm":   (LightGBMStrategy(),  weights.get("lightgbm", 0.25)),
            "lstm":        (LSTMStrategy(),       weights.get("lstm", 0.25)),
            "rl_agent":   (RLStrategy(),          weights.get("rl_agent", 0.20)),
        }
        self._normalise_weights()

        # Live precision tracking: how often each strategy's non-HOLD call
        # agreed with the ensemble direction that eventually won.
        # {name: [correct_calls, total_calls]}
        self._precision: dict[str, list[int]] = {name: [0, 0] for name in self._strategies}
        # Minimum calls before precision-based reweighting kicks in
        self._MIN_CALLS_FOR_CALIBRATION = 50

    def _normalise_weights(self) -> None:
        total_weight = sum(w for _, w in self._strategies.values())
        if abs(total_weight - 1.0) > 0.01:
            log.warning(f"Ensemble weights sum to {total_weight:.3f}, not 1.0 — normalising")
            self._strategies = {
                name: (strat, w / total_weight)
                for name, (strat, w) in self._strategies.items()
            }

    def recalibrate_weights(self) -> None:
        """
        Recompute strategy weights based on live directional precision.
        Called automatically after each signal generation once enough data is
        accumulated. Precision = correct non-HOLD calls / total non-HOLD calls.
        Strategies with higher precision get proportionally more weight.
        Minimum weight floor of 0.05 prevents any strategy from being silenced.
        """
        MIN_WEIGHT = 0.05
        ready = {
            name: stats[0] / stats[1]
            for name, stats in self._precision.items()
            if stats[1] >= self._MIN_CALLS_FOR_CALIBRATION
        }
        if len(ready) < 2:
            return  # Not enough data yet

        total_precision = sum(ready.values())
        if total_precision == 0:
            return

        new_weights: dict[str, float] = {}
        for name in self._strategies:
            if name in ready:
                raw = ready[name] / total_precision
                new_weights[name] = max(MIN_WEIGHT, raw)
            else:
                # Strategy without enough calls keeps its current weight
                _, w = self._strategies[name]
                new_weights[name] = w

        # Renormalise to sum to 1.0
        total = sum(new_weights.values())
        self._strategies = {
            name: (strat, new_weights[name] / total)
            for name, (strat, _) in self._strategies.items()
        }
        log.info(
            f"Ensemble weights recalibrated from precision data: "
            + ", ".join(f"{n}={w:.3f}" for n, (_, w) in self._strategies.items())
        )

    def _update_precision(self, signals: list[StrategySignal], winning_direction: str) -> None:
        """Update precision counters after a signal is generated."""
        for sig, name in zip(signals, list(self._strategies.keys())):
            if sig.direction in ("BUY", "SELL"):
                self._precision[name][1] += 1  # total calls
                if sig.direction == winning_direction:
                    self._precision[name][0] += 1  # correct calls

        # Recalibrate every 20 new calls from any strategy
        total_calls = sum(s[1] for s in self._precision.values())
        if total_calls > 0 and total_calls % 20 == 0:
            self.recalibrate_weights()

    def generate_signal(
        self,
        df: pd.DataFrame,
        instrument: str,
        timeframe: str,
        current_price: float,
        htf_df: pd.DataFrame | None = None,
        bid: float | None = None,
        ask: float | None = None,
    ) -> EnsembleSignal:
        """
        Run all strategies synchronously and combine their votes.
        For async use, call generate_signal_async().
        """
        # ── Spread filter ─────────────────────────────────────────────────────
        # Wide spreads eat into profit — don't trade when spread exceeds config limit.
        if bid is not None and ask is not None:
            from forexmind.utils.helpers import price_to_pips
            from forexmind.config.settings import get_settings
            spread_pips = price_to_pips(ask - bid, instrument)
            _cfg = get_settings()
            max_spread = _cfg.risk_config_yaml.get("spread_filter_pips", 3.0)
            # Practice/paper spreads are typically 3-5× wider than live — scale the limit
            # so valid signals aren't suppressed in the demo environment.
            if _cfg.app.paper_trading:
                max_spread *= 3.0
            if spread_pips > max_spread:
                log.info(f"{instrument}: spread {spread_pips:.1f} pips > limit {max_spread:.1f} — HOLD")
                return _hold_ensemble(instrument, timeframe, current_price,
                                      f"Spread {spread_pips:.1f}>{max_spread:.1f} pips")

        signals: list[StrategySignal] = []
        for name, (strategy, _) in self._strategies.items():
            try:
                if name == "rule_based" and hasattr(strategy, "generate_signal"):
                    sig = strategy.generate_signal(df, instrument, timeframe, current_price, htf_df)  # type: ignore
                else:
                    sig = strategy.generate_signal(df, instrument, timeframe, current_price)
                signals.append(sig)
                log.debug(f"{name}: {sig.direction} conf={sig.confidence:.2f}")
            except Exception as e:
                log.warning(f"Strategy {name} raised an error: {e}")
                signals.append(BaseStrategy._hold_signal(  # type: ignore
                    strategy, instrument, timeframe, current_price, f"{name} error: {e}"
                ))

        return self._combine(signals, instrument, timeframe, current_price, df)

    async def generate_signal_async(
        self,
        df: pd.DataFrame,
        instrument: str,
        timeframe: str,
        current_price: float,
        htf_df: pd.DataFrame | None = None,
    ) -> EnsembleSignal:
        """
        Run all strategies concurrently using asyncio thread pool.
        CPU-bound ML inference runs in ThreadPoolExecutor to avoid blocking.
        """
        loop = asyncio.get_event_loop()

        async def _run(name: str, strat: BaseStrategy, weight: float) -> StrategySignal:
            try:
                if name == "rule_based":
                    sig = await loop.run_in_executor(
                        None, lambda: strat.generate_signal(df, instrument, timeframe, current_price, htf_df)  # type: ignore
                    )
                else:
                    sig = await loop.run_in_executor(
                        None, lambda: strat.generate_signal(df, instrument, timeframe, current_price)
                    )
                return sig
            except Exception as e:
                log.warning(f"Async strategy {name} error: {e}")
                return strat._hold_signal(instrument, timeframe, current_price, str(e))

        tasks = [
            _run(name, strat, weight)
            for name, (strat, weight) in self._strategies.items()
        ]
        signals = await asyncio.gather(*tasks)
        return self._combine(list(signals), instrument, timeframe, current_price, df)

    def _combine(
        self,
        signals: list[StrategySignal],
        instrument: str,
        timeframe: str,
        current_price: float,
        df: pd.DataFrame,
    ) -> EnsembleSignal:
        """Weighted-vote combination of all strategy signals."""
        weights = {name: w for name, (_, w) in self._strategies.items()}
        strategy_names = list(self._strategies.keys())

        buy_score = 0.0
        sell_score = 0.0
        agreeing_buy = 0
        agreeing_sell = 0

        for sig, name in zip(signals, strategy_names):
            w = weights.get(name, 0.25)
            if sig.direction == "BUY":
                buy_score += w * sig.confidence
                agreeing_buy += 1
            elif sig.direction == "SELL":
                sell_score += w * sig.confidence
                agreeing_sell += 1

        score_margin = abs(buy_score - sell_score)
        if (
            buy_score > sell_score
            and buy_score >= MIN_ENSEMBLE_CONFIDENCE
            and score_margin >= MIN_SCORE_MARGIN
        ):
            direction = "BUY"
            confidence = buy_score
            agreeing = agreeing_buy
        elif (
            sell_score > buy_score
            and sell_score >= MIN_ENSEMBLE_CONFIDENCE
            and score_margin >= MIN_SCORE_MARGIN
        ):
            direction = "SELL"
            confidence = sell_score
            agreeing = agreeing_sell
        else:
            direction = "HOLD"
            confidence = max(buy_score, sell_score)
            agreeing = 0

        # Use ATR from the last bar for stop calculation
        atr = 0.0
        if "atr" in df.columns:
            atr = float(df["atr"].dropna().iloc[-1]) if not df["atr"].dropna().empty else 0.0

        if direction != "HOLD" and atr > 0:
            stop_loss = atr_stop_loss(current_price, atr, direction, multiplier=1.5)
            take_profit = atr_take_profit(current_price, stop_loss, direction, rr_ratio=2.0)
        elif direction != "HOLD":
            # Fallback: use average of component signal stops
            active = [s for s in signals if s.direction == direction]
            if active:
                stop_loss = sum(s.stop_loss for s in active) / len(active)
                take_profit = sum(s.take_profit for s in active) / len(active)
            else:
                stop_loss = take_profit = 0.0
        else:
            stop_loss = take_profit = 0.0

        # Update precision tracking for non-HOLD signals
        if direction != "HOLD":
            self._update_precision(signals, direction)

        cfg = get_settings().risk
        risk_pct = cfg.min_risk_per_trade_pct + confidence * (
            cfg.max_risk_per_trade_pct - cfg.min_risk_per_trade_pct
        )

        reasoning_parts = [
            f"Ensemble: {direction} (BUY={buy_score:.3f}, SELL={sell_score:.3f})",
            f"Agreeing: {agreeing}/{len(signals)} strategies",
        ]
        for sig, name in zip(signals, strategy_names):
            reasoning_parts.append(f"  [{name}] {sig.direction} conf={sig.confidence:.2f}: {sig.reasoning[:80]}")

        return EnsembleSignal(
            instrument=instrument,
            timeframe=timeframe,
            direction=direction,
            confidence=round(confidence, 4),
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_pct=round(risk_pct, 2),
            buy_score=round(buy_score, 4),
            sell_score=round(sell_score, 4),
            agreeing_count=agreeing,
            total_strategies=len(signals),
            component_signals=signals,
            reasoning="\n".join(reasoning_parts),
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hold_ensemble(instrument: str, timeframe: str, price: float, reason: str) -> EnsembleSignal:
    """Return a zero-confidence HOLD EnsembleSignal."""
    return EnsembleSignal(
        instrument=instrument, timeframe=timeframe, direction="HOLD",
        confidence=0.0, entry_price=price, stop_loss=0.0, take_profit=0.0,
        risk_pct=0.0, buy_score=0.0, sell_score=0.0,
        agreeing_count=0, total_strategies=0, reasoning=f"HOLD — {reason}",
    )


# ── Singleton ─────────────────────────────────────────────────────────────────

_ensemble: EnsembleStrategy | None = None


def get_ensemble() -> EnsembleStrategy:
    global _ensemble
    if _ensemble is None:
        _ensemble = EnsembleStrategy()
    return _ensemble
