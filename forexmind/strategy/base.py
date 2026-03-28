"""
ForexMind — Abstract Strategy Base
=====================================
All strategies (rule-based, ML, RL) implement this interface.

Advanced Python concepts:
  - Abstract Base Classes (ABC) enforcing interface contracts
  - Protocol for duck-type compatibility
  - dataclass-based signal output
  - ClassVar for class-level metadata
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar

import pandas as pd


@dataclass
class StrategySignal:
    """
    Unified signal output from any strategy.
    All strategy types return this same structure so the ensemble can combine them.
    """
    instrument: str
    timeframe: str
    direction: str           # "BUY" | "SELL" | "HOLD"
    confidence: float        # 0.0–1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_pct: float = 1.0
    reasoning: str = ""
    source: str = "unknown"
    generated_at: datetime = field(default_factory=datetime.utcnow)

    # Derived properties
    @property
    def risk_reward(self) -> float:
        """Actual R:R ratio of this signal."""
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.entry_price)
        return reward / risk if risk > 0 else 0.0

    @property
    def is_actionable(self) -> bool:
        """True if the signal is worth trading."""
        return (
            self.direction != "HOLD"
            and self.confidence >= 0.45
            and self.risk_reward >= 1.5
            and self.stop_loss > 0
            and self.take_profit > 0
        )


class BaseStrategy(ABC):
    """
    Abstract base for all ForexMind strategies.

    Subclasses must implement:
      - generate_signal() → StrategySignal
      - name (class variable)
    """

    name: ClassVar[str] = "base"

    @abstractmethod
    def generate_signal(
        self,
        df: pd.DataFrame,
        instrument: str,
        timeframe: str,
        current_price: float,
    ) -> StrategySignal:
        """
        Generate a trading signal from OHLCV data.

        Args:
            df: DataFrame with DatetimeIndex, OHLCV columns, and pre-computed
                indicator columns (output of IndicatorEngine.compute()).
            instrument: Forex pair, e.g. "EUR_USD"
            timeframe: e.g. "M5"
            current_price: Latest mid price.

        Returns:
            StrategySignal — direction, confidence, entry/SL/TP.
        """
        ...

    def _hold_signal(
        self,
        instrument: str,
        timeframe: str,
        price: float,
        reason: str = "No clear signal",
    ) -> StrategySignal:
        """Convenience helper: return a HOLD signal."""
        return StrategySignal(
            instrument=instrument,
            timeframe=timeframe,
            direction="HOLD",
            confidence=0.0,
            entry_price=price,
            stop_loss=0.0,
            take_profit=0.0,
            reasoning=reason,
            source=self.name,
        )
