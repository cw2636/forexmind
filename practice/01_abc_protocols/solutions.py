"""
Topic 01 — Abstract Base Classes & Protocols
=============================================
SOLUTIONS — Complete working implementations with explanations.

Only read this AFTER attempting exercises.py yourself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Protocol, runtime_checkable
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# SOLUTION 1 — StrategyLike Protocol
# ─────────────────────────────────────────────────────────────────────────────
# Key insight: @runtime_checkable must come BEFORE Protocol in the decorator
# stack. Without it, isinstance(obj, StrategyLike) raises TypeError.
# With it, isinstance() only checks that the attributes/methods EXIST —
# it does NOT verify signatures.

@runtime_checkable                        # enables isinstance() at runtime
class StrategyLike(Protocol):             # Protocol = structural subtyping
    name: str                             # attribute must exist

    def generate_signal(
        self,
        df: pd.DataFrame,
        instrument: str,
        timeframe: str,
        current_price: float,
    ) -> dict: ...                        # method must exist (signature not enforced at runtime)


# ─────────────────────────────────────────────────────────────────────────────
# SOLUTION 2 — BaseStrategy ABC
# ─────────────────────────────────────────────────────────────────────────────

class BaseStrategy(ABC):
    """
    Abstract base for all ForexMind strategies.

    Key design choices:
    - ClassVar[str] for `name` — shared across instances, excluded from __init__
    - @abstractmethod on generate_signal ONLY — subclasses still inherit
      concrete helpers like validate_data and _hold_signal for free
    - __subclasshook__ lets "alien" classes pass isinstance() without inheritance
    """

    name: ClassVar[str] = "base"          # ClassVar → not an instance field

    @abstractmethod
    def generate_signal(
        self,
        df: pd.DataFrame,
        instrument: str,
        timeframe: str,
        current_price: float,
    ) -> dict:
        """Must be implemented by every concrete strategy."""
        ...

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Concrete default — subclasses can override but don't have to.
        Returns True if the DataFrame has enough rows for indicator lookback.
        """
        return len(df) >= 50

    def _hold_signal(
        self,
        instrument: str,
        timeframe: str,
        price: float,
        reason: str = "No clear signal",
    ) -> dict:
        """Convenience helper: build a HOLD signal dict."""
        return {
            "direction":  "HOLD",
            "confidence": 0.0,
            "source":     self.name,
            "reasoning":  reason,
        }

    # ── Exercise 4 solution: __subclasshook__ ──────────────────────────────
    @classmethod
    def __subclasshook__(cls, C):
        """
        If a class has BOTH `name` and `generate_signal`, treat it as a
        virtual subclass of BaseStrategy — no inheritance needed.

        C.__mro__ is the method resolution order (list of parent classes).
        We search each class's own __dict__ (not inherited) for the attributes.

        Returns:
            True         → C IS considered a subclass
            NotImplemented → fall back to normal inheritance check
        """
        if cls is BaseStrategy:
            has_signal = any("generate_signal" in B.__dict__ for B in C.__mro__)
            has_name   = any("name"            in B.__dict__ for B in C.__mro__)
            if has_signal and has_name:
                return True
        return NotImplemented   # tell Python: use normal subclass check


# ─────────────────────────────────────────────────────────────────────────────
# SOLUTION 3 — SimpleMAStrategy
# ─────────────────────────────────────────────────────────────────────────────

class SimpleMAStrategy(BaseStrategy):
    """Crosses EMA-9 and EMA-21 to determine direction."""

    name = "simple_ma"                    # satisfies ClassVar requirement

    def generate_signal(
        self,
        df: pd.DataFrame,
        instrument: str,
        timeframe: str,
        current_price: float,
    ) -> dict:
        if not self.validate_data(df):
            return self._hold_signal(instrument, timeframe, current_price,
                                     reason="Not enough data")

        last = df.iloc[-1]                # most recent bar
        ema_9  = last["ema_9"]
        ema_21 = last["ema_21"]

        if ema_9 > ema_21:
            return {
                "direction":  "BUY",
                "confidence": 0.6,
                "source":     self.name,
                "reasoning":  f"EMA9 ({ema_9:.5f}) > EMA21 ({ema_21:.5f})",
            }
        elif ema_9 < ema_21:
            return {
                "direction":  "SELL",
                "confidence": 0.6,
                "source":     self.name,
                "reasoning":  f"EMA9 ({ema_9:.5f}) < EMA21 ({ema_21:.5f})",
            }
        else:
            return self._hold_signal(instrument, timeframe, current_price,
                                     reason="EMAs are equal — no signal")

    def validate_data(self, df: pd.DataFrame) -> bool:
        """Override: only need 21 rows for EMA-21 lookback."""
        return len(df) >= 21              # overrides the default of 50


# ─────────────────────────────────────────────────────────────────────────────
# SOLUTION 4 — AlienStrategy (no inheritance, passes isinstance via __subclasshook__)
# ─────────────────────────────────────────────────────────────────────────────

class AlienStrategy:
    """
    Third-party strategy — NOT inheriting from BaseStrategy.
    But it has `name` + `generate_signal`, so __subclasshook__ accepts it.
    """
    name = "alien"

    def generate_signal(self, df, instrument, timeframe, current_price):
        return {"direction": "BUY", "confidence": 0.6}


# ─────────────────────────────────────────────────────────────────────────────
# SOLUTION 5 — LegacyStrategy + .register()
# ─────────────────────────────────────────────────────────────────────────────

class LegacyStrategy:
    """
    Old strategy from a library — cannot be modified.
    No generate_signal! register() will still make it a "virtual subclass".
    """
    name = "legacy"


# Register — tells Python's type system to treat LegacyStrategy as a subclass
# of BaseStrategy WITHOUT inheritance.
#
# IMPORTANT: register() bypasses all abstract method checks.
# Python trusts that you know what you're doing. If you call
# LegacyStrategy().generate_signal(), you get AttributeError.
BaseStrategy.register(LegacyStrategy)


# ─────────────────────────────────────────────────────────────────────────────
# Run all tests
# ─────────────────────────────────────────────────────────────────────────────

def test_exercise_1():
    print("\n--- Exercise 1: Protocol ---")
    class MinimalStrategy:
        name = "minimal"
        def generate_signal(self, df, instrument, timeframe, price): return {}

    assert isinstance(MinimalStrategy(), StrategyLike)

    class NotAStrategy:
        name = "broken"

    assert not isinstance(NotAStrategy(), StrategyLike)
    print("  PASS")


def test_exercise_2():
    print("\n--- Exercise 2: ABC enforcement ---")
    try:
        BaseStrategy()
        print("  FAIL: should have raised TypeError")
    except TypeError as e:
        print(f"  PASS: {e}")

    class Concrete(BaseStrategy):
        name = "concrete"
        def generate_signal(self, df, instrument, timeframe, price): return {}

    s = Concrete()
    assert not s.validate_data(pd.DataFrame({"c": range(10)}))
    assert     s.validate_data(pd.DataFrame({"c": range(60)}))
    print("  PASS: validate_data works")


def test_exercise_3():
    print("\n--- Exercise 3: SimpleMAStrategy ---")
    import numpy as np
    strategy = SimpleMAStrategy()
    n = 30

    df_buy = pd.DataFrame({"close": [1.0]*n, "ema_9":  [1.1050]*n, "ema_21": [1.1030]*n})
    assert strategy.generate_signal(df_buy,  "EUR_USD", "M5", 1.1050)["direction"] == "BUY"

    df_sell = pd.DataFrame({"close": [1.0]*n, "ema_9": [1.1020]*n, "ema_21": [1.1035]*n})
    assert strategy.generate_signal(df_sell, "EUR_USD", "M5", 1.1020)["direction"] == "SELL"

    assert strategy.validate_data(pd.DataFrame({"close": range(10)})) is False
    assert strategy.validate_data(pd.DataFrame({"close": range(25)})) is True
    print("  PASS")


def test_exercise_4():
    print("\n--- Exercise 4: __subclasshook__ ---")
    assert isinstance(AlienStrategy(), BaseStrategy)
    print(f"  isinstance(AlienStrategy(), BaseStrategy) = True  PASS")


def test_exercise_5():
    print("\n--- Exercise 5: ABC.register() ---")
    assert issubclass(LegacyStrategy, BaseStrategy)
    legacy = LegacyStrategy()
    try:
        legacy.generate_signal(None, None, None, None)
    except AttributeError as e:
        print(f"  CORRECT: AttributeError: {e}")
        print("  register() bypasses abstract method enforcement — dangerous!")
    print("  PASS: issubclass works")


if __name__ == "__main__":
    print("=" * 60)
    print("Topic 01 — ABC & Protocols — SOLUTIONS")
    print("=" * 60)
    test_exercise_1()
    test_exercise_2()
    test_exercise_3()
    test_exercise_4()
    test_exercise_5()
    print("\n" + "=" * 60)
    print("All solutions verified!")
