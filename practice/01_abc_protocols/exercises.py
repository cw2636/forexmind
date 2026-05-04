"""
Topic 01 — Abstract Base Classes & Protocols
=============================================
EXERCISES — Fill in every section marked TODO.

Run this file to check your work:
    python 01_abc_protocols/exercises.py

All exercises are based on code from forexmind/strategy/base.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Protocol, runtime_checkable
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 1 — Create a Protocol
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Define a Protocol called `StrategyLike` that:
#   - Has a `name` attribute of type str
#   - Has a `generate_signal(df, instrument, timeframe, price)` method
#   - Supports isinstance() checks at runtime
#
# Why: Allows any object with those attributes to be used as a strategy
#      without forcing inheritance from BaseStrategy.
#
# Hint: You need two decorators — one from `typing`, one already imported above.

# TODO: Define StrategyLike Protocol here
 @runtime_checkable
 class StrategyLike(Protocol):
     name: str
     def generate_signal(self, df: pd.DataFrame, instrument: str,
#                         timeframe: str, current_price: float): ...


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 2 — Build the ABC
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Create `BaseStrategy(ABC)` with:
#   - ClassVar `name` defaulting to "base"
#   - Abstract method `generate_signal(df, instrument, timeframe, current_price)`
#   - Concrete method `validate_data(df)` returning True if len(df) >= 50
#   - Helper `_hold_signal(instrument, timeframe, price, reason)` that returns
#     a dict {"direction": "HOLD", "confidence": 0.0, "source": self.name,
#             "reasoning": reason}

# TODO: Define BaseStrategy here
 class BaseStrategy(ABC):
     name: ClassVar[str] = "base"

     @abstractmethod
     def generate_signal(self, df: pd.DataFrame, instrument: str, timeframe: str, current_price: float):
         pass

     def validate_data(self, df: pd.DataFrame) -> bool:
         return len(df) >= 50

     def _hold_signal(self, instrument: str, timeframe: str, price: float, reason: str) -> dict:
         return {
             "direction": "HOLD",
             "confidence": 0.0,
             "source": self.name,
             "reasoning": reason
         }


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 3 — Implement a concrete strategy
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Create `SimpleMAStrategy(BaseStrategy)` that:
#   - Sets `name = "simple_ma"`
#   - Implements `generate_signal`:
#       * Returns BUY  if the last row's "ema_9" > last row's "ema_21"
#       * Returns SELL if the last row's "ema_9" < last row's "ema_21"
#       * Returns the _hold_signal dict otherwise
#   - Overrides `validate_data` to require >= 21 rows (MA lookback)
#
# Hint: Access last row with df.iloc[-1]

# TODO: Define SimpleMAStrategy here
class SimpleMAStrategy(BaseStrategy):
    name: ClassVar[str] = "simple_ma"

    def generate_signal(self, df: pd.DataFrame, instrument: str, timeframe: str, current_price: float) -> dict:
        last_row = df.iloc[-1]
        ema_9 = last_row["ema_9"]
        ema_21 = last_row["ema_21"]

        if ema_9 > ema_21:
            return {"direction": "BUY", "confidence": 0.8, "source": self.name}
        elif ema_9 < ema_21:
            return {"direction": "SELL", "confidence": 0.8, "source": self.name}
        else:
            return self._hold_signal(instrument, timeframe, current_price, reason="EMA crossover unclear")

    def validate_data(self, df: pd.DataFrame) -> bool:
        return len(df) >= 21

# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 4 — __subclasshook__
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Add `__subclasshook__` to BaseStrategy (edit your class above) so that
#       ANY class with BOTH a `name` attribute AND a `generate_signal` method
#       is automatically recognised as a subclass — even without inheriting.
#
# Then create `AlienStrategy` below (NO inheritance from BaseStrategy) and
# verify that isinstance(AlienStrategy(), BaseStrategy) returns True.

class AlienStrategy:
    """Third-party strategy — cannot be edited to inherit from BaseStrategy."""
    name = "alien"

    def generate_signal(self, df, instrument, timeframe, current_price):
        return {"direction": "BUY", "confidence": 0.6}


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 5 — ABC.register()
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Create `LegacyStrategy` (no inheritance, no generate_signal method).
#       Register it as a virtual subclass of BaseStrategy using .register().
#       Verify issubclass(LegacyStrategy, BaseStrategy) is True.
#
# Then answer (as a comment): does LegacyStrategy need to implement
# generate_signal? What happens if you call .generate_signal() on an instance?

class LegacyStrategy:
    """Old strategy from a library — cannot be modified."""
    name = "legacy"
    # No generate_signal method!


# TODO: Register LegacyStrategy with BaseStrategy


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — Run these to check your answers
# ─────────────────────────────────────────────────────────────────────────────

def test_exercise_1():
    """StrategyLike Protocol exists and is runtime-checkable."""
    print("\n--- Exercise 1: Protocol ---")

    # Any object with name + generate_signal should satisfy the protocol
    class MinimalStrategy:
        name = "minimal"
        def generate_signal(self, df, instrument, timeframe, price):
            return {}

    assert isinstance(MinimalStrategy(), StrategyLike), \
        "MinimalStrategy should satisfy StrategyLike Protocol"

    # An object WITHOUT generate_signal should NOT satisfy it
    class NotAStrategy:
        name = "broken"

    assert not isinstance(NotAStrategy(), StrategyLike), \
        "NotAStrategy should NOT satisfy StrategyLike Protocol"

    print("  PASS: Protocol works correctly")


def test_exercise_2():
    """BaseStrategy ABC enforces abstract method."""
    print("\n--- Exercise 2: ABC enforcement ---")

    try:
        b = BaseStrategy()
        print("  FAIL: Should have raised TypeError")
    except TypeError as e:
        print(f"  PASS: TypeError raised correctly: {e}")

    # validate_data default
    class ConcreteStrategy(BaseStrategy):
        name = "concrete"
        def generate_signal(self, df, instrument, timeframe, price):
            return {}

    s = ConcreteStrategy()
    small_df = pd.DataFrame({"close": range(10)})
    large_df = pd.DataFrame({"close": range(60)})
    assert not s.validate_data(small_df), "10 rows should fail validate_data"
    assert s.validate_data(large_df),     "60 rows should pass validate_data"
    print("  PASS: validate_data default works")


def test_exercise_3():
    """SimpleMAStrategy generates correct signals."""
    print("\n--- Exercise 3: SimpleMAStrategy ---")

    import numpy as np
    strategy = SimpleMAStrategy()

    # Build a DataFrame where ema_9 > ema_21 (BUY signal)
    n = 30
    df_buy = pd.DataFrame({
        "close": np.ones(n),
        "ema_9":  np.full(n, 1.1050),
        "ema_21": np.full(n, 1.1030),
    })
    result = strategy.generate_signal(df_buy, "EUR_USD", "M5", 1.1050)
    assert result["direction"] == "BUY", f"Expected BUY, got {result['direction']}"

    # Build a DataFrame where ema_9 < ema_21 (SELL signal)
    df_sell = pd.DataFrame({
        "close": np.ones(n),
        "ema_9":  np.full(n, 1.1020),
        "ema_21": np.full(n, 1.1035),
    })
    result = strategy.generate_signal(df_sell, "EUR_USD", "M5", 1.1020)
    assert result["direction"] == "SELL", f"Expected SELL, got {result['direction']}"

    # validate_data override
    assert strategy.validate_data(pd.DataFrame({"close": range(10)})) is False
    assert strategy.validate_data(pd.DataFrame({"close": range(25)})) is True
    print("  PASS: SimpleMAStrategy signals correct")


def test_exercise_4():
    """__subclasshook__ makes AlienStrategy a virtual subclass."""
    print("\n--- Exercise 4: __subclasshook__ ---")

    result = isinstance(AlienStrategy(), BaseStrategy)
    assert result, (
        "AlienStrategy has name + generate_signal, "
        "should be recognised via __subclasshook__"
    )
    print(f"  isinstance(AlienStrategy(), BaseStrategy) = {result}")
    print("  PASS: __subclasshook__ works")


def test_exercise_5():
    """LegacyStrategy registered as virtual subclass."""
    print("\n--- Exercise 5: ABC.register() ---")

    assert issubclass(LegacyStrategy, BaseStrategy), \
        "LegacyStrategy should be registered as a BaseStrategy subclass"

    # What happens if you try to call generate_signal?
    legacy = LegacyStrategy()
    try:
        legacy.generate_signal(None, None, None, None)
        print("  NOTE: LegacyStrategy.generate_signal() didn't exist — "
              "register() does NOT enforce abstract methods!")
    except AttributeError as e:
        print(f"  CORRECT: AttributeError raised — {e}")
        print("  NOTE: register() bypasses abstract method enforcement!")

    print("  PASS: Registration works")


if __name__ == "__main__":
    print("=" * 60)
    print("Topic 01 — ABC & Protocols — Exercise Runner")
    print("=" * 60)

    # Comment out tests for exercises you haven't done yet
    try:
        test_exercise_1()
    except (NameError, AssertionError) as e:
        print(f"  INCOMPLETE: {e}")

    try:
        test_exercise_2()
    except (NameError, AssertionError) as e:
        print(f"  INCOMPLETE: {e}")

    try:
        test_exercise_3()
    except (NameError, AssertionError) as e:
        print(f"  INCOMPLETE: {e}")

    try:
        test_exercise_4()
    except (NameError, AssertionError) as e:
        print(f"  INCOMPLETE: {e}")

    try:
        test_exercise_5()
    except (NameError, AssertionError) as e:
        print(f"  INCOMPLETE: {e}")

    print("\n" + "=" * 60)
    print("Done! Compare with solutions.py when ready.")
