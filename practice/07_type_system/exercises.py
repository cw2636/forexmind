"""
Topic 07 — The Type System
===========================
EXERCISES + SOLUTIONS combined (type hints are verified by mypy, not runtime tests)

Run:  python 07_type_system/exercises.py
Type check: mypy 07_type_system/exercises.py --strict
"""

from __future__ import annotations

from typing import (
    TypedDict, Generic, TypeVar, Literal, Annotated,
    Optional, Protocol, overload, runtime_checkable
)
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 1 — TypedDict
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Define `SignalDict` as a TypedDict with:
#   - instrument: str      (required)
#   - direction: Literal["BUY", "SELL", "HOLD"]
#   - confidence: float
#   - reasoning: str       (optional — use total=False trick or NotRequired)
#
# Write function `summarise(sig: SignalDict) -> str` that returns
# "{instrument}: {direction} @ {confidence:.0%}"

# TODO:
# class SignalDict(TypedDict): ...

# TODO:
# def summarise(sig: SignalDict) -> str: ...


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 2 — Generic[T] + TypeVar
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Build `RingBuffer(Generic[T])` — a fixed-size circular buffer:
#   - __init__(self, max_size: int)
#   - push(item: T) -> None   — add item; drop oldest if full
#   - latest(n: int) -> list[T] — return last n items
#   - __len__() -> int
#
# Use TypeVar T for the item type.

# TODO:
# T = TypeVar("T")
# class RingBuffer(Generic[T]): ...


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 3 — Annotated + custom metadata
# ─────────────────────────────────────────────────────────────────────────────
# Goal:
#   - Define type aliases: `Price = Annotated[float, "unit=price"]`
#                          `Pips  = Annotated[float, "unit=pips"]`
#                          `Direction = Literal["BUY", "SELL", "HOLD"]`
#
#   - Create `TradeSetup` dataclass using these annotations
#     (instrument: str, direction: Direction, entry: Price, sl_pips: Pips)
#
#   - Write `pip_value(sl_pips: Pips, units: int) -> float`
#     that returns sl_pips * units * 0.0001

# TODO


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 4 — Protocol (runtime_checkable)
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Define `Priceable` Protocol with:
#   - property `mid_price: float`
#   - method `spread() -> float`
#
# Create two classes that satisfy it WITHOUT inheriting:
#   - `LiveQuote(bid, ask)` — mid_price = (bid+ask)/2, spread = ask-bid
#   - `MockQuote(price)` — mid_price = price, spread = 0.0
#
# Verify both pass isinstance(obj, Priceable) check.

# TODO


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 5 — overload
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Write `format_price` with two overloads:
#   - format_price(value: float) -> str  (returns "1.10500")
#   - format_price(value: str)   -> float (returns float(value))
# Then implement the actual function body.

# TODO:
# @overload
# def format_price(value: float) -> str: ...
# @overload
# def format_price(value: str) -> float: ...
# def format_price(value): ...


# ─────────────────────────────────────────────────────────────────────────────
# TESTS (runtime behaviour)
# ─────────────────────────────────────────────────────────────────────────────

def test_exercise_1():
    print("\n--- Exercise 1: TypedDict ---")
    sig: SignalDict = {
        "instrument": "EUR_USD",
        "direction":  "BUY",
        "confidence": 0.75,
        "reasoning":  "EMA cross",
    }
    result = summarise(sig)
    assert result == "EUR_USD: BUY @ 75%", f"Got: {result!r}"
    print(f"  {result}  PASS")


def test_exercise_2():
    print("\n--- Exercise 2: RingBuffer ---")
    buf: RingBuffer[float] = RingBuffer(max_size=3)
    buf.push(1.0); buf.push(2.0); buf.push(3.0)
    assert len(buf) == 3
    buf.push(4.0)     # drops 1.0
    assert len(buf) == 3
    assert buf.latest(2) == [3.0, 4.0], f"Got {buf.latest(2)}"
    print(f"  latest(2) = {buf.latest(2)}  PASS")


def test_exercise_3():
    print("\n--- Exercise 3: Annotated ---")
    setup = TradeSetup("EUR_USD", "BUY", 1.1050, 30.0)
    result = pip_value(setup.sl_pips, 1000)
    assert abs(result - 3.0) < 0.001, f"Expected 3.0, got {result}"
    print(f"  pip_value={result}  PASS")


def test_exercise_4():
    print("\n--- Exercise 4: Protocol ---")
    lq = LiveQuote(bid=1.1048, ask=1.1052)
    mq = MockQuote(price=1.1050)
    assert isinstance(lq, Priceable), "LiveQuote should satisfy Priceable"
    assert isinstance(mq, Priceable), "MockQuote should satisfy Priceable"
    assert abs(lq.mid_price - 1.1050) < 0.0001
    assert abs(lq.spread()  - 0.0004) < 0.00001
    assert mq.spread() == 0.0
    print(f"  LiveQuote mid={lq.mid_price:.4f}, spread={lq.spread():.4f}  PASS")


def test_exercise_5():
    print("\n--- Exercise 5: overload ---")
    result_str = format_price(1.1050)
    assert isinstance(result_str, str), f"Expected str, got {type(result_str)}"
    result_float = format_price("1.1050")
    assert isinstance(result_float, float)
    assert abs(result_float - 1.1050) < 0.00001
    print(f"  format_price(1.1050) = {result_str!r}  PASS")
    print(f"  format_price('1.1050') = {result_float}  PASS")


# ── SOLUTIONS (uncomment to see working code) ─────────────────────────────────
"""
# Exercise 1
class SignalDict(TypedDict, total=False):
    instrument: str           # effectively required (but total=False means optional)
    direction: Literal["BUY", "SELL", "HOLD"]
    confidence: float
    reasoning: str

def summarise(sig: SignalDict) -> str:
    return f"{sig['instrument']}: {sig['direction']} @ {sig['confidence']:.0%}"

# Exercise 2
T = TypeVar("T")

class RingBuffer(Generic[T]):
    def __init__(self, max_size: int) -> None:
        self._max  = max_size
        self._data: list[T] = []

    def push(self, item: T) -> None:
        if len(self._data) >= self._max:
            self._data.pop(0)
        self._data.append(item)

    def latest(self, n: int) -> list[T]:
        return self._data[-n:]

    def __len__(self) -> int:
        return len(self._data)

# Exercise 3
Price     = Annotated[float, "unit=price"]
Pips      = Annotated[float, "unit=pips"]
Direction = Literal["BUY", "SELL", "HOLD"]

@dataclass
class TradeSetup:
    instrument: str
    direction:  Direction
    entry:      Price
    sl_pips:    Pips

def pip_value(sl_pips: Pips, units: int) -> float:
    return sl_pips * units * 0.0001

# Exercise 4
@runtime_checkable
class Priceable(Protocol):
    @property
    def mid_price(self) -> float: ...
    def spread(self) -> float: ...

class LiveQuote:
    def __init__(self, bid: float, ask: float):
        self.bid = bid; self.ask = ask
    @property
    def mid_price(self) -> float: return (self.bid + self.ask) / 2
    def spread(self) -> float: return self.ask - self.bid

class MockQuote:
    def __init__(self, price: float): self._price = price
    @property
    def mid_price(self) -> float: return self._price
    def spread(self) -> float: return 0.0

# Exercise 5
@overload
def format_price(value: float) -> str: ...
@overload
def format_price(value: str) -> float: ...
def format_price(value):
    if isinstance(value, float): return f"{value:.5f}"
    return float(value)
"""


if __name__ == "__main__":
    # Uncomment the solution block above, then run
    print("=" * 60)
    print("Topic 07 — Type System — Exercises")
    print("Implement the TODOs above, then run the tests below.")
    print("=" * 60)
    for fn in [test_exercise_1, test_exercise_2, test_exercise_3,
               test_exercise_4, test_exercise_5]:
        try:
            fn()
        except (NameError, AssertionError, TypeError) as e:
            print(f"  INCOMPLETE: {e}")
    print("\n" + "=" * 60)
