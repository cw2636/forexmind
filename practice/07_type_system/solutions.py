"""
Topic 07 — The Type System
===========================
SOLUTIONS (standalone, runnable)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Annotated, Generic, Literal, Optional, Protocol,
    TypeVar, TypedDict, overload, runtime_checkable
)


# ── Solution 1: TypedDict ─────────────────────────────────────────────────────
class SignalDict(TypedDict, total=False):
    instrument: str
    direction:  Literal["BUY", "SELL", "HOLD"]
    confidence: float
    reasoning:  str

def summarise(sig: SignalDict) -> str:
    return f"{sig['instrument']}: {sig['direction']} @ {sig['confidence']:.0%}"


# ── Solution 2: Generic[T] RingBuffer ────────────────────────────────────────
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


# ── Solution 3: Annotated ────────────────────────────────────────────────────
Price     = Annotated[float, "unit=price"]
Pips      = Annotated[float, "unit=pips"]
Direction = Literal["BUY", "SELL", "HOLD"]

@dataclass
class TradeSetup:
    instrument: str
    direction:  Direction
    entry:      Price
    sl_pips:    Pips

def pip_value(sl_pips: "Pips", units: int) -> float:
    return sl_pips * units * 0.0001


# ── Solution 4: Protocol ──────────────────────────────────────────────────────
@runtime_checkable
class Priceable(Protocol):
    @property
    def mid_price(self) -> float: ...
    def spread(self) -> float: ...

class LiveQuote:
    def __init__(self, bid: float, ask: float):
        self.bid = bid
        self.ask = ask
    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2
    def spread(self) -> float:
        return self.ask - self.bid

class MockQuote:
    def __init__(self, price: float):
        self._price = price
    @property
    def mid_price(self) -> float:
        return self._price
    def spread(self) -> float:
        return 0.0


# ── Solution 5: overload ──────────────────────────────────────────────────────
@overload
def format_price(value: float) -> str: ...
@overload
def format_price(value: str)   -> float: ...
def format_price(value):
    if isinstance(value, float):
        return f"{value:.5f}"
    return float(value)


# ── Tests ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Topic 07 — Type System — SOLUTIONS")
    print("=" * 60)

    sig: SignalDict = {"instrument": "EUR_USD", "direction": "BUY", "confidence": 0.75}
    assert summarise(sig) == "EUR_USD: BUY @ 75%"
    print("\n  Ex1 TypedDict: PASS")

    buf: RingBuffer[float] = RingBuffer(3)
    buf.push(1.0); buf.push(2.0); buf.push(3.0); buf.push(4.0)
    assert buf.latest(2) == [3.0, 4.0]
    print("  Ex2 RingBuffer: PASS")

    setup = TradeSetup("EUR_USD", "BUY", 1.1050, 30.0)
    assert abs(pip_value(setup.sl_pips, 1000) - 3.0) < 0.001
    print("  Ex3 Annotated: PASS")

    lq, mq = LiveQuote(1.1048, 1.1052), MockQuote(1.1050)
    assert isinstance(lq, Priceable) and isinstance(mq, Priceable)
    print("  Ex4 Protocol: PASS")

    assert format_price(1.1050) == "1.10500"
    assert format_price("1.1050") == 1.1050
    print("  Ex5 overload: PASS")

    print("\nAll solutions verified!")
