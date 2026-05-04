# Topic 07 — The Type System

## The Four Interview Questions

---

### 1. WHAT is it?

Python's **type system** (via the `typing` module) lets you annotate variables,
parameters, and return values with type hints. These hints are not enforced at
runtime by Python itself — they're used by static type checkers (`mypy`, `pyright`)
and IDEs for autocomplete & error detection.

---

### 2. WHY does Python have it?

Python is dynamically typed — any variable can hold anything. Type hints add
documentation, enable static analysis, and make large codebases maintainable.
For a trading system with dozens of modules, strong typing catches errors before they
reach live trades.

---

### 3. WHEN do you use each construct?

| Construct | Purpose |
|-----------|---------|
| `TypedDict` | Typed dict — like a dataclass but stays a plain dict |
| `Protocol` | Structural interface (duck typing with type checking) |
| `Generic[T]` | Parameterisable container class (`Queue[float]`) |
| `TypeVar` | Generic placeholder for consistent type inference |
| `Annotated[T, metadata]` | Attach metadata to a type hint |
| `Literal["BUY", "SELL"]` | Value must be exactly one of these strings |
| `Union[A, B]` / `A \| B` | Either type (Python 3.10+ union syntax) |
| `Optional[T]` | `T \| None` shorthand |
| `overload` | Multiple signatures for the same function |

---

### 4. SHOW ME — Annotated Examples

```python
from typing import TypedDict, Generic, TypeVar, Literal, Annotated, Protocol
from typing import overload

# ── TypedDict ─────────────────────────────────────────────────────────────────
# Like a plain dict but type-checked. Stays a dict at runtime.
class IndicatorSnapshot(TypedDict):
    instrument: str
    timeframe:  str
    ema_9:      float
    ema_21:     float
    rsi:        float
    direction:  Literal["bullish", "bearish", "choppy"]   # constrained string

snap: IndicatorSnapshot = {
    "instrument": "EUR_USD",
    "timeframe":  "M5",
    "ema_9":      1.1055,
    "ema_21":     1.1045,
    "rsi":        62.5,
    "direction":  "bullish",
}
# mypy knows snap["ema_9"] is float, snap["direction"] is one of three strings


# ── Generic[T] + TypeVar ──────────────────────────────────────────────────────
T = TypeVar("T")

class SignalQueue(Generic[T]):
    """A typed FIFO queue. SignalQueue[float] holds floats."""
    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        return self._items.pop(0)

    def __len__(self) -> int:
        return len(self._items)

price_q: SignalQueue[float] = SignalQueue()
price_q.push(1.1050)
price_q.push(1.1060)
val: float = price_q.pop()   # mypy knows this is float, not Any


# ── Annotated — metadata on types ────────────────────────────────────────────
from typing import Annotated
from dataclasses import dataclass

Pips  = Annotated[float, "unit=pips"]
Price = Annotated[float, "unit=price, precision=5"]

@dataclass
class RiskProposal:
    entry_price: Price
    stop_loss:   Price
    risk_pips:   Pips     # mypy treats these as floats, but metadata documents unit

# ── Literal ───────────────────────────────────────────────────────────────────
Direction = Literal["BUY", "SELL", "HOLD"]

def make_signal(direction: Direction) -> str:
    return f"Signal: {direction}"

make_signal("BUY")    # OK
make_signal("LONG")   # mypy error: Literal["LONG"] not assignable to Direction


# ── overload ─────────────────────────────────────────────────────────────────
@overload
def parse_price(value: str)   -> float: ...
@overload
def parse_price(value: float) -> float: ...

def parse_price(value):
    if isinstance(value, str):
        return float(value.replace(",", ""))
    return value
```

---

## Key Concepts Deep-Dive

### `TypeVar` with bounds

```python
from typing import TypeVar
from forexmind.strategy.base import BaseStrategy

# Bound: S must be BaseStrategy or a subclass
S = TypeVar("S", bound=BaseStrategy)

def clone_strategy(strategy: S) -> S:
    """Returns a copy of the same strategy type."""
    return type(strategy)()

# mypy knows clone_strategy(RuleBasedStrategy()) returns RuleBasedStrategy
```

### `TypeVar` with constraints

```python
# Constrained: ONLY int or float allowed
Numeric = TypeVar("Numeric", int, float)

def double(x: Numeric) -> Numeric:
    return x * 2
```

### `Protocol` vs `ABC` in typing

```python
# Protocol — structural: no inheritance needed
class Priceable(Protocol):
    @property
    def mid_price(self) -> float: ...

# Both of these satisfy Priceable without inheriting:
class LivePrice:
    @property
    def mid_price(self) -> float: return 1.1050

class MockPrice:
    @property
    def mid_price(self) -> float: return 1.0000
```

### `TypedDict` — required vs optional fields

```python
from typing import TypedDict, Required, NotRequired  # Python 3.11+

class SignalDict(TypedDict, total=False):   # total=False → all fields optional
    instrument: Required[str]               # but this one is required
    confidence: float
    reasoning:  str
```

---

## Quick Reference Cheatsheet

```python
from typing import (
    TypeVar, Generic, TypedDict, Protocol, Literal, Annotated,
    Optional, Union, overload, runtime_checkable
)

# TypeVar
T = TypeVar("T")
S = TypeVar("S", bound=SomeBase)

# Generic class
class Box(Generic[T]):
    def __init__(self, value: T) -> None: self.value = value
    def get(self) -> T: return self.value

# TypedDict
class MyDict(TypedDict):
    name: str
    value: float

# Literal
Status = Literal["open", "closed", "pending"]

# Annotated
Percentage = Annotated[float, "range: 0-100"]

# Union (Python 3.10+)
def foo(x: int | str) -> None: ...

# Optional
def bar(x: Optional[str] = None) -> None: ...
```
