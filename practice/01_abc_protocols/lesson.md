# Topic 01 — Abstract Base Classes & Protocols

## The Four Interview Questions

---

### 1. WHAT is it?

An **Abstract Base Class (ABC)** is a class that cannot be instantiated directly.
It defines a *contract* — a list of methods that every subclass **must** implement.
If a subclass skips implementing even one abstract method, Python raises a `TypeError`
the moment you try to create an instance.

A **Protocol** is a structural interface (from `typing`). Any class that has the right
methods/attributes *automatically* satisfies a Protocol — no inheritance required.
This is called **structural subtyping** or "duck typing with type hints".

---

### 2. WHY does Python have them?

**Without ABCs**, nothing stops a developer from creating a strategy class that forgets
to implement `generate_signal()`. You'd only discover the bug at runtime, deep inside
the ensemble combiner.

```python
# Without ABC — the bug is silent until runtime
class BadStrategy:
    name = "bad"
    # generate_signal() completely missing!

ensemble = [BadStrategy(), RuleBasedStrategy()]
for s in ensemble:
    signal = s.generate_signal(df, ...)   # AttributeError — too late!
```

**With ABC**, the bug is caught at *instantiation*:

```python
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    @abstractmethod
    def generate_signal(self, df, instrument, timeframe, price): ...

class BadStrategy(BaseStrategy):
    name = "bad"
    # No generate_signal() implementation

s = BadStrategy()
# TypeError: Can't instantiate abstract class BadStrategy
# without implementing abstract method 'generate_signal'
```

**Protocols** solve a different problem: what if you want to accept *any* object that
behaves like a strategy — including ones from third-party libraries that can't inherit
from your ABC?

```python
from typing import Protocol
import pandas as pd

class StrategyLike(Protocol):
    name: str
    def generate_signal(self, df: pd.DataFrame, instrument: str,
                        timeframe: str, price: float): ...

def run_strategy(s: StrategyLike, df: pd.DataFrame) -> None:
    """Accepts ANY object with name + generate_signal — no inheritance needed."""
    print(f"Running {s.name}")
    signal = s.generate_signal(df, "EUR_USD", "M5", 1.1000)
```

---

### 3. WHEN do you use each?

| Situation | Use |
|-----------|-----|
| You own all subclasses, want a strict contract | `ABC` + `@abstractmethod` |
| You want to type-hint "anything that looks like X" | `Protocol` |
| You want type checking AND runtime `isinstance()` | `ABC` with `__subclasshook__` |
| Third-party code, you can't touch it | `Protocol` or `ABC.register()` |

---

### 4. SHOW ME — Annotated Code from ForexMind

```python
# forexmind/strategy/base.py (simplified)

from abc import ABC, abstractmethod
from typing import ClassVar, Protocol, runtime_checkable
import pandas as pd

# ── PROTOCOL (structural — no inheritance needed) ─────────────────────────────
@runtime_checkable  # enables isinstance() checks at runtime
class StrategyLike(Protocol):
    """Any object with name + generate_signal qualifies automatically."""
    name: str

    def generate_signal(
        self,
        df: pd.DataFrame,
        instrument: str,
        timeframe: str,
        current_price: float,
    ) -> "StrategySignal": ...


# ── ABC (nominal — must inherit) ──────────────────────────────────────────────
class BaseStrategy(ABC):
    name: ClassVar[str] = "base"   # ClassVar = shared across all instances

    @abstractmethod
    def generate_signal(self, df, instrument, timeframe, current_price): ...

    # Concrete method with a default — subclasses CAN override but don't HAVE to
    def validate_data(self, df: pd.DataFrame) -> bool:
        return len(df) >= 50      # need at least 50 rows for indicator lookback

    # Virtual subclass hook — lets "alien" classes pass isinstance() without inheriting
    @classmethod
    def __subclasshook__(cls, C):
        if cls is BaseStrategy:
            has_signal = any("generate_signal" in B.__dict__ for B in C.__mro__)
            has_name   = any("name"            in B.__dict__ for B in C.__mro__)
            if has_signal and has_name:
                return True
        return NotImplemented
```

---

## Key Concepts Deep-Dive

### `@abstractmethod` stacking

You can stack decorators on abstract methods:

```python
class Base(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...          # MUST be a property in subclass

    @classmethod
    @abstractmethod
    def from_config(cls, cfg: dict): ... # MUST be a classmethod in subclass

    @staticmethod
    @abstractmethod
    def description() -> str: ...       # MUST be a staticmethod in subclass
```

> **Interview trap**: The order matters. `@property` must come *before* `@abstractmethod`
> (outermost decorator applied last). Same rule: `@classmethod` before `@abstractmethod`.

---

### `ClassVar` vs instance variable

```python
from typing import ClassVar

class BaseStrategy(ABC):
    name: ClassVar[str] = "base"   # shared by ALL instances of the class
    call_count: int = 0            # per-instance (but defined at class level — a trap!)
```

`ClassVar` tells type checkers "this is a class variable, not an instance variable".
Without `ClassVar`, `mypy` would complain when you try to set `self.name`.

---

### `ABC.register()` — virtual subclasses

```python
class BaseStrategy(ABC):
    @abstractmethod
    def generate_signal(self, ...): ...

class ThirdPartyStrategy:           # from an external library, can't edit
    name = "external"
    def generate_signal(self, df, instrument, timeframe, price):
        return {"direction": "BUY"}

# Register it as a "virtual subclass" — no inheritance, but isinstance() returns True
BaseStrategy.register(ThirdPartyStrategy)

s = ThirdPartyStrategy()
isinstance(s, BaseStrategy)  # True
issubclass(ThirdPartyStrategy, BaseStrategy)  # True
```

> **Warning**: `register()` bypasses the abstract method check. Python trusts you.
> The registered class DOES NOT have to implement the abstract methods.

---

### `@runtime_checkable` Protocol

By default, `isinstance(obj, MyProtocol)` raises `TypeError`.
Adding `@runtime_checkable` enables it — but it only checks for attribute/method
*existence*, not signatures.

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class StrategyLike(Protocol):
    name: str
    def generate_signal(self, df, instrument, timeframe, price): ...

class Fake:
    name = "fake"
    def generate_signal(self): pass   # wrong signature — but isinstance still says True!

isinstance(Fake(), StrategyLike)  # True (only checks name + generate_signal exist)
```

---

## Common Interview Mistakes

1. **Forgetting `@runtime_checkable`** before using `isinstance()` with a Protocol
2. **Wrong decorator order** — `@property` + `@abstractmethod` (property must be outer)
3. **Thinking `register()` enforces abstract methods** — it doesn't
4. **`ClassVar` in dataclasses** — if you declare `name: ClassVar[str]` in a dataclass,
   it is excluded from `__init__` and `__repr__`
5. **ABCs are not interfaces** — they can have concrete methods with implementations

---

## Quick Reference Cheatsheet

```python
from abc import ABC, abstractmethod
from typing import ClassVar, Protocol, runtime_checkable

# Pattern 1 — Standard ABC
class MyABC(ABC):
    class_var: ClassVar[str] = "default"

    @abstractmethod
    def required_method(self): ...

    def optional_method(self):    # concrete — subclasses inherit for free
        return "default"

# Pattern 2 — Abstract property
class MyABC2(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

# Pattern 3 — Protocol for duck typing
@runtime_checkable
class MyProtocol(Protocol):
    name: str
    def required_method(self) -> int: ...

# Pattern 4 — Virtual subclass registration
MyABC.register(SomeThirdPartyClass)

# Pattern 5 — __subclasshook__ for automatic detection
class MyABC3(ABC):
    @classmethod
    def __subclasshook__(cls, C):
        if cls is MyABC3:
            if any("required_method" in B.__dict__ for B in C.__mro__):
                return True
        return NotImplemented
```
