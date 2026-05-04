# Topic 02 — Dataclasses Deep-Dive

## The Four Interview Questions

---

### 1. WHAT is it?

`@dataclass` is a decorator that auto-generates boilerplate methods (`__init__`,
`__repr__`, `__eq__`) from class-level field annotations. Instead of writing
20 lines of constructor code, you describe your data and Python generates it.

```python
# Without @dataclass — 15 lines of boilerplate
class StrategySignal:
    def __init__(self, instrument, direction, confidence, entry_price):
        self.instrument  = instrument
        self.direction   = direction
        self.confidence  = confidence
        self.entry_price = entry_price
    def __repr__(self):
        return f"StrategySignal(instrument={self.instrument!r}, ...)"
    def __eq__(self, other):
        return (self.instrument == other.instrument and ...)

# With @dataclass — 6 lines, same result
from dataclasses import dataclass

@dataclass
class StrategySignal:
    instrument:  str
    direction:   str
    confidence:  float
    entry_price: float
```

---

### 2. WHY does Python have it?

Data classes were added in Python 3.7 (PEP 557) to reduce the friction of writing
plain data-holding objects. Before that, developers used `namedtuple`, `dict`,
`attr`, or hand-written classes. Data classes give you the expressiveness of a
regular class with the convenience of auto-generated boilerplate.

---

### 3. WHEN do you use each variant?

| Variant | When to use |
|---------|-------------|
| `@dataclass` | Default — mutable, general purpose |
| `@dataclass(frozen=True)` | Immutable data (usable as dict keys, in sets) |
| `@dataclass(order=True)` | Need `<`, `>`, `<=`, `>=` comparison |
| `@dataclass(slots=True)` | High-performance (Python 3.10+), many instances |
| `__post_init__` | Validation or derived field computation after `__init__` |
| `field(default_factory=...)` | Mutable defaults (lists, dicts, datetimes) |
| `field(init=False)` | Fields computed from others, not set by caller |
| `field(repr=False)` | Exclude sensitive data from `__repr__` |
| `ClassVar` | Class-level constant — excluded from `__init__` entirely |

---

### 4. SHOW ME — Annotated Code from ForexMind

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import ClassVar, Optional

@dataclass
class StrategySignal:
    # ── Required fields (no default) — must be passed to __init__ ────────────
    instrument:  str
    timeframe:   str
    direction:   str       # "BUY" | "SELL" | "HOLD"
    confidence:  float     # 0.0–1.0
    entry_price: float

    # ── Optional fields with defaults ────────────────────────────────────────
    stop_loss:   float = 0.0
    take_profit: float = 0.0
    risk_pct:    float = 1.0
    reasoning:   str   = ""
    source:      str   = "unknown"

    # ── Mutable default — MUST use field(default_factory=...) ────────────────
    # Never write:  tags: list = []   ← shared across ALL instances (bug!)
    tags: list = field(default_factory=list)

    # ── Auto-set field — caller cannot pass this ─────────────────────────────
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # ── Excluded from __repr__ (security concern) ────────────────────────────
    _raw_score: float = field(default=0.0, repr=False)

    # ── ClassVar — NOT a dataclass field, excluded from __init__ ─────────────
    VERSION: ClassVar[str] = "1.0"

    # ── Post-init validation ──────────────────────────────────────────────────
    def __post_init__(self):
        # Called automatically after __init__ finishes
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be 0–1, got {self.confidence}")
        if self.direction not in ("BUY", "SELL", "HOLD"):
            raise ValueError(f"Invalid direction: {self.direction!r}")
        # Normalise instrument: EUR/USD → EUR_USD
        self.instrument = self.instrument.replace("/", "_")

    # ── Computed property (not a dataclass field) ────────────────────────────
    @property
    def risk_reward(self) -> float:
        risk   = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.entry_price)
        return reward / risk if risk > 0 else 0.0
```

---

## Key Concepts Deep-Dive

### The Mutable Default Trap

This is the #1 dataclass interview gotcha:

```python
@dataclass
class Bad:
    items: list = []          # SyntaxError in modern Python!
                               # (was a silent bug in older versions)

@dataclass
class Good:
    items: list = field(default_factory=list)   # correct: new list per instance
```

Why is `[]` dangerous? In a regular class:

```python
class Shared:
    items = []         # CLASS variable — ALL instances share it!

a = Shared()
b = Shared()
a.items.append(1)
print(b.items)        # [1]  ← b sees it too! Bug!
```

`field(default_factory=list)` calls `list()` fresh for every new instance.

---

### `__post_init__` — The Validator

`__post_init__` is called by the auto-generated `__init__` *after* all fields are set.
Use it for:
- Field validation
- Derived/computed fields based on other fields
- Normalisation (e.g. `EUR/USD` → `EUR_USD`)

```python
@dataclass
class RiskProposal:
    entry_price: float
    stop_loss:   float
    take_profit: float
    risk_reward_ratio: float = field(init=False)   # computed, not passed in

    def __post_init__(self):
        risk   = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.entry_price)
        self.risk_reward_ratio = reward / risk if risk > 0 else 0.0
        if self.risk_reward_ratio < 1.5:
            raise ValueError(f"R:R {self.risk_reward_ratio:.2f} too low (min 1.5)")
```

---

### `frozen=True` — Immutable Dataclasses

```python
@dataclass(frozen=True)
class TradingPair:
    base:  str    # e.g. "EUR"
    quote: str    # e.g. "USD"

    @property
    def symbol(self) -> str:
        return f"{self.base}_{self.quote}"

pair = TradingPair("EUR", "USD")
pair.base = "GBP"    # FrozenInstanceError: cannot assign to field 'base'

# Because it's frozen, it's hashable — can be used as dict key or in sets
cache = {pair: 1.1050}
seen  = {pair}
```

---

### `__slots__` in Dataclasses (Python 3.10+)

```python
@dataclass(slots=True)   # Python 3.10+
class StrategySignal:
    instrument: str
    direction:  str
    confidence: float
```

`__slots__` prevents adding arbitrary attributes and reduces memory by ~40%
per instance. Critical when you're creating millions of signals.

For Python < 3.10, you can manually combine:

```python
@dataclass
class StrategySignal:
    __slots__ = ("instrument", "direction", "confidence")
    instrument: str
    direction:  str
    confidence: float
    # Note: default values don't work with manual __slots__ + @dataclass
    # Use slots=True in Python 3.10+ instead
```

---

### `field()` Complete Reference

```python
from dataclasses import field

@dataclass
class Example:
    # default       — fixed default value
    name: str = field(default="unknown")

    # default_factory — callable that returns a fresh default each time
    items: list = field(default_factory=list)

    # init=False    — excluded from __init__, must be set in __post_init__
    computed: float = field(init=False, default=0.0)

    # repr=False    — excluded from __repr__ (hide sensitive data)
    api_key: str = field(default="", repr=False)

    # compare=False — excluded from __eq__ and ordering comparisons
    cache: dict = field(default_factory=dict, compare=False)

    # hash=None     — inherits from compare by default
    # metadata      — read-only mapping for third-party tools (e.g. marshmallow)
    amount: float = field(default=0.0, metadata={"unit": "USD", "precision": 2})
```

---

## Common Interview Mistakes

1. **Using mutable defaults** (`list`, `dict`) instead of `field(default_factory=...)`
2. **Setting `field(init=False)` fields in `__init__` arguments** — they can't be passed
3. **Forgetting that `frozen=True` makes the dataclass hashable** (often asked as a follow-up)
4. **`ClassVar` fields ARE excluded from `__init__`** — forgetting this causes confusion
5. **`__post_init__` with `InitVar`** — advanced: passing extra args to `__post_init__` that aren't stored as fields

```python
from dataclasses import dataclass, InitVar, field

@dataclass
class Price:
    raw_value: float
    precision: InitVar[int] = 5          # passed to __post_init__ but NOT stored

    value: str = field(init=False)

    def __post_init__(self, precision: int):
        self.value = f"{self.raw_value:.{precision}f}"

p = Price(1.10503, precision=4)
print(p.value)   # "1.1050"
print(p.raw_value)  # 1.10503
# p.precision → AttributeError (not stored — that's the point)
```

---

## Quick Reference Cheatsheet

```python
from dataclasses import dataclass, field, fields, asdict, astuple
from typing import ClassVar

@dataclass(
    order=True,          # enables <, >, <=, >=
    frozen=False,        # True = immutable, hashable
    slots=True,          # Python 3.10+ — memory efficiency
)
class MyData:
    required_field: str                          # no default
    with_default:   int  = 0                     # scalar default OK
    list_field:     list = field(default_factory=list)   # mutable default
    computed:       float = field(init=False)     # set in __post_init__
    hidden:        str   = field(default="", repr=False)
    class_const:   ClassVar[str] = "v1"          # excluded from init entirely

    def __post_init__(self):
        self.computed = self.required_field      # set init=False fields here

# Introspection helpers
fields(MyData)           # tuple of Field objects (name, type, default, ...)
asdict(instance)         # recursively convert to dict
astuple(instance)        # recursively convert to tuple
```
