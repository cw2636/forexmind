# Topic 10 — Design Patterns

## The Four Interview Questions

---

### 1. WHAT is it?

Design patterns are **proven, named solutions** to recurring software design problems.
They're not code — they're templates. Three categories:
- **Creational**: how objects are created (Factory, Singleton, Builder)
- **Structural**: how objects are composed (Adapter, Composite, Decorator)
- **Behavioural**: how objects communicate (Strategy, Observer, Command)

---

### 2. WHY does Python have them?

Python's first-class functions and dynamic typing make many GoF patterns trivial or
built-in (e.g. Decorator = `@decorator`). But several patterns are still valuable:
- **Strategy** — swap algorithms at runtime (our ensemble does this!)
- **Observer/Event Bus** — decouple components (price update → indicators → signal → risk)
- **Factory** — create objects without knowing their concrete type

---

### 3. Key Patterns for Trading Systems

| Pattern | ForexMind use |
|---------|---------------|
| **Strategy** | `EnsembleStrategy` switches between RuleBased/ML/RL |
| **Observer** | Price ticks notify indicator engine, which notifies signal generator |
| **Factory** | `StrategyFactory.create("ml")` returns correct strategy class |
| **Command** | Each trade = a command object (execute, undo, replay) |
| **Template Method** | `BaseStrategy.generate_signal` is the template |

---

### 4. SHOW ME — Annotated Examples

```python
# ── Strategy Pattern ─────────────────────────────────────────────────────────
from abc import ABC, abstractmethod

class PricingStrategy(ABC):
    @abstractmethod
    def calculate(self, risk_pct: float, balance: float) -> float: ...

class KellyCriterion(PricingStrategy):
    def calculate(self, risk_pct, balance):
        return balance * risk_pct * 0.5   # half-Kelly

class FixedRisk(PricingStrategy):
    def calculate(self, risk_pct, balance):
        return balance * (risk_pct / 100)

class PositionSizer:
    def __init__(self, strategy: PricingStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: PricingStrategy):
        self._strategy = strategy           # swap at runtime!

    def size(self, risk_pct: float, balance: float) -> float:
        return self._strategy.calculate(risk_pct, balance)

sizer = PositionSizer(KellyCriterion())
units = sizer.size(2.0, 10000)
sizer.set_strategy(FixedRisk())     # switch strategy
units = sizer.size(2.0, 10000)


# ── Observer / Event Bus ──────────────────────────────────────────────────────
from typing import Callable

class EventBus:
    """
    Minimal publish-subscribe event bus.
    Components subscribe to event names; when event fires, all callbacks run.
    """
    def __init__(self):
        self._listeners: dict[str, list[Callable]] = {}

    def subscribe(self, event: str, callback: Callable) -> None:
        self._listeners.setdefault(event, []).append(callback)

    def publish(self, event: str, **data) -> None:
        for cb in self._listeners.get(event, []):
            cb(**data)

bus = EventBus()
bus.subscribe("price_tick", lambda pair, price: print(f"Indicator: {pair} {price}"))
bus.subscribe("price_tick", lambda pair, price: print(f"Risk:  {pair} {price}"))
bus.publish("price_tick", pair="EUR_USD", price=1.1053)


# ── Factory Pattern ──────────────────────────────────────────────────────────
class StrategyFactory:
    _registry = {
        "rule_based": RuleBasedStrategy,
        "ml":         MLStrategy,
        "rl":         RLStrategy,
    }

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseStrategy:
        if name not in cls._registry:
            raise ValueError(f"Unknown strategy: {name!r}")
        return cls._registry[name](**kwargs)

strategy = StrategyFactory.create("ml")
```

---

## Quick Reference Cheatsheet

```python
# Strategy
class Context:
    def __init__(self, strategy): self._s = strategy
    def execute(self): return self._s.algorithm()

# Observer
class Event:
    def __init__(self): self._cbs = []
    def subscribe(self, cb): self._cbs.append(cb)
    def fire(self, **kw): [cb(**kw) for cb in self._cbs]

# Factory method
class Factory:
    @classmethod
    def create(cls, kind): return {"a": A, "b": B}[kind]()

# Command
class Command(ABC):
    @abstractmethod
    def execute(self): ...
    @abstractmethod
    def undo(self): ...

# Template Method
class Template(ABC):
    def run(self):          # fixed order
        self.setup()
        self.process()      # abstract
        self.teardown()
    def setup(self): pass
    def teardown(self): pass
    @abstractmethod
    def process(self): ...
```
