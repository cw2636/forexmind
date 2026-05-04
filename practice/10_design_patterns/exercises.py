"""
Topic 10 — Design Patterns
============================
EXERCISES + SOLUTIONS

Run: python 10_design_patterns/exercises.py
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable


# ── EXERCISE 1: Strategy Pattern ─────────────────────────────────────────────
# Build a `PositionSizer` that accepts any `SizingStrategy` and delegates to it.
# Strategies: FixedRisk(risk_pct, balance) and KellyHalf(win_rate, avg_win, avg_loss)
#
# TODO


# ── EXERCISE 2: Observer / Event Bus ─────────────────────────────────────────
# Build EventBus with subscribe(event, callback) and publish(event, **data).
# Demonstrate with "new_signal" event notifying two log callbacks.
#
# TODO


# ── EXERCISE 3: Factory ───────────────────────────────────────────────────────
# Build StrategyFactory.create(name) backed by a dict of simple strategy classes.
# If name unknown, raise ValueError with list of valid options.
#
# TODO


# ── EXERCISE 4: Command Pattern ───────────────────────────────────────────────
# Build trade commands: PlaceTrade(pair, units) and CloseTrade(trade_id).
# Both have execute() and undo(). Maintain a command history and undoStack.
# Write `TradeHistory` that can execute() and undo_last().
#
# TODO


# ── EXERCISE 5: Template Method ───────────────────────────────────────────────
# Build `AnalysisPipeline` ABC with run() = fetch_data() → compute_indicators()
#   → generate_signal() (abstract) → log_result().
# Create `MACDPipeline` that implements generate_signal.
#
# TODO


# ─────────────────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_exercise_1():
    print("\n--- Exercise 1: Strategy ---")
    sizer = PositionSizer(FixedRisk())
    fixed = sizer.size(risk_pct=2.0, balance=10000)
    assert fixed == 200.0, f"Expected 200, got {fixed}"
    sizer.set_strategy(KellyHalf(win_rate=0.6, avg_win=100, avg_loss=50))
    kelly = sizer.size(risk_pct=0, balance=10000)  # kelly ignores risk_pct
    assert kelly > 0
    print(f"  fixed={fixed}, kelly={kelly:.2f}  PASS")


def test_exercise_2():
    print("\n--- Exercise 2: EventBus ---")
    bus   = EventBus()
    log   = []
    bus.subscribe("new_signal", lambda direction, pair: log.append((direction, pair)))
    bus.subscribe("new_signal", lambda direction, pair: log.append("AUDIT"))
    bus.publish("new_signal", direction="BUY", pair="EUR_USD")
    assert len(log) == 2
    assert ("BUY", "EUR_USD") in log
    assert "AUDIT" in log
    print(f"  log={log}  PASS")


def test_exercise_3():
    print("\n--- Exercise 3: Factory ---")
    s1 = StrategyFactory.create("rule_based")
    s2 = StrategyFactory.create("ml")
    assert s1 is not s2
    try:
        StrategyFactory.create("unknown")
    except ValueError as e:
        print(f"  PASS: {e}")


def test_exercise_4():
    print("\n--- Exercise 4: Command ---")
    history = TradeHistory()
    history.execute(PlaceTrade("EUR_USD", 1000))
    history.execute(PlaceTrade("GBP_USD", 500))
    assert len(history.done) == 2
    history.undo_last()
    assert len(history.done) == 1
    print(f"  Commands: {[str(c) for c in history.done]}  PASS")


def test_exercise_5():
    print("\n--- Exercise 5: Template Method ---")
    pipeline = MACDPipeline(pair="EUR_USD")
    result   = pipeline.run()
    assert result["direction"] in ("BUY", "SELL", "HOLD")
    print(f"  signal={result}  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("Topic 10 — Design Patterns — Exercise Runner")
    print("=" * 60)
    for fn in [test_exercise_1, test_exercise_2, test_exercise_3,
               test_exercise_4, test_exercise_5]:
        try:
            fn()
        except (NameError, AssertionError, TypeError) as e:
            print(f"  INCOMPLETE: {e}")
    print("\n" + "=" * 60)


# ── SOLUTIONS ─────────────────────────────────────────────────────────────────
"""
# Ex1
class SizingStrategy(ABC):
    @abstractmethod
    def calculate(self, risk_pct: float, balance: float) -> float: ...

class FixedRisk(SizingStrategy):
    def calculate(self, risk_pct, balance):
        return balance * (risk_pct / 100)

class KellyHalf(SizingStrategy):
    def __init__(self, win_rate, avg_win, avg_loss):
        self.win_rate = win_rate
        self.avg_win  = avg_win
        self.avg_loss = avg_loss
    def calculate(self, risk_pct, balance):
        kelly = self.win_rate - (1 - self.win_rate) * (self.avg_loss / self.avg_win)
        return balance * max(0, kelly) * 0.5

class PositionSizer:
    def __init__(self, strategy: SizingStrategy):
        self._strategy = strategy
    def set_strategy(self, strategy): self._strategy = strategy
    def size(self, risk_pct, balance): return self._strategy.calculate(risk_pct, balance)

# Ex2
class EventBus:
    def __init__(self): self._listeners: dict = {}
    def subscribe(self, event, cb):
        self._listeners.setdefault(event, []).append(cb)
    def publish(self, event, **data):
        for cb in self._listeners.get(event, []): cb(**data)

# Ex3
class _RuleBased:
    name = "rule_based"
class _ML:
    name = "ml"
class StrategyFactory:
    _reg = {"rule_based": _RuleBased, "ml": _ML}
    @classmethod
    def create(cls, name):
        if name not in cls._reg: raise ValueError(f"Unknown: {name!r}. Valid: {list(cls._reg)}")
        return cls._reg[name]()

# Ex4
class Command(ABC):
    @abstractmethod
    def execute(self): ...
    @abstractmethod
    def undo(self): ...

class PlaceTrade(Command):
    def __init__(self, pair, units): self.pair = pair; self.units = units
    def execute(self): print(f"    Place {self.pair} x{self.units}")
    def undo(self):    print(f"    Cancel {self.pair} x{self.units}")
    def __str__(self): return f"Place({self.pair})"

class CloseTrade(Command):
    def __init__(self, trade_id): self.trade_id = trade_id
    def execute(self): print(f"    Close {self.trade_id}")
    def undo(self):    print(f"    Reopen {self.trade_id}")
    def __str__(self): return f"Close({self.trade_id})"

class TradeHistory:
    def __init__(self): self.done = []
    def execute(self, cmd):
        cmd.execute()
        self.done.append(cmd)
    def undo_last(self):
        if self.done:
            cmd = self.done.pop()
            cmd.undo()

# Ex5
class AnalysisPipeline(ABC):
    def __init__(self, pair): self.pair = pair; self._data = []; self._indicators = {}
    def run(self):
        self._data = self.fetch_data()
        self._indicators = self.compute_indicators(self._data)
        result = self.generate_signal(self._indicators)
        self.log_result(result)
        return result
    def fetch_data(self): return [1.1050, 1.1060, 1.1055]
    def compute_indicators(self, data): return {"macd": data[-1]-data[0]}
    @abstractmethod
    def generate_signal(self, indicators): ...
    def log_result(self, result): print(f"    Signal: {result}")

class MACDPipeline(AnalysisPipeline):
    def generate_signal(self, indicators):
        return {"direction": "BUY" if indicators["macd"] > 0 else "SELL"}
"""
