"""
Topic 03 — Descriptors
=======================
SOLUTIONS
"""

from __future__ import annotations
from functools import cached_property


# ─────────────────────────────────────────────────────────────────────────────
# SOLUTION 1 — RangeValidator
# ─────────────────────────────────────────────────────────────────────────────

class RangeValidator:
    def __init__(self, min_val: float, max_val: float):
        self.min_val = min_val
        self.max_val = max_val

    def __set_name__(self, owner: type, name: str) -> None:
        self.public_name  = name
        self.private_name = f"_{name}"   # e.g. "_confidence"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self              # class-level access → return descriptor itself
        return getattr(obj, self.private_name, 0.0)

    def __set__(self, obj, value: float) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError(f"{self.public_name} must be numeric")
        if not (self.min_val <= value <= self.max_val):
            raise ValueError(
                f"{self.public_name} must be between {self.min_val} "
                f"and {self.max_val}, got {value}"
            )
        setattr(obj, self.private_name, float(value))


# ─────────────────────────────────────────────────────────────────────────────
# SOLUTION 2 — SignalData
# ─────────────────────────────────────────────────────────────────────────────

class SignalData:
    confidence = RangeValidator(0.0, 1.0)
    risk_pct   = RangeValidator(0.1, 5.0)

    def __init__(self, confidence: float, risk_pct: float = 1.0):
        self.confidence = confidence   # triggers __set__ → validates
        self.risk_pct   = risk_pct


# ─────────────────────────────────────────────────────────────────────────────
# SOLUTION 3 — TypeEnforced + TradeRecord
# ─────────────────────────────────────────────────────────────────────────────

class TypeEnforced:
    def __init__(self, expected_type: type):
        self.expected_type = expected_type

    def __set_name__(self, owner: type, name: str) -> None:
        self.public_name  = name
        self.private_name = f"_{name}"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.private_name, None)

    def __set__(self, obj, value) -> None:
        if not isinstance(value, self.expected_type):
            raise TypeError(
                f"{self.public_name} must be {self.expected_type.__name__}, "
                f"got {type(value).__name__}"
            )
        setattr(obj, self.private_name, value)


class TradeRecord:
    instrument:  str   = TypeEnforced(str)
    units:       int   = TypeEnforced(int)
    entry_price: float = TypeEnforced(float)

    def __init__(self, instrument: str, units: int, entry_price: float):
        self.instrument  = instrument
        self.units       = units
        self.entry_price = entry_price


# ─────────────────────────────────────────────────────────────────────────────
# SOLUTION 4 — PriceLevel with cached_property
# ─────────────────────────────────────────────────────────────────────────────

class PriceLevel:
    def __init__(self, entry: float, stop_loss: float, take_profit: float):
        self.entry       = entry
        self.stop_loss   = stop_loss
        self.take_profit = take_profit

    @cached_property
    def risk_reward(self) -> float:
        print("    [computing risk_reward...]")   # only prints on first access
        risk   = abs(self.entry - self.stop_loss)
        reward = abs(self.take_profit - self.entry)
        return reward / risk if risk > 0 else 0.0

    # NOTE: cached_property is a NON-DATA descriptor (no __set__).
    # After first access, the result is stored in instance.__dict__["risk_reward"].
    # On subsequent access, Python finds it in __dict__ BEFORE checking the descriptor.
    # This means the descriptor is bypassed — effectively a one-time cache.


# ─────────────────────────────────────────────────────────────────────────────
# SOLUTION 5 — TrackedField
# ─────────────────────────────────────────────────────────────────────────────

class TrackedField:
    """
    Data descriptor that counts read accesses per instance.
    Counts are stored in a dict on the descriptor itself, keyed by instance id.
    """

    def __set_name__(self, owner: type, name: str) -> None:
        self.public_name  = name
        self.private_name = f"_{name}"
        self._counts: dict = {}   # instance_id → read count

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        self._counts[id(obj)] = self._counts.get(id(obj), 0) + 1
        return getattr(obj, self.private_name, None)

    def __set__(self, obj, value) -> None:
        setattr(obj, self.private_name, value)

    def get_read_count(self, instance) -> int:
        return self._counts.get(id(instance), 0)


class MonitoredSignal:
    direction = TrackedField()

    def __init__(self, direction: str):
        self.direction = direction


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_exercise_1_2():
    print("\n--- Exercise 1 & 2: RangeValidator + SignalData ---")
    s1 = SignalData(0.8, 1.5)
    s2 = SignalData(0.6, 2.0)
    s1.confidence = 0.9
    assert s2.confidence == 0.6
    assert isinstance(SignalData.confidence, RangeValidator)
    try:
        s1.confidence = 1.5
    except ValueError as e:
        print(f"  PASS: {e}")
    print("  PASS")


def test_exercise_3():
    print("\n--- Exercise 3: TypeEnforced ---")
    t = TradeRecord("EUR_USD", 1000, 1.1050)
    try:
        t.units = 1000.5
    except TypeError as e:
        print(f"  PASS: {e}")
    try:
        t.instrument = 42
    except TypeError as e:
        print(f"  PASS: {e}")
    print("  PASS")


def test_exercise_4():
    print("\n--- Exercise 4: cached_property ---")
    level = PriceLevel(1.1050, 1.1020, 1.1110)
    print("  First access:")
    rr1 = level.risk_reward
    print("  Second access:")
    rr2 = level.risk_reward
    assert rr1 == rr2
    assert "risk_reward" in level.__dict__
    print(f"  risk_reward = {rr1:.4f}  PASS")


def test_exercise_5():
    print("\n--- Exercise 5: TrackedField ---")
    sig = MonitoredSignal("BUY")
    for _ in range(3):
        _ = sig.direction
    desc  = MonitoredSignal.__dict__["direction"]
    count = desc.get_read_count(sig)
    assert count == 3
    print(f"  Read count: {count}  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("Topic 03 — Descriptors — SOLUTIONS")
    print("=" * 60)
    test_exercise_1_2()
    test_exercise_3()
    test_exercise_4()
    test_exercise_5()
    print("\nAll solutions verified!")
