"""
Topic 03 — Descriptors
=======================
EXERCISES — Fill in every section marked TODO.

Run: python 03_descriptors/exercises.py
"""

from __future__ import annotations
from functools import cached_property


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 1 — Build a RangeValidator descriptor
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Create `RangeValidator` that:
#   - Takes min_val and max_val in __init__
#   - Uses __set_name__ to learn its own attribute name
#   - Stores the value in the INSTANCE dict (key = "_<attr_name>")
#   - __get__: returns the stored value (or 0.0 if not yet set)
#   - __set__: validates min_val <= value <= max_val, raises ValueError if not
#   - When accessed from the CLASS (obj is None), returns the descriptor itself

# TODO: Define RangeValidator


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 2 — Apply descriptor to a class
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Create `SignalData` class (not a dataclass — plain class) with:
#   - confidence = RangeValidator(0.0, 1.0)
#   - risk_pct   = RangeValidator(0.1, 5.0)
#   - __init__(self, confidence, risk_pct=1.0)
#
# Demonstrate that two instances have INDEPENDENT values (not shared).

# TODO: Define SignalData


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 3 — TypeEnforced descriptor
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Create `TypeEnforced` descriptor that:
#   - Takes expected_type in __init__ (e.g. str, int, float)
#   - Raises TypeError if value is not an instance of expected_type
#   - Uses __set_name__
#
# Then create `TradeRecord` using TypeEnforced:
#   - instrument:  TypeEnforced(str)
#   - units:       TypeEnforced(int)
#   - entry_price: TypeEnforced(float)

# TODO: Define TypeEnforced

# TODO: Define TradeRecord


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 4 — cached_property (non-data descriptor pattern)
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Create `PriceLevel` class with:
#   - __init__(self, entry, stop_loss, take_profit)
#   - A @cached_property `risk_reward` that computes:
#       abs(take_profit - entry) / abs(entry - stop_loss)  (or 0.0 if SL == entry)
#   - Print a message inside the computation so you can see it only runs ONCE
#
# Verify that accessing .risk_reward twice only prints the message once.

# TODO: Define PriceLevel


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 5 — Descriptor that tracks access count (advanced)
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Create `TrackedField` descriptor that:
#   - Stores the value normally
#   - Also counts how many times the field has been READ (per instance)
#   - Provides a method get_read_count(instance) on the descriptor object
#
# Then use it in a `MonitoredSignal` class with a `direction` field.
# After reading direction 3 times, get_read_count should return 3.

# TODO: Define TrackedField and MonitoredSignal


# ─────────────────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_exercise_1_2():
    print("\n--- Exercise 1 & 2: RangeValidator + SignalData ---")

    s1 = SignalData(confidence=0.8, risk_pct=1.5)
    s2 = SignalData(confidence=0.6, risk_pct=2.0)

    assert s1.confidence == 0.8
    assert s2.confidence == 0.6

    # Independence — changing s1 doesn't affect s2
    s1.confidence = 0.9
    assert s2.confidence == 0.6, "Instances must have independent values!"
    print(f"  s1.confidence={s1.confidence}, s2.confidence={s2.confidence}  PASS")

    # Class-level access returns descriptor
    assert isinstance(SignalData.confidence, RangeValidator), \
        "Class-level access should return the descriptor object"
    print("  Class-level access returns descriptor  PASS")

    # Validation
    try:
        s1.confidence = 1.5
        print("  FAIL: should raise ValueError")
    except ValueError as e:
        print(f"  PASS: ValueError: {e}")


def test_exercise_3():
    print("\n--- Exercise 3: TypeEnforced ---")

    t = TradeRecord(instrument="EUR_USD", units=1000, entry_price=1.1050)
    assert t.instrument == "EUR_USD"

    try:
        t.units = 1000.5   # float, not int
        print("  FAIL: should raise TypeError")
    except TypeError as e:
        print(f"  PASS: TypeError: {e}")

    try:
        t.instrument = 42   # int, not str
        print("  FAIL: should raise TypeError")
    except TypeError as e:
        print(f"  PASS: TypeError: {e}")


def test_exercise_4():
    print("\n--- Exercise 4: cached_property ---")

    level = PriceLevel(entry=1.1050, stop_loss=1.1020, take_profit=1.1110)
    print("  First access:")
    rr1 = level.risk_reward
    print("  Second access (should NOT recompute):")
    rr2 = level.risk_reward
    assert rr1 == rr2
    assert "risk_reward" in level.__dict__, \
        "cached_property should store result in instance __dict__"
    print(f"  risk_reward = {rr1:.4f}  PASS")


def test_exercise_5():
    print("\n--- Exercise 5: TrackedField ---")

    sig = MonitoredSignal(direction="BUY")
    _ = sig.direction   # read 1
    _ = sig.direction   # read 2
    _ = sig.direction   # read 3

    desc  = MonitoredSignal.__dict__["direction"]   # get descriptor object
    count = desc.get_read_count(sig)
    assert count == 3, f"Expected 3 reads, got {count}"
    print(f"  Read count: {count}  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("Topic 03 — Descriptors — Exercise Runner")
    print("=" * 60)

    for fn in [test_exercise_1_2, test_exercise_3, test_exercise_4, test_exercise_5]:
        try:
            fn()
        except (NameError, AssertionError, TypeError) as e:
            print(f"  INCOMPLETE: {e}")

    print("\n" + "=" * 60)
