"""
Topic 04 — Decorators
======================
EXERCISES — Fill in every section marked TODO.

Run: python 04_decorators/exercises.py
"""

from __future__ import annotations

import functools
import time


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 1 — Simple timing decorator
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Write `timer` decorator that:
#   - Prints "⏱  <func_name> took X.XXXXs"
#   - Uses functools.wraps
#   - Returns the original return value unchanged
#
# Test: Apply it to `slow_indicator()` below and verify timing output.

# TODO: def timer(func): ...

def slow_indicator(pair: str, period: int = 14) -> float:
    """Simulates a slow indicator calculation."""
    time.sleep(0.05)
    return 42.0


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 2 — Parametrized retry decorator
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Write `retry(max_attempts=3, exceptions=(Exception,))` that:
#   - Retries the function up to max_attempts times
#   - Only catches exceptions listed in `exceptions` tuple
#   - On final failure, re-raises the last exception
#   - Uses functools.wraps
#
# Test: Apply @retry(max_attempts=3) to flaky_api() below.

call_count = 0   # global counter to simulate failure then success

def flaky_api() -> str:
    """Fails the first 2 times, succeeds on 3rd call."""
    global call_count
    call_count += 1
    if call_count < 3:
        raise ConnectionError(f"Connection failed (attempt {call_count})")
    return "OK"

# TODO: def retry(max_attempts=3, exceptions=(Exception,)): ...
# Then decorate flaky_api with @retry(max_attempts=3)


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 3 — Class-based decorator with state
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Write `CallCounter` class-based decorator that:
#   - Wraps any function
#   - Tracks `.count` (how many times called)
#   - Tracks `.last_result` (last return value)
#   - Uses functools.update_wrapper
#
# Apply it to `generate_signal()` below.

# TODO: class CallCounter: ...

def generate_signal(pair: str) -> str:
    return "BUY" if "EUR" in pair else "SELL"


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 4 — Stacking decorators
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Write `log_calls` decorator that:
#   - Prints "→ calling <func_name>(<args>)"  BEFORE the call
#   - Prints "← <func_name> returned <result>" AFTER the call
#
# Then apply BOTH @timer AND @log_calls to `compute_rsi()` below, so that:
#   - The outer wrapper logs calls
#   - The inner wrapper times the execution
# (i.e. log_calls should be the OUTER decorator)

# TODO: def log_calls(func): ...

# TODO: Apply @log_calls and @timer (correct order!) to this function:
def compute_rsi(prices: list, period: int = 14) -> float:
    """Compute simplified RSI."""
    time.sleep(0.02)
    return 55.5


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 5 — Dual-mode decorator (optional args)
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Write `validate_pair` that works BOTH ways:
#   @validate_pair           (no parentheses)
#   @validate_pair(valid=["EUR_USD", "GBP_USD"])
#
# Behaviour: before calling the function, check that the first argument
# (assumed to be `pair: str`) is in the valid list. Raise ValueError if not.
# Default valid list: ["EUR_USD", "GBP_USD", "USD_JPY"]
#
# Hint: Use the _func=None pattern from the lesson.

# TODO: def validate_pair(_func=None, *, valid=None): ...

# TODO: Apply @validate_pair to price_lookup() (no args form)
def price_lookup(pair: str) -> float:
    return 1.1050

# TODO: Apply @validate_pair(valid=["EUR_USD"]) to restricted_lookup()
def restricted_lookup(pair: str) -> float:
    return 1.1050


# ─────────────────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_exercise_1():
    print("\n--- Exercise 1: timer ---")
    result = slow_indicator("EUR_USD")
    assert result == 42.0
    assert slow_indicator.__name__ == "slow_indicator", \
        "functools.wraps must preserve __name__"
    print(f"  result={result}  PASS")


def test_exercise_2():
    print("\n--- Exercise 2: retry ---")
    global call_count
    call_count = 0
    result = flaky_api()
    assert result == "OK", f"Expected 'OK', got {result!r}"
    assert call_count == 3, f"Expected 3 attempts, got {call_count}"
    print(f"  Succeeded after {call_count} attempts  PASS")

    # Test that it eventually re-raises
    def always_fails():
        raise ValueError("always")

    @retry(max_attempts=2, exceptions=(ValueError,))
    def wrapped_fail():
        return always_fails()

    try:
        wrapped_fail()
        print("  FAIL: should have re-raised")
    except ValueError:
        print("  PASS: re-raises after max_attempts")


def test_exercise_3():
    print("\n--- Exercise 3: CallCounter ---")
    generate_signal("EUR_USD")
    generate_signal("GBP_USD")
    generate_signal("EUR_USD")
    assert generate_signal.count == 3
    assert generate_signal.last_result in ("BUY", "SELL")
    assert generate_signal.__name__ == "generate_signal"
    print(f"  count={generate_signal.count}, last={generate_signal.last_result}  PASS")


def test_exercise_4():
    print("\n--- Exercise 4: stacked decorators ---")
    result = compute_rsi([1, 2, 3, 4], period=14)
    assert result == 55.5
    assert compute_rsi.__name__ == "compute_rsi"
    print(f"  result={result}  PASS")


def test_exercise_5():
    print("\n--- Exercise 5: dual-mode validate_pair ---")

    # No-args form
    result = price_lookup("EUR_USD")
    assert result == 1.1050
    try:
        price_lookup("XXX_YYY")
        print("  FAIL: should raise ValueError")
    except ValueError as e:
        print(f"  PASS (no-args form): {e}")

    # With-args form
    result = restricted_lookup("EUR_USD")
    assert result == 1.1050
    try:
        restricted_lookup("GBP_USD")
        print("  FAIL: should raise ValueError")
    except ValueError as e:
        print(f"  PASS (with-args form): {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Topic 04 — Decorators — Exercise Runner")
    print("=" * 60)

    for fn in [test_exercise_1, test_exercise_2, test_exercise_3,
               test_exercise_4, test_exercise_5]:
        try:
            fn()
        except (NameError, AssertionError, TypeError) as e:
            print(f"  INCOMPLETE: {e}")

    print("\n" + "=" * 60)
