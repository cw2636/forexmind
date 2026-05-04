"""
Topic 06 — Generators & Iterators
===================================
EXERCISES — Fill in every section marked TODO.

Run: python 06_generators/exercises.py
"""

from __future__ import annotations
import asyncio


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 1 — Basic generator + generator expression
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Write `candle_generator(prices: list[float], window: int)` that:
#   - Yields sliding windows of length `window` as tuples
#   - Example: prices=[1,2,3,4,5], window=3 → (1,2,3), (2,3,4), (3,4,5)
#
# Also write a generator expression `log_returns` that given a list of closes,
# yields (closes[i]/closes[i-1] - 1) for i in 1..n

# TODO: def candle_generator(prices, window): ...

# TODO: create log_returns as a generator expression (fill in below)
closes = [1.1050, 1.1060, 1.1055, 1.1045, 1.1070]
# log_returns = (... for i in range(1, len(closes)))


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 2 — yield from + pipeline
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Write three generator functions that form a pipeline:
#   1. `raw_ticks(pairs: list)` — yields (pair, price) tuples for each pair
#      (just use enumerate to make a simple price like 1.0 + i*0.01)
#   2. `filter_major(ticks)` — yields only ticks where pair starts with "EUR"
#   3. `format_tick(ticks)` — yields string "EUR_USD @ 1.0050" for each tick
#
# Then write `all_formatted_ticks(pairs)` using yield from to chain all three.

# TODO: def raw_ticks(pairs): ...
# TODO: def filter_major(ticks): ...
# TODO: def format_tick(ticks): ...
# TODO: def all_formatted_ticks(pairs): ...


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 3 — Custom iterator class
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Build `RollingWindow` iterator class that:
#   - __init__(self, data: list, window: int)
#   - __iter__: returns self
#   - __next__: returns the next window slice as a list
#   - Raises StopIteration when no more complete windows exist
#   - Is REUSABLE — calling iter() on it again should restart from the beginning

# TODO: class RollingWindow: ...


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 4 — send() coroutine
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Write `running_stats()` generator that:
#   - When primed with next(), yields None
#   - On each send(value), yields a dict:
#       {"count": n, "mean": mean, "min": min_val, "max": max_val}
#   - Maintains running state WITHOUT storing all values

# TODO: def running_stats(): ...


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 5 — Async generator
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Write async generator `price_feed(pair: str, ticks: int, interval: float)`
#   - Yields simulated prices (start at 1.1050, increment by 0.0001 each tick)
#   - Waits `interval` seconds between ticks (use asyncio.sleep)
#
# Write `consume_feed(pair, ticks)` async function that:
#   - Iterates the price_feed
#   - Collects all ticks into a list and returns it

# TODO: async def price_feed(pair, ticks, interval=0.01): ...
# TODO: async def consume_feed(pair, ticks): ...


# ─────────────────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_exercise_1():
    print("\n--- Exercise 1: candle_generator ---")
    prices = [1.0, 2.0, 3.0, 4.0, 5.0]
    windows = list(candle_generator(prices, 3))
    assert windows == [(1.0,2.0,3.0), (2.0,3.0,4.0), (3.0,4.0,5.0)], \
        f"Got {windows}"
    print(f"  windows: {windows}  PASS")

    # log_returns
    returns = list(log_returns)
    assert len(returns) == 4
    assert all(isinstance(r, float) for r in returns)
    print(f"  log_returns: {[round(r,6) for r in returns]}  PASS")


def test_exercise_2():
    print("\n--- Exercise 2: pipeline ---")
    pairs = ["EUR_USD", "GBP_USD", "EUR_GBP", "USD_JPY"]
    result = list(all_formatted_ticks(pairs))
    # Only EUR pairs should remain
    assert all("EUR" in r for r in result), f"Non-EUR ticks leaked: {result}"
    assert len(result) == 2, f"Expected 2 EUR ticks, got {len(result)}"
    print(f"  {result}  PASS")


def test_exercise_3():
    print("\n--- Exercise 3: RollingWindow ---")
    data = [1, 2, 3, 4, 5, 6]
    rw = RollingWindow(data, window=3)
    result1 = list(rw)
    assert result1 == [[1,2,3],[2,3,4],[3,4,5],[4,5,6]], f"Got {result1}"

    # Reusability — second iteration should restart
    result2 = list(rw)
    assert result2 == result1, "RollingWindow should be reusable"
    print(f"  windows: {result1}  PASS")


def test_exercise_4():
    print("\n--- Exercise 4: running_stats ---")
    gen = running_stats()
    next(gen)   # prime
    s1 = gen.send(10.0)
    s2 = gen.send(20.0)
    s3 = gen.send(5.0)

    assert s3["count"] == 3
    assert abs(s3["mean"] - 11.666666) < 0.001
    assert s3["min"] == 5.0
    assert s3["max"] == 20.0
    print(f"  stats after 3 values: {s3}  PASS")


def test_exercise_5():
    print("\n--- Exercise 5: async price_feed ---")
    ticks = asyncio.run(consume_feed("EUR_USD", 5))
    assert len(ticks) == 5
    assert ticks[0] == 1.1050
    assert ticks[-1] == pytest_approx(1.1090)
    print(f"  ticks: {ticks}  PASS")

def pytest_approx(val, rel=1e-3):
    """Tiny stand-in for pytest.approx."""
    class Approx:
        def __eq__(self, other): return abs(other - val) < abs(val) * rel
        def __repr__(self): return f"~{val}"
    return Approx()


if __name__ == "__main__":
    print("=" * 60)
    print("Topic 06 — Generators — Exercise Runner")
    print("=" * 60)

    for fn in [test_exercise_1, test_exercise_2, test_exercise_3,
               test_exercise_4, test_exercise_5]:
        try:
            fn()
        except (NameError, AssertionError, TypeError) as e:
            print(f"  INCOMPLETE: {e}")

    print("\n" + "=" * 60)
