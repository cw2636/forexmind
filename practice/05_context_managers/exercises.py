"""
Topic 05 — Context Managers
============================
EXERCISES — Fill in every section marked TODO.

Run: python 05_context_managers/exercises.py
"""

from __future__ import annotations
import contextlib
import time


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 1 — Class-based context manager
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Build `TradeSession` that:
#   - Accepts pair: str, units: int
#   - __enter__: prints "OPEN  <pair> x<units>", returns self
#   - __exit__:  prints "CLOSE <pair>" always
#               If an exception occurred, print "ERROR: <exc_val>" and SUPPRESS it
# 
# Test: a division-by-zero inside the with block should NOT propagate.

# TODO: class TradeSession: ...


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 2 — @contextmanager generator-based
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Write `timer_ctx()` using @contextlib.contextmanager that:
#   - Records start time before yield
#   - Yields the start time so the caller can use it
#   - After yield, prints "Elapsed: X.XXXXs"
#   - Uses try/finally to guarantee the elapsed message prints even on exception

# TODO: @contextlib.contextmanager
# def timer_ctx(): ...


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 3 — __exit__ exception inspection
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Build `SafeAPICall` context manager (class-based) that:
#   - __enter__: returns self
#   - __exit__:
#       * If exc_type is ConnectionError → print "Network error, retrying later"
#         and SUPPRESS
#       * If exc_type is ValueError → print "Invalid data: <exc_val>" and SUPPRESS
#       * Any other exception → print "Unhandled: <exc_type.__name__>" and PROPAGATE
#       * No exception → do nothing, return False

# TODO: class SafeAPICall: ...


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 4 — Async context manager
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Write `async_rate_limiter(calls_per_sec: float)` using
# @contextlib.asynccontextmanager that:
#   - Records start time
#   - Yields
#   - After the block, sleeps for any remaining time to enforce the rate limit
#     (i.e. if calls_per_sec=2, each call should take at least 0.5s total)
#
# Test with asyncio.run()

# TODO: @contextlib.asynccontextmanager
# async def async_rate_limiter(calls_per_sec: float): ...


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 5 — ExitStack
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Use contextlib.ExitStack to open multiple TradeSession contexts
# dynamically (number of pairs determined at runtime).
#
# Write `open_basket(pairs: list[str], units: int)` that:
#   - Uses ExitStack to open a TradeSession for EACH pair
#   - Returns the list of active sessions
#   - All sessions are closed when the `with` block exits
#
# NOTE: open_basket itself should be a context manager (use @contextmanager)

# TODO: @contextlib.contextmanager
# def open_basket(pairs: list, units: int): ...


# ─────────────────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_exercise_1():
    print("\n--- Exercise 1: TradeSession ---")
    print("  Normal (no exception):")
    with TradeSession("EUR_USD", 1000) as ts:
        print(f"    trading: {ts.pair}")

    print("  With exception (should be suppressed):")
    with TradeSession("GBP_USD", 500):
        x = 1 / 0   # ZeroDivisionError — should be suppressed
    print("  PASS: exception suppressed")


def test_exercise_2():
    print("\n--- Exercise 2: timer_ctx ---")
    with timer_ctx() as start:
        time.sleep(0.05)
        assert isinstance(start, float)
    print("  PASS: timer_ctx works")


def test_exercise_3():
    print("\n--- Exercise 3: SafeAPICall ---")
    with SafeAPICall():
        raise ConnectionError("timeout")   # should be suppressed
    print("  PASS: ConnectionError suppressed")

    with SafeAPICall():
        raise ValueError("bad payload")    # should be suppressed
    print("  PASS: ValueError suppressed")

    try:
        with SafeAPICall():
            raise RuntimeError("unknown")  # should propagate
        print("  FAIL: RuntimeError should not be suppressed")
    except RuntimeError:
        print("  PASS: RuntimeError propagated")


def test_exercise_4():
    print("\n--- Exercise 4: async_rate_limiter ---")
    import asyncio

    async def run():
        start = time.perf_counter()
        async with async_rate_limiter(calls_per_sec=5):  # min 0.2s per call
            pass  # instant work
        elapsed = time.perf_counter() - start
        assert elapsed >= 0.18, f"Expected >= 0.2s, got {elapsed:.3f}s"
        print(f"  elapsed={elapsed:.3f}s  PASS")

    asyncio.run(run())


def test_exercise_5():
    print("\n--- Exercise 5: ExitStack / open_basket ---")
    pairs = ["EUR_USD", "GBP_USD", "USD_JPY"]
    with open_basket(pairs, units=500) as sessions:
        assert len(sessions) == 3
        for s in sessions:
            assert s.pair in pairs
    print("  PASS: all sessions opened and closed")


if __name__ == "__main__":
    print("=" * 60)
    print("Topic 05 — Context Managers — Exercise Runner")
    print("=" * 60)

    for fn in [test_exercise_1, test_exercise_2, test_exercise_3,
               test_exercise_4, test_exercise_5]:
        try:
            fn()
        except (NameError, AssertionError, TypeError) as e:
            print(f"  INCOMPLETE: {e}")

    print("\n" + "=" * 60)
