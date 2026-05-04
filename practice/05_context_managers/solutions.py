"""
Topic 05 — Context Managers
============================
SOLUTIONS
"""

from __future__ import annotations
import asyncio
import contextlib
import time


# ── Solution 1 ───────────────────────────────────────────────────────────────

class TradeSession:
    def __init__(self, pair: str, units: int):
        self.pair  = pair
        self.units = units

    def __enter__(self) -> "TradeSession":
        print(f"  OPEN  {self.pair} x{self.units}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        print(f"  CLOSE {self.pair}")
        if exc_type is not None:
            print(f"  ERROR: {exc_val}")
            return True   # suppress ALL exceptions
        return False


# ── Solution 2 ───────────────────────────────────────────────────────────────

@contextlib.contextmanager
def timer_ctx():
    start = time.perf_counter()
    try:
        yield start
    finally:
        elapsed = time.perf_counter() - start
        print(f"  Elapsed: {elapsed:.4f}s")


# ── Solution 3 ───────────────────────────────────────────────────────────────

class SafeAPICall:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is ConnectionError:
            print(f"  Network error, retrying later")
            return True   # suppress
        if exc_type is ValueError:
            print(f"  Invalid data: {exc_val}")
            return True   # suppress
        if exc_type is not None:
            print(f"  Unhandled: {exc_type.__name__}")
            return False  # propagate
        return False


# ── Solution 4 ───────────────────────────────────────────────────────────────

@contextlib.asynccontextmanager
async def async_rate_limiter(calls_per_sec: float):
    min_interval = 1.0 / calls_per_sec
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed   = time.perf_counter() - start
        remaining = min_interval - elapsed
        if remaining > 0:
            await asyncio.sleep(remaining)


# ── Solution 5 ───────────────────────────────────────────────────────────────

@contextlib.contextmanager
def open_basket(pairs: list, units: int):
    with contextlib.ExitStack() as stack:
        sessions = [stack.enter_context(TradeSession(p, units)) for p in pairs]
        yield sessions


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_exercise_1():
    print("\n--- Exercise 1 ---")
    with TradeSession("EUR_USD", 1000) as ts:
        assert ts.pair == "EUR_USD"
    with TradeSession("GBP_USD", 500):
        x = 1 / 0
    print("  PASS")

def test_exercise_2():
    print("\n--- Exercise 2 ---")
    with timer_ctx() as start:
        time.sleep(0.05)
        assert isinstance(start, float)
    print("  PASS")

def test_exercise_3():
    print("\n--- Exercise 3 ---")
    with SafeAPICall(): raise ConnectionError("timeout")
    with SafeAPICall(): raise ValueError("bad")
    try:
        with SafeAPICall(): raise RuntimeError("whoops")
    except RuntimeError:
        print("  PASS: RuntimeError propagated")

def test_exercise_4():
    print("\n--- Exercise 4 ---")
    async def run():
        s = time.perf_counter()
        async with async_rate_limiter(5):
            pass
        assert time.perf_counter() - s >= 0.18
        print("  PASS: rate limiting works")
    asyncio.run(run())

def test_exercise_5():
    print("\n--- Exercise 5 ---")
    with open_basket(["EUR_USD", "GBP_USD"], 500) as sessions:
        assert len(sessions) == 2
    print("  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("Topic 05 — Context Managers — SOLUTIONS")
    print("=" * 60)
    test_exercise_1()
    test_exercise_2()
    test_exercise_3()
    test_exercise_4()
    test_exercise_5()
    print("\nAll solutions verified!")
