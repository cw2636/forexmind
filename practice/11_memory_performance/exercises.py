"""
Topic 11 — Memory & Performance
==================================
EXERCISES + SOLUTIONS

Run: python 11_memory_performance/exercises.py
"""

from __future__ import annotations
import sys
import timeit
import tracemalloc
from functools import lru_cache


# ── EXERCISE 1: __slots__ memory comparison ───────────────────────────────────
# Create NoSlots and WithSlots versions of a Tick class (pair, bid, ask, ts).
# Create 10,000 of each and compare peak memory using tracemalloc.
# TODO


# ── EXERCISE 2: lru_cache ─────────────────────────────────────────────────────
# Write `compute_sma(prices: tuple, period: int) -> float` with lru_cache(128).
# Verify cache hits after repeated calls with same arguments.
# TODO


# ── EXERCISE 3: timeit benchmark ──────────────────────────────────────────────
# Benchmark three ways to build a list of squared values [0 to N]:
#   a) list comprehension
#   b) map(lambda x: x**2, range(N))
#   c) for loop
# Print which is fastest.
# TODO


# ── EXERCISE 4: tracemalloc leak detection ─────────────────────────────────────
# Create a function that allocates memory (builds a large list).
# Use tracemalloc to measure peak allocation, then verify it's freed.
# TODO


# ── EXERCISE 5: slotted dataclass-like class ──────────────────────────────────
# Create `FastSignal` using __slots__ with 5 fields.
# Compare sys.getsizeof() between FastSignal and a regular class version.
# TODO


# ─────────────────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_exercise_1():
    print("\n--- Exercise 1: __slots__ memory ---")
    tracemalloc.start()
    objs_no = [NoSlots("EUR_USD", 1.1048, 1.1052, "2024-01-01") for _ in range(10_000)]
    _, peak_no = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()

    objs_sl = [WithSlots("EUR_USD", 1.1048, 1.1052, "2024-01-01") for _ in range(10_000)]
    _, peak_sl = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  NoSlots  peak: {peak_no/1024:.1f} KB")
    print(f"  WithSlots peak: {peak_sl/1024:.1f} KB")
    assert peak_sl < peak_no, "WithSlots should use less memory"
    print("  PASS: WithSlots uses less memory")


def test_exercise_2():
    print("\n--- Exercise 2: lru_cache ---")
    prices = tuple([1.1 + i*0.0001 for i in range(50)])
    compute_sma(prices, 14)   # miss
    compute_sma(prices, 14)   # hit
    compute_sma(prices, 14)   # hit
    info = compute_sma.cache_info()
    assert info.hits >= 2 and info.misses >= 1
    print(f"  cache_info: {info}  PASS")


def test_exercise_3():
    print("\n--- Exercise 3: timeit ---")
    run_benchmarks()   # just verify it runs without error
    print("  PASS: benchmarks complete")


def test_exercise_4():
    print("\n--- Exercise 4: tracemalloc ---")
    peak = measure_peak()
    print(f"  Peak allocation: {peak/1024:.1f} KB  PASS")


def test_exercise_5():
    print("\n--- Exercise 5: FastSignal vs RegularSignal ---")
    fs = FastSignal("EUR_USD", "BUY", 0.8, 1.1050, 1.1020)
    rs = RegularSignal("EUR_USD", "BUY", 0.8, 1.1050, 1.1020)
    fast_size    = sys.getsizeof(fs)
    regular_size = sys.getsizeof(rs) + sys.getsizeof(rs.__dict__)
    print(f"  FastSignal size:    {fast_size} bytes")
    print(f"  RegularSignal size: {regular_size} bytes")
    assert fast_size < regular_size
    print("  PASS: FastSignal is smaller")


if __name__ == "__main__":
    print("=" * 60)
    print("Topic 11 — Memory & Performance — Exercise Runner")
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
class NoSlots:
    def __init__(self, pair, bid, ask, ts):
        self.pair=pair; self.bid=bid; self.ask=ask; self.ts=ts

class WithSlots:
    __slots__ = ("pair","bid","ask","ts")
    def __init__(self, pair, bid, ask, ts):
        self.pair=pair; self.bid=bid; self.ask=ask; self.ts=ts

@lru_cache(maxsize=128)
def compute_sma(prices: tuple, period: int) -> float:
    return sum(prices[-period:]) / period

def run_benchmarks():
    N = 1000
    t_comp = timeit.timeit(f"[x**2 for x in range({N})]", number=10_000)
    t_map  = timeit.timeit(f"list(map(lambda x:x**2, range({N})))", number=10_000)
    t_loop = timeit.timeit(
        f"r=[]\nfor x in range({N}): r.append(x**2)", number=10_000
    )
    fastest = min(("comprehension",t_comp),("map",t_map),("loop",t_loop),key=lambda x:x[1])
    print(f"    comp={t_comp:.3f}s  map={t_map:.3f}s  loop={t_loop:.3f}s")
    print(f"    Fastest: {fastest[0]}")

def measure_peak() -> int:
    tracemalloc.start()
    data = [i for i in range(100_000)]
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del data
    return peak

class FastSignal:
    __slots__ = ("instrument","direction","confidence","entry","stop")
    def __init__(self,i,d,c,e,s):
        self.instrument=i; self.direction=d; self.confidence=c
        self.entry=e; self.stop=s

class RegularSignal:
    def __init__(self,i,d,c,e,s):
        self.instrument=i; self.direction=d; self.confidence=c
        self.entry=e; self.stop=s
"""
