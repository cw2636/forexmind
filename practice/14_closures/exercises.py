"""
Topic 14 — Closures, nonlocal, Free Variables
===============================================
EXERCISES + SOLUTIONS

Run: python 14_closures/exercises.py
"""

from __future__ import annotations


# ── EXERCISE 1: Basic closure factory ────────────────────────────────────────
# Write `make_pip_calculator(pair: str)` that returns a closure `calc(price_diff)`.
# The closure multiplies price_diff by the pair's pip multiplier:
#   "USD_JPY" → multiply by 100  (1 pip = 0.01)
#   all others → multiply by 10000 (1 pip = 0.0001)
# Return value: number of pips

# TODO: def make_pip_calculator(pair): ...


# ── EXERCISE 2: nonlocal counter ─────────────────────────────────────────────
# Write `make_trade_tracker()` that returns three closures: add(units), remove(units), total()
# `add` and `remove` modify an internal `position` counter using nonlocal.
# `total` returns the current position.

# TODO: def make_trade_tracker(): ...


# ── EXERCISE 3: Late binding trap + fix ───────────────────────────────────────
# Demonstrate the late binding trap:
#   Create bad_funcs = [lambda: i for i in range(5)]
#   Show that all return 4 (the final value of i)
#
# Fix it two ways:
#   good_funcs_1 using default argument trick
#   good_funcs_2 using a factory function

# TODO: show bad, then fix both ways


# ── EXERCISE 4: Inspect __closure__ ──────────────────────────────────────────
# Write `inspect_closure(fn)` that prints:
#   "Free variables: x, y, ..."
#   "Values: x=..., y=..."
# Use fn.__code__.co_freevars and fn.__closure__

# TODO: def inspect_closure(fn): ...


# ── EXERCISE 5: Memoize using closure (manual lru_cache) ─────────────────────
# Write `memoize(fn)` using a closure that:
#   - Keeps a `cache` dict as a free variable
#   - Returns cached result on repeated calls with same args
#   - Works for any function with hashable arguments

# TODO: def memoize(fn): ...


# ─────────────────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────────────────

def tests():
    print("\n--- Exercise 1: pip calculator ---")
    eur_calc = make_pip_calculator("EUR_USD")
    jpy_calc = make_pip_calculator("USD_JPY")
    assert abs(eur_calc(0.0030) - 30.0) < 0.01, f"Got {eur_calc(0.0030)}"
    assert abs(jpy_calc(0.30)  - 30.0) < 0.01,  f"Got {jpy_calc(0.30)}"
    print(f"  EUR 30 pips: {eur_calc(0.0030):.1f}, JPY 30 pips: {jpy_calc(0.30):.1f}  PASS")

    print("\n--- Exercise 2: trade tracker ---")
    add, remove, total = make_trade_tracker()
    add(1000); add(500); remove(300)
    assert total() == 1200, f"Expected 1200, got {total()}"
    print(f"  position={total()}  PASS")

    print("\n--- Exercise 3: late binding ---")
    bad_funcs = [lambda: i for i in range(5)]
    results   = [f() for f in bad_funcs]
    assert all(r == 4 for r in results), f"Expected all 4s, got {results}"
    print(f"  bad_funcs: {results} (all 4 — late binding)")

    good_funcs_1 = [lambda i=i: i for i in range(5)]
    good_funcs_2 = [(lambda x: lambda: x)(i) for i in range(5)]
    assert [f() for f in good_funcs_1] == [0,1,2,3,4]
    assert [f() for f in good_funcs_2] == [0,1,2,3,4]
    print("  fixed: [0,1,2,3,4]  PASS")

    print("\n--- Exercise 4: inspect_closure ---")
    def outer(x, y):
        def inner(z): return x + y + z
        return inner
    fn = outer(10, 20)
    inspect_closure(fn)
    assert fn.__code__.co_freevars == ("x", "y")
    print("  PASS")

    print("\n--- Exercise 5: memoize ---")
    call_count = [0]
    @memoize
    def slow_square(n):
        call_count[0] += 1
        return n * n

    slow_square(5); slow_square(5); slow_square(5)
    assert slow_square(5) == 25
    assert call_count[0] == 1, f"Expected 1 real call, got {call_count[0]}"
    print(f"  calls: {call_count[0]}  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("Topic 14 — Closures — Exercise Runner")
    print("=" * 60)
    try:
        tests()
    except (NameError, AssertionError) as e:
        print(f"  INCOMPLETE: {e}")
    print("=" * 60)


# ── SOLUTIONS ─────────────────────────────────────────────────────────────────
"""
def make_pip_calculator(pair):
    multiplier = 100 if pair == "USD_JPY" else 10_000
    def calc(price_diff): return abs(price_diff) * multiplier
    return calc

def make_trade_tracker():
    position = 0
    def add(units):
        nonlocal position; position += units
    def remove(units):
        nonlocal position; position -= units
    def total(): return position
    return add, remove, total

# late binding fix already in tests

def inspect_closure(fn):
    names  = fn.__code__.co_freevars
    cells  = fn.__closure__ or ()
    print(f"  Free variables: {', '.join(names)}")
    for name, cell in zip(names, cells):
        print(f"    {name} = {cell.cell_contents}")

def memoize(fn):
    cache = {}
    def wrapper(*args):
        if args not in cache:
            cache[args] = fn(*args)
        return cache[args]
    return wrapper
"""
