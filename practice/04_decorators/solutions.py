"""
Topic 04 — Decorators
======================
SOLUTIONS
"""

from __future__ import annotations
import functools
import time


# ── Solution 1: timer ────────────────────────────────────────────────────────

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start  = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"  ⏱  {func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

@timer
def slow_indicator(pair: str, period: int = 14) -> float:
    time.sleep(0.05)
    return 42.0


# ── Solution 2: retry ─────────────────────────────────────────────────────────

def retry(max_attempts: int = 3, exceptions: tuple = (Exception,)):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    print(f"    [retry] attempt {attempt} failed: {exc}")
            raise last_exc
        return wrapper
    return decorator

call_count = 0

@retry(max_attempts=3)
def flaky_api() -> str:
    global call_count
    call_count += 1
    if call_count < 3:
        raise ConnectionError(f"Connection failed (attempt {call_count})")
    return "OK"


# ── Solution 3: CallCounter ──────────────────────────────────────────────────

class CallCounter:
    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func        = func
        self.count       = 0
        self.last_result = None

    def __call__(self, *args, **kwargs):
        self.count      += 1
        self.last_result = self.func(*args, **kwargs)
        return self.last_result

@CallCounter
def generate_signal(pair: str) -> str:
    return "BUY" if "EUR" in pair else "SELL"


# ── Solution 4: log_calls + stacking ────────────────────────────────────────

def log_calls(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        arg_str = ", ".join(repr(a) for a in args)
        print(f"  → calling {func.__name__}({arg_str})")
        result = func(*args, **kwargs)
        print(f"  ← {func.__name__} returned {result!r}")
        return result
    return wrapper

# Stacking: @log_calls(outer) @timer(inner)
# Call chain: log_calls → timer → compute_rsi
@log_calls
@timer
def compute_rsi(prices: list, period: int = 14) -> float:
    time.sleep(0.02)
    return 55.5


# ── Solution 5: dual-mode validate_pair ──────────────────────────────────────

_DEFAULT_PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY"]

def validate_pair(_func=None, *, valid=None):
    """
    Dual-mode: works as @validate_pair AND @validate_pair(valid=[...])
    The trick: if called WITHOUT parentheses, Python passes the function
    as _func. If called WITH parens, _func=None and we return a decorator.
    """
    effective_valid = valid or _DEFAULT_PAIRS

    def decorator(func):
        @functools.wraps(func)
        def wrapper(pair: str, *args, **kwargs):
            if pair not in effective_valid:
                raise ValueError(
                    f"pair {pair!r} not in allowed list: {effective_valid}"
                )
            return func(pair, *args, **kwargs)
        return wrapper

    if _func is not None:       # @validate_pair  (no parens)
        return decorator(_func)
    return decorator             # @validate_pair(...)  (with parens)

@validate_pair
def price_lookup(pair: str) -> float:
    return 1.1050

@validate_pair(valid=["EUR_USD"])
def restricted_lookup(pair: str) -> float:
    return 1.1050


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_exercise_1():
    print("\n--- Exercise 1: timer ---")
    result = slow_indicator("EUR_USD")
    assert result == 42.0
    assert slow_indicator.__name__ == "slow_indicator"
    print(f"  result={result}  PASS")

def test_exercise_2():
    print("\n--- Exercise 2: retry ---")
    global call_count
    call_count = 0
    result = flaky_api()
    assert result == "OK"
    assert call_count == 3
    print(f"  Succeeded after {call_count} attempts  PASS")

    @retry(max_attempts=2, exceptions=(ValueError,))
    def always_fails():
        raise ValueError("always")

    try:
        always_fails()
    except ValueError:
        print("  PASS: re-raises")

def test_exercise_3():
    print("\n--- Exercise 3: CallCounter ---")
    generate_signal("EUR_USD")
    generate_signal("GBP_USD")
    generate_signal("EUR_USD")
    assert generate_signal.count == 3
    assert generate_signal.__name__ == "generate_signal"
    print(f"  count={generate_signal.count}  PASS")

def test_exercise_4():
    print("\n--- Exercise 4: stacked decorators ---")
    result = compute_rsi([1, 2, 3], 14)
    assert result == 55.5
    assert compute_rsi.__name__ == "compute_rsi"
    print(f"  PASS")

def test_exercise_5():
    print("\n--- Exercise 5: dual-mode ---")
    assert price_lookup("EUR_USD") == 1.1050
    try:
        price_lookup("XXX_YYY")
    except ValueError as e:
        print(f"  PASS (no-args): {e}")
    assert restricted_lookup("EUR_USD") == 1.1050
    try:
        restricted_lookup("GBP_USD")
    except ValueError as e:
        print(f"  PASS (with-args): {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Topic 04 — Decorators — SOLUTIONS")
    print("=" * 60)
    test_exercise_1()
    test_exercise_2()
    test_exercise_3()
    test_exercise_4()
    test_exercise_5()
    print("\nAll solutions verified!")
