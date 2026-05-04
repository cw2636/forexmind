# Topic 11 — Memory & Performance

## The Four Interview Questions

---

### 1. WHAT is it?

Python provides several tools to reduce memory usage and execution time:
- **`__slots__`** — replaces instance `__dict__` with fixed slots (40-50% less memory)
- **`functools.lru_cache`** — memoization (cache function results)
- **`cProfile` / `timeit`** — measure where time is spent
- **`sys.getsizeof` / `tracemalloc`** — measure memory usage
- **`array`, `numpy`** — typed arrays instead of Python lists

---

### 2. WHY does Python have them?

A signal generator running on 18 pairs × 3 timeframes = 54 concurrent streams.
If each stream creates thousands of `StrategySignal` objects per minute, memory
pressure compounds fast. `__slots__` and `lru_cache` are frequently the difference
between a system that runs for hours vs one that OOMs.

---

### 3. When to use each?

| Tool | Use |
|------|-----|
| `__slots__` | High-frequency object creation (signals, ticks, candles) |
| `lru_cache(maxsize=N)` | Pure functions called repeatedly with same args |
| `cache` (Python 3.9+) | `lru_cache(maxsize=None)` — unbounded |
| `timeit.timeit` | Micro-benchmark a snippet |
| `cProfile.run` | Profile which functions are slowest |
| `tracemalloc` | Find memory leaks |

---

### 4. SHOW ME — Annotated Examples

```python
import sys
from functools import lru_cache

# ── Without __slots__ ─────────────────────────────────────────────────────────
class SignalNoSlots:
    def __init__(self, instrument, direction, confidence):
        self.instrument = instrument
        self.direction  = direction
        self.confidence = confidence

obj = SignalNoSlots("EUR_USD", "BUY", 0.8)
print(sys.getsizeof(obj))               # ~48 bytes (object shell)
print(sys.getsizeof(obj.__dict__))      # ~232 bytes (dict overhead!) ← the waste


# ── With __slots__ ────────────────────────────────────────────────────────────
class SignalWithSlots:
    __slots__ = ("instrument", "direction", "confidence")

    def __init__(self, instrument, direction, confidence):
        self.instrument = instrument
        self.direction  = direction
        self.confidence = confidence

obj2 = SignalWithSlots("EUR_USD", "BUY", 0.8)
print(sys.getsizeof(obj2))    # ~64 bytes — no __dict__ overhead
# obj2.new_attr = "x"         # AttributeError: can't add new attributes


# ── lru_cache ─────────────────────────────────────────────────────────────────
@lru_cache(maxsize=128)
def compute_ema(prices_tuple: tuple, period: int) -> float:
    """
    Only computed once per unique (prices, period) pair.
    Note: arguments must be HASHABLE — use tuple, not list.
    """
    k = 2 / (period + 1)
    ema = prices_tuple[0]
    for p in prices_tuple[1:]:
        ema = p * k + ema * (1 - k)
    return ema

prices = tuple([1.1050 + i * 0.0001 for i in range(100)])
result = compute_ema(prices, 14)   # computed
result = compute_ema(prices, 14)   # returned from cache (instant)

print(compute_ema.cache_info())
# CacheInfo(hits=1, misses=1, maxsize=128, currsize=1)


# ── Profiling with cProfile ────────────────────────────────────────────────────
import cProfile

def slow_function():
    return sum(x**2 for x in range(100_000))

cProfile.run("slow_function()", sort="cumulative")
# Outputs a table: ncalls, tottime, percall, cumtime, filename


# ── timeit for micro-benchmarks ───────────────────────────────────────────────
import timeit

# Compare list comprehension vs generator
list_time = timeit.timeit("[x**2 for x in range(1000)]", number=10_000)
gen_time  = timeit.timeit("list(x**2 for x in range(1000))", number=10_000)
print(f"list: {list_time:.3f}s, gen: {gen_time:.3f}s")


# ── tracemalloc — find memory leaks ──────────────────────────────────────────
import tracemalloc

tracemalloc.start()
objects = [SignalNoSlots("EUR_USD", "BUY", 0.8) for _ in range(10_000)]
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"Peak memory: {peak / 1024:.1f} KB")
```

---

## Key Concepts Deep-Dive

### `__slots__` and inheritance

```python
class Base:
    __slots__ = ("x",)

class Child(Base):
    __slots__ = ("y",)   # MUST redeclare slots in each class

    # If you omit __slots__ in Child, Child gets __dict__ again —
    # defeating the purpose for Child instances.

c = Child()
c.x = 1       # from Base
c.y = 2       # from Child
c.z = 3       # AttributeError — no __dict__
```

### `lru_cache` gotchas

```python
# Arguments MUST be hashable:
@lru_cache
def bad(prices: list):    # lists are unhashable → TypeError
    ...

@lru_cache
def good(prices: tuple):  # tuples are hashable → OK
    ...

# Instance methods: lru_cache + self creates a memory leak
# (the cache holds a reference to self, preventing GC)
# Use functools.cached_property or methodtools.lru_cache instead

# Clear the cache:
good.cache_clear()

# Inspect:
good.cache_info()   # CacheInfo(hits, misses, maxsize, currsize)
```

---

## Quick Reference Cheatsheet

```python
# __slots__
class MyClass:
    __slots__ = ("attr1", "attr2")

# lru_cache
from functools import lru_cache, cache
@lru_cache(maxsize=256)
def pure_fn(arg): ...   # arg must be hashable

@cache   # Python 3.9+ unbounded
def pure_fn2(arg): ...

# timeit
import timeit
timeit.timeit("fn()", setup="from __main__ import fn", number=1000)

# cProfile
import cProfile
cProfile.run("main()", sort="cumulative")

# tracemalloc
import tracemalloc
tracemalloc.start()
# ... code ...
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
```
