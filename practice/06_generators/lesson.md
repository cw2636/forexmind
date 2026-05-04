# Topic 06 — Generators & Iterators

## The Four Interview Questions

---

### 1. WHAT is it?

A **generator** is a function that uses `yield` to produce values one at a time,
pausing execution between each value. It returns a **generator object** (an iterator)
that generates the sequence lazily — values are computed only when needed.

An **iterator** is any object implementing `__iter__()` and `__next__()`.
All generators are iterators, but not all iterators are generators.

---

### 2. WHY does Python have them?

**Memory efficiency**: A generator that yields 1M prices uses O(1) memory.
An equivalent list uses O(1M) memory.

**Lazy evaluation**: You only compute values when they're consumed — useful for
streaming market data, reading large CSV files, or generating training samples.

**Pipeline composition**: Generators can be chained — each stage consumes from
the previous lazily, like Unix pipes.

---

### 3. WHEN do you use each?

| Pattern | Use |
|---------|-----|
| `yield` (generator function) | Lazy sequence, streaming data |
| `yield from` | Delegating to sub-generator |
| Generator expression | One-liner lazy sequence |
| `send()` | Two-way communication with generator (coroutine-lite) |
| Async generator (`async def` + `yield`) | Streaming async data (e.g. WebSocket ticks) |
| `__iter__` + `__next__` | Custom iterator class |

---

### 4. SHOW ME — Annotated Examples

```python
# ── Basic generator ───────────────────────────────────────────────────────────
def price_stream(start: float, n: int, step: float = 0.0001):
    """Yields n prices starting from `start`, incrementing by `step`."""
    price = start
    for _ in range(n):
        yield price
        price += step

gen = price_stream(1.1050, 5)
print(next(gen))   # 1.1050
print(next(gen))   # 1.1051
list(gen)          # [1.1052, 1.1053, 1.1054]  (consumed)


# ── yield from (delegation) ──────────────────────────────────────────────────
def all_prices():
    yield from price_stream(1.1050, 3)   # EUR/USD prices
    yield from price_stream(1.2600, 3)   # GBP/USD prices

list(all_prices())  # [1.1050, 1.1051, 1.1052, 1.2600, 1.2601, 1.2602]


# ── Generator expression (lazy list comprehension) ───────────────────────────
closes = [1.1050, 1.1060, 1.1055, 1.1070]
returns = (closes[i] / closes[i-1] - 1 for i in range(1, len(closes)))
# Nothing computed yet! Only when iterated:
list(returns)   # [0.000906, -0.000453, 0.001421]


# ── send() — two-way communication ──────────────────────────────────────────
def running_average():
    """
    A coroutine-style generator.
    Send prices in, receive running average back.
    """
    total, count = 0.0, 0
    value = yield None        # first next() primes the generator
    while True:
        total += value
        count += 1
        value = yield total / count   # yield avg, receive next price

avg = running_average()
next(avg)           # prime (advance to first yield)
print(avg.send(1.1050))   # 1.1050
print(avg.send(1.1060))   # 1.1055
print(avg.send(1.1070))   # 1.1060


# ── Async generator ──────────────────────────────────────────────────────────
import asyncio

async def tick_stream(pair: str, n: int):
    """Async generator yielding simulated live price ticks."""
    price = 1.1050
    for i in range(n):
        await asyncio.sleep(0.01)   # simulate network delay
        price += 0.0001 * (1 if i % 2 == 0 else -1)
        yield price

async def consume():
    async for tick in tick_stream("EUR_USD", 5):
        print(f"Tick: {tick:.5f}")
```

---

## Key Concepts Deep-Dive

### Generator State Machine

When Python calls `next()` on a generator, it:
1. Resumes from where it last `yield`ed
2. Runs until the next `yield` (or `return`)
3. Suspends and returns the yielded value

```
            gen = my_gen()
                 │
                 ▼
[ CREATED ]──next()──► [ RUNNING ] ──yield v──► [ SUSPENDED ]
                              ▲                        │
                              └──────next()────────────┘
                              │
                           StopIteration (on return or end)
                              │
                              ▼
                          [ CLOSED ]
```

---

### `__iter__` and `__next__` from scratch

```python
class CandleIterator:
    """Iterates over OHLCV candles in a DataFrame, one at a time."""
    def __init__(self, df):
        self.df    = df
        self.index = 0

    def __iter__(self):
        return self    # the iterator IS the iterable

    def __next__(self):
        if self.index >= len(self.df):
            raise StopIteration
        row = self.df.iloc[self.index]
        self.index += 1
        return row

# Any for loop calls __iter__ then repeatedly calls __next__
for candle in CandleIterator(df):
    print(candle["close"])
```

---

### Generator Pipeline (like Unix pipes)

```python
def read_prices(filename):
    with open(filename) as f:
        for line in f:
            yield float(line.strip())

def filter_outliers(prices, max_val=2.0):
    for p in prices:
        if p < max_val:
            yield p

def compute_returns(prices):
    prev = None
    for p in prices:
        if prev is not None:
            yield (p - prev) / prev
        prev = p

# Chain: file → filter → returns  (all lazy, O(1) memory)
pipeline = compute_returns(filter_outliers(read_prices("prices.txt")))
for ret in pipeline:
    print(ret)
```

---

## Common Interview Mistakes

1. **Iterating a generator twice** — generators are exhausted after one pass
2. **Forgetting to prime `send()` generators** — must call `next(gen)` first
3. **`yield` in a regular function vs async** — you can't `yield` in an `async def` without `async def` + `yield`
4. **`list(gen)` before using it** — defeats the lazy evaluation purpose
5. **`StopIteration` inside a generator causes `RuntimeError`** (PEP 479, Python 3.7+)

---

## Quick Reference Cheatsheet

```python
# Basic generator
def my_gen():
    yield 1
    yield 2
    yield 3

# Generator expression
gen = (x**2 for x in range(10))

# yield from
def chained():
    yield from range(3)
    yield from range(10, 13)

# send() coroutine
def accumulator():
    total = 0
    while True:
        value = yield total
        if value is None: break
        total += value

gen = accumulator()
next(gen)           # prime
gen.send(10)        # → 10
gen.send(20)        # → 30

# Async generator
async def async_gen():
    for i in range(3):
        await asyncio.sleep(0)
        yield i

async for item in async_gen():
    print(item)

# Custom iterator
class MyIter:
    def __init__(self, data): self.data = data; self.i = 0
    def __iter__(self): return self
    def __next__(self):
        if self.i >= len(self.data): raise StopIteration
        v = self.data[self.i]; self.i += 1; return v
```
