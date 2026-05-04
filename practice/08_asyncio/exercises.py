"""
Topic 08 — asyncio & Concurrency
==================================
EXERCISES + SOLUTIONS

Run: python 08_asyncio/exercises.py
"""

from __future__ import annotations
import asyncio
import time


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 1 — gather: fetch prices concurrently
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Write `fetch_price(pair: str) -> float` coroutine that:
#   - Sleeps 0.1s (simulates API latency)
#   - Returns 1.1050 for EUR pairs, 1.2600 for GBP, 130.0 for JPY
#
# Write `fetch_all(pairs: list) -> dict` that uses asyncio.gather to fetch
# all pairs CONCURRENTLY and returns {pair: price}.
# Verify total time ≈ 0.1s (not N * 0.1s).

# TODO: async def fetch_price(pair): ...
# TODO: async def fetch_all(pairs): ...


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 2 — asyncio.Lock protecting shared state
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Build `Account` class with:
#   - balance: float (starts at 10000)
#   - async debit(amount) — subtracts amount; raises ValueError if insufficient
#   - async credit(amount) — adds amount
#   - Lock protecting both debit and credit
#
# Test: run 5 concurrent debits of 1000 each — final balance must be 5000

# TODO: class Account: ...


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 3 — asyncio.Queue (producer/consumer)
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Write:
#   - `tick_producer(queue, pairs)` — puts (pair, simulated_price) for each pair
#     then puts a None sentinel
#   - `signal_consumer(queue, results)` — reads until sentinel, appends to results
#   - `run_pipeline(pairs)` — gathers producer + consumer, returns results list

# TODO: async def tick_producer(queue, pairs): ...
# TODO: async def signal_consumer(queue, results): ...
# TODO: async def run_pipeline(pairs): ...


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 4 — asyncio.wait_for (timeout)
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Write `safe_fetch(pair, timeout=0.5)` that:
#   - Calls fetch_price (simulates slow fetch with 0.2s sleep)
#   - If it takes longer than `timeout`, returns None instead of raising
#   - Uses asyncio.wait_for and catches asyncio.TimeoutError

# TODO: async def safe_fetch(pair, timeout=0.5): ...


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 5 — asyncio.to_thread (blocking code bridge)
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Write a BLOCKING function `load_historical_data(pair)` that sleeps 0.3s
# and returns a list of 5 fake prices.
#
# Write async `async_load(pair)` that offloads it via asyncio.to_thread.
# Verify it does NOT block the event loop by running it alongside another coroutine.

# TODO: def load_historical_data(pair): ...
# TODO: async def async_load(pair): ...


# ─────────────────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_exercise_1():
    print("\n--- Exercise 1: gather ---")
    pairs = ["EUR_USD", "GBP_USD", "USD_JPY"]
    start = time.perf_counter()
    results = asyncio.run(fetch_all(pairs))
    elapsed = time.perf_counter() - start
    assert len(results) == 3
    assert elapsed < 0.3, f"Should be ~0.1s, took {elapsed:.2f}s"
    print(f"  {results}  elapsed={elapsed:.2f}s  PASS")


def test_exercise_2():
    print("\n--- Exercise 2: Lock ---")
    async def run():
        acc = Account(balance=10000)
        await asyncio.gather(*[acc.debit(1000) for _ in range(5)])
        assert acc.balance == 5000, f"Expected 5000, got {acc.balance}"
        print(f"  balance={acc.balance}  PASS")
    asyncio.run(run())


def test_exercise_3():
    print("\n--- Exercise 3: Queue ---")
    pairs = ["EUR_USD", "GBP_USD", "USD_JPY"]
    results = asyncio.run(run_pipeline(pairs))
    assert len(results) == 3
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
    print(f"  {results}  PASS")


def test_exercise_4():
    print("\n--- Exercise 4: wait_for ---")
    async def run():
        fast = await safe_fetch("EUR_USD", timeout=0.5)
        slow = await safe_fetch("EUR_USD", timeout=0.05)  # too short
        assert fast is not None
        assert slow is None
        print(f"  fast={fast}, slow(timeout)={slow}  PASS")
    asyncio.run(run())


def test_exercise_5():
    print("\n--- Exercise 5: to_thread ---")
    async def run():
        start = time.perf_counter()
        # Run blocking load alongside a sleep — should complete in ~0.3s not 0.6s
        results, _ = await asyncio.gather(
            async_load("EUR_USD"),
            asyncio.sleep(0.3),
        )
        elapsed = time.perf_counter() - start
        assert len(results) == 5
        assert elapsed < 0.5, f"Took {elapsed:.2f}s — blocking not offloaded?"
        print(f"  data={results}, elapsed={elapsed:.2f}s  PASS")
    asyncio.run(run())


if __name__ == "__main__":
    print("=" * 60)
    print("Topic 08 — asyncio — Exercise Runner")
    print("=" * 60)

    for fn in [test_exercise_1, test_exercise_2, test_exercise_3,
               test_exercise_4, test_exercise_5]:
        try:
            fn()
        except (NameError, AssertionError, TypeError) as e:
            print(f"  INCOMPLETE: {e}")

    print("\n" + "=" * 60)


# ── SOLUTIONS (collapsed) ────────────────────────────────────────────────────
"""
async def fetch_price(pair: str) -> float:
    await asyncio.sleep(0.1)
    if "EUR" in pair: return 1.1050
    if "GBP" in pair: return 1.2600
    return 130.0

async def fetch_all(pairs: list) -> dict:
    prices = await asyncio.gather(*[fetch_price(p) for p in pairs])
    return dict(zip(pairs, prices))

class Account:
    def __init__(self, balance=10000):
        self.balance = float(balance)
        self._lock   = asyncio.Lock()
    async def debit(self, amount):
        async with self._lock:
            if self.balance < amount: raise ValueError("Insufficient")
            self.balance -= amount
    async def credit(self, amount):
        async with self._lock:
            self.balance += amount

async def tick_producer(queue, pairs):
    for i, pair in enumerate(pairs):
        await asyncio.sleep(0.01)
        await queue.put((pair, 1.0 + i * 0.01))
    await queue.put(None)

async def signal_consumer(queue, results):
    while True:
        item = await queue.get()
        if item is None: break
        results.append(item)
        queue.task_done()

async def run_pipeline(pairs):
    queue   = asyncio.Queue()
    results = []
    await asyncio.gather(tick_producer(queue, pairs), signal_consumer(queue, results))
    return results

async def safe_fetch(pair, timeout=0.5):
    try:
        return await asyncio.wait_for(fetch_price(pair), timeout=timeout)
    except asyncio.TimeoutError:
        return None

def load_historical_data(pair):
    time.sleep(0.3)
    return [1.1 + i*0.001 for i in range(5)]

async def async_load(pair):
    return await asyncio.to_thread(load_historical_data, pair)
"""
