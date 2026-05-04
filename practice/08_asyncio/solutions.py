"""
Topic 08 — asyncio & Concurrency
==================================
SOLUTIONS (standalone, runnable)
"""

from __future__ import annotations
import asyncio
import time


async def fetch_price(pair: str) -> float:
    await asyncio.sleep(0.1)
    if "EUR" in pair: return 1.1050
    if "GBP" in pair: return 1.2600
    return 130.0

async def fetch_all(pairs: list) -> dict:
    prices = await asyncio.gather(*[fetch_price(p) for p in pairs])
    return dict(zip(pairs, prices))

class Account:
    def __init__(self, balance: float = 10000.0):
        self.balance = balance
        self._lock   = asyncio.Lock()
    async def debit(self, amount: float) -> None:
        async with self._lock:
            if self.balance < amount: raise ValueError("Insufficient funds")
            self.balance -= amount
    async def credit(self, amount: float) -> None:
        async with self._lock:
            self.balance += amount

async def tick_producer(queue: asyncio.Queue, pairs: list) -> None:
    for i, pair in enumerate(pairs):
        await asyncio.sleep(0.01)
        await queue.put((pair, 1.0 + i * 0.01))
    await queue.put(None)

async def signal_consumer(queue: asyncio.Queue, results: list) -> None:
    while True:
        item = await queue.get()
        if item is None: break
        results.append(item)
        queue.task_done()

async def run_pipeline(pairs: list) -> list:
    queue, results = asyncio.Queue(), []
    await asyncio.gather(tick_producer(queue, pairs), signal_consumer(queue, results))
    return results

async def safe_fetch(pair: str, timeout: float = 0.5):
    try:
        return await asyncio.wait_for(fetch_price(pair), timeout=timeout)
    except asyncio.TimeoutError:
        return None

def load_historical_data(pair: str) -> list:
    time.sleep(0.3)
    return [1.1 + i * 0.001 for i in range(5)]

async def async_load(pair: str) -> list:
    return await asyncio.to_thread(load_historical_data, pair)


if __name__ == "__main__":
    print("=" * 60)
    print("Topic 08 — asyncio — SOLUTIONS")
    print("=" * 60)

    # Ex1
    pairs = ["EUR_USD", "GBP_USD", "USD_JPY"]
    start = time.perf_counter()
    r = asyncio.run(fetch_all(pairs))
    elapsed = time.perf_counter() - start
    assert elapsed < 0.3
    print(f"\n  Ex1 gather: {r}  ({elapsed:.2f}s)  PASS")

    # Ex2
    async def check_lock():
        acc = Account(10000)
        await asyncio.gather(*[acc.debit(1000) for _ in range(5)])
        assert acc.balance == 5000
    asyncio.run(check_lock())
    print("  Ex2 Lock: PASS")

    # Ex3
    results = asyncio.run(run_pipeline(["EUR_USD","GBP_USD","USD_JPY"]))
    assert len(results) == 3
    print(f"  Ex3 Queue: {results}  PASS")

    # Ex4
    async def check_timeout():
        fast = await safe_fetch("EUR_USD", 0.5)
        slow = await safe_fetch("EUR_USD", 0.05)
        assert fast is not None and slow is None
    asyncio.run(check_timeout())
    print("  Ex4 wait_for: PASS")

    # Ex5
    async def check_thread():
        start = time.perf_counter()
        data, _ = await asyncio.gather(async_load("EUR_USD"), asyncio.sleep(0.3))
        assert time.perf_counter() - start < 0.5
        assert len(data) == 5
    asyncio.run(check_thread())
    print("  Ex5 to_thread: PASS")

    print("\nAll solutions verified!")
