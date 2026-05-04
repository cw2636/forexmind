"""
Topic 06 — Generators & Iterators
===================================
SOLUTIONS
"""

from __future__ import annotations
import asyncio


# ── Solution 1 ───────────────────────────────────────────────────────────────

def candle_generator(prices: list, window: int):
    for i in range(len(prices) - window + 1):
        yield tuple(prices[i:i + window])

closes = [1.1050, 1.1060, 1.1055, 1.1045, 1.1070]
log_returns = (closes[i] / closes[i-1] - 1 for i in range(1, len(closes)))


# ── Solution 2 ───────────────────────────────────────────────────────────────

def raw_ticks(pairs: list):
    for i, pair in enumerate(pairs):
        yield (pair, 1.0 + i * 0.01)

def filter_major(ticks):
    for pair, price in ticks:
        if pair.startswith("EUR"):
            yield (pair, price)

def format_tick(ticks):
    for pair, price in ticks:
        yield f"{pair} @ {price:.4f}"

def all_formatted_ticks(pairs: list):
    yield from format_tick(filter_major(raw_ticks(pairs)))


# ── Solution 3 ───────────────────────────────────────────────────────────────

class RollingWindow:
    def __init__(self, data: list, window: int):
        self.data   = data
        self.window = window
        self.index  = 0

    def __iter__(self):
        self.index = 0   # reset for reusability
        return self

    def __next__(self):
        if self.index + self.window > len(self.data):
            raise StopIteration
        window = self.data[self.index : self.index + self.window]
        self.index += 1
        return window


# ── Solution 4 ───────────────────────────────────────────────────────────────

def running_stats():
    count    = 0
    total    = 0.0
    min_val  = float("inf")
    max_val  = float("-inf")
    value    = yield None   # prime

    while True:
        count   += 1
        total   += value
        min_val  = min(min_val, value)
        max_val  = max(max_val, value)
        value = yield {
            "count": count,
            "mean":  total / count,
            "min":   min_val,
            "max":   max_val,
        }


# ── Solution 5 ───────────────────────────────────────────────────────────────

async def price_feed(pair: str, ticks: int, interval: float = 0.01):
    price = 1.1050
    for _ in range(ticks):
        await asyncio.sleep(interval)
        yield round(price, 5)
        price += 0.0001

async def consume_feed(pair: str, ticks: int) -> list:
    result = []
    async for tick in price_feed(pair, ticks, interval=0.005):
        result.append(tick)
    return result


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_1():
    print("\n--- Exercise 1 ---")
    prices = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert list(candle_generator(prices, 3)) == [(1,2,3),(2,3,4),(3,4,5)]
    # regenerate since generator is exhausted
    global log_returns, closes
    log_returns = (closes[i] / closes[i-1] - 1 for i in range(1, len(closes)))
    returns = list(log_returns)
    assert len(returns) == 4
    print(f"  PASS")

def test_2():
    print("\n--- Exercise 2 ---")
    pairs = ["EUR_USD", "GBP_USD", "EUR_GBP", "USD_JPY"]
    result = list(all_formatted_ticks(pairs))
    assert len(result) == 2
    assert all("EUR" in r for r in result)
    print(f"  {result}  PASS")

def test_3():
    print("\n--- Exercise 3 ---")
    rw = RollingWindow([1,2,3,4,5,6], 3)
    r1 = list(rw)
    r2 = list(rw)
    assert r1 == [[1,2,3],[2,3,4],[3,4,5],[4,5,6]]
    assert r1 == r2
    print(f"  PASS")

def test_4():
    print("\n--- Exercise 4 ---")
    gen = running_stats()
    next(gen)
    gen.send(10.0); gen.send(20.0)
    s = gen.send(5.0)
    assert s["count"] == 3
    assert abs(s["mean"] - 35/3) < 0.001
    assert s["min"] == 5.0 and s["max"] == 20.0
    print(f"  {s}  PASS")

def test_5():
    print("\n--- Exercise 5 ---")
    ticks = asyncio.run(consume_feed("EUR_USD", 5))
    assert len(ticks) == 5
    assert ticks[0] == 1.1050
    print(f"  {ticks}  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("Topic 06 — Generators — SOLUTIONS")
    print("=" * 60)
    test_1(); test_2(); test_3(); test_4(); test_5()
    print("\nAll solutions verified!")
