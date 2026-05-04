# Topic 08 — asyncio & Concurrency

## The Four Interview Questions

---

### 1. WHAT is it?

`asyncio` is Python's built-in library for **cooperative multitasking** using
coroutines. A coroutine is a function defined with `async def` that can `await`
other coroutines or I/O operations, yielding control back to the event loop
while waiting — without blocking other work.

---

### 2. WHY does Python have it?

Most trading system bottlenecks are **I/O-bound** (waiting for API responses,
database writes, WebSocket ticks). Traditional threads would waste CPU spinning.
`asyncio` lets you run thousands of concurrent I/O operations with a single thread
and event loop — no GIL problems, low overhead.

---

### 3. WHEN do you use each?

| Construct | Use |
|-----------|-----|
| `asyncio.gather(*coros)` | Run multiple coroutines concurrently, wait for ALL |
| `asyncio.create_task(coro)` | Start coroutine in background, don't wait yet |
| `asyncio.TaskGroup` | Python 3.11+ — structured concurrency, cancels all on error |
| `asyncio.Lock` | Protect shared state from concurrent mutation |
| `asyncio.Queue` | Producer/consumer pattern between coroutines |
| `asyncio.wait_for(coro, timeout)` | Cancel if takes too long |
| `asyncio.to_thread(fn)` | Run blocking (sync) code in a thread pool |
| `asyncio.Event` | Signal between coroutines (set/wait pattern) |

---

### 4. SHOW ME — Annotated Examples

```python
import asyncio

# ── Basic coroutine ───────────────────────────────────────────────────────────
async def fetch_price(pair: str) -> float:
    await asyncio.sleep(0.1)   # simulate API latency
    return 1.1050

# ── gather: run concurrently ─────────────────────────────────────────────────
async def fetch_all_prices(pairs: list[str]) -> list[float]:
    # Fires all requests at once — total time ≈ max(individual times)
    results = await asyncio.gather(
        *[fetch_price(p) for p in pairs]
    )
    return results

# Sequential:  N * latency
# Concurrent:  ~1 * latency (if all the same)

asyncio.run(fetch_all_prices(["EUR_USD", "GBP_USD", "USD_JPY"]))


# ── asyncio.Lock — protect shared state ─────────────────────────────────────
class AccountManager:
    def __init__(self):
        self.balance = 10000.0
        self._lock   = asyncio.Lock()

    async def debit(self, amount: float) -> None:
        async with self._lock:            # only one coroutine at a time
            if self.balance < amount:
                raise ValueError("Insufficient funds")
            self.balance -= amount        # atomic under the lock


# ── asyncio.Queue — producer/consumer ────────────────────────────────────────
async def price_producer(queue: asyncio.Queue, pairs: list[str]) -> None:
    for pair in pairs:
        price = await fetch_price(pair)
        await queue.put((pair, price))
    await queue.put(None)   # sentinel: signals producer is done

async def signal_consumer(queue: asyncio.Queue) -> None:
    while True:
        item = await queue.get()
        if item is None:
            break
        pair, price = item
        print(f"Signal for {pair}: {price}")
        queue.task_done()

async def main():
    queue = asyncio.Queue()
    # run producer and consumer concurrently
    await asyncio.gather(
        price_producer(queue, ["EUR_USD", "GBP_USD"]),
        signal_consumer(queue),
    )


# ── asyncio.TaskGroup (Python 3.11+) ────────────────────────────────────────
async def fetch_with_group(pairs: list[str]) -> list[float]:
    results = []
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(fetch_price(p)) for p in pairs]
    # All tasks complete here (or if one raises, ALL are cancelled)
    return [t.result() for t in tasks]


# ── asyncio.to_thread — blocking code ────────────────────────────────────────
import time

def blocking_csv_read(filename: str) -> list:
    time.sleep(0.5)   # simulates blocking I/O
    return [1.1050, 1.1060]

async def async_read(filename: str) -> list:
    # Offloads to thread pool — event loop is NOT blocked
    return await asyncio.to_thread(blocking_csv_read, filename)
```

---

## Key Concepts Deep-Dive

### Event Loop Lifecycle

```
asyncio.run(main())
    │
    ▼  Creates event loop
    ├── schedules main()
    ├── runs event loop until main() completes
    └── closes event loop

Inside event loop:
    while tasks_not_done:
        for task in ready_tasks:
            task.step()   # run until next await
        poll_io()         # wake tasks waiting on I/O
```

### `gather` vs `create_task`

```python
# gather: returns when ALL tasks complete
await asyncio.gather(task_a(), task_b())   # waits for both

# create_task: fire-and-forget, doesn't wait
t = asyncio.create_task(background_task())
# ... do other work ...
await t    # wait for it when ready
```

### Exception handling in `gather`

```python
# Default: first exception is re-raised, others are silently cancelled
results = await asyncio.gather(
    fetch_price("EUR_USD"),
    fetch_price("INVALID"),
    return_exceptions=True,   # ← returns Exception objects instead of raising
)
# results = [1.1050, ValueError("Invalid pair")]
```

---

## Common Interview Mistakes

1. **Calling blocking code in a coroutine** — `time.sleep()` blocks the ENTIRE event loop
   Use `await asyncio.sleep()` or `await asyncio.to_thread(blocking_fn)`
2. **Forgetting `await`** — coroutine objects are created but not executed without `await`
3. **Creating `asyncio.Lock()` outside the event loop** — must create inside an async context
4. **`asyncio.run()` called from inside the event loop** — raises RuntimeError
5. **Not handling `CancelledError`** — when tasks are cancelled, `CancelledError` is raised

---

## Quick Reference Cheatsheet

```python
import asyncio

# Run entry point
asyncio.run(main())

# Concurrent execution
results = await asyncio.gather(coro1(), coro2(), return_exceptions=True)

# Background task
task = asyncio.create_task(coro())
await task

# Timeout
try:
    result = await asyncio.wait_for(coro(), timeout=5.0)
except asyncio.TimeoutError:
    ...

# Lock
lock = asyncio.Lock()
async with lock: ...

# Queue
q = asyncio.Queue()
await q.put(item)
item = await q.get()
q.task_done()
await q.join()

# Thread offload
result = await asyncio.to_thread(sync_function, arg1, arg2)

# Event
event = asyncio.Event()
event.set()
await event.wait()
```
