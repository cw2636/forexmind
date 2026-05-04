# Topic 05 — Context Managers

## The Four Interview Questions

---

### 1. WHAT is it?

A **context manager** is an object that defines `__enter__` and `__exit__` methods,
enabling the `with` statement. It guarantees that setup and teardown code always runs —
even if an exception is raised inside the `with` block.

```python
with open("prices.csv") as f:    # __enter__ called → returns file object
    data = f.read()
# __exit__ called here → file closed, even if read() raised an error
```

---

### 2. WHY does Python have them?

The `try/finally` pattern is verbose and error-prone. Context managers make resource
management declarative and safe. They're the standard way to handle:
- File I/O, database connections, network sockets
- Locks and semaphores (thread safety)
- Database transactions (commit/rollback)
- Trade lifecycle tracking (open → monitor → close)

---

### 3. WHEN do you use each variant?

| Pattern | Use |
|---------|-----|
| Class with `__enter__`/`__exit__` | Reusable, stateful resource managers |
| `@contextlib.contextmanager` | Quick one-off context managers using `yield` |
| `@contextlib.asynccontextmanager` | Async version (use with `async with`) |
| `contextlib.suppress(exc)` | Silently ignore specific exceptions |
| `contextlib.ExitStack` | Dynamically manage multiple context managers |

---

### 4. SHOW ME — Annotated Examples

```python
import contextlib
from typing import Generator

# ── Class-based context manager ───────────────────────────────────────────────
class TradeSession:
    """
    Tracks an open trade lifecycle.
    Ensures the trade is always closed (SL hit / TP hit / timeout).
    """

    def __init__(self, pair: str, units: int):
        self.pair  = pair
        self.units = units
        self.trade_id = None

    def __enter__(self) -> "TradeSession":
        # Setup: open the trade, log entry
        self.trade_id = f"T-{self.pair}-{self.units}"
        print(f"[TRADE OPEN] {self.trade_id}")
        return self          # value bound to `as trade`

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        # Teardown: always runs
        print(f"[TRADE CLOSE] {self.trade_id}")
        if exc_type is not None:
            print(f"[ERROR] Closed due to exception: {exc_val}")
        # Return True  → suppress the exception
        # Return False → let the exception propagate (usual choice)
        return False

with TradeSession("EUR_USD", 1000) as trade:
    print(f"Trading {trade.trade_id}")
    # even if this raises, __exit__ runs


# ── Generator-based (contextlib.contextmanager) ───────────────────────────────
@contextlib.contextmanager
def managed_trade(pair: str) -> Generator:
    print(f"Opening {pair}")
    try:
        yield pair          # everything before yield = __enter__
    except Exception as e:
        print(f"Exception: {e}")
        raise
    finally:
        print(f"Closing {pair}")   # always runs = __exit__

with managed_trade("EUR_USD") as pair:
    print(f"Active: {pair}")


# ── Async context manager ─────────────────────────────────────────────────────
import asyncio

@contextlib.asynccontextmanager
async def async_trade(pair: str):
    print(f"Async open {pair}")
    try:
        yield pair
    finally:
        print(f"Async close {pair}")

async def main():
    async with async_trade("EUR_USD") as pair:
        await asyncio.sleep(0.01)
        print(f"Async trading {pair}")
```

---

## `__exit__` Signature Deep-Dive

```python
def __exit__(self, exc_type, exc_val, exc_tb):
    #           ^^^^^^^^   ^^^^^^^   ^^^^^^
    #           Exception  Exception Traceback
    #           TYPE       VALUE     object
    #           (None if no exception)

    if exc_type is ValueError:
        # Handle specific exception
        print(f"Caught ValueError: {exc_val}")
        return True    # ← suppress: does NOT propagate

    if exc_type is not None:
        # Some other exception occurred — log and let it propagate
        print(f"Unexpected: {exc_type.__name__}: {exc_val}")
        return False   # ← propagate

    return False       # no exception — normal exit
```

---

## `contextlib.ExitStack` — Dynamic Context Managers

```python
from contextlib import ExitStack

pairs = ["EUR_USD", "GBP_USD", "USD_JPY"]

# Open 1, 2, or 3 trades dynamically
with ExitStack() as stack:
    sessions = [stack.enter_context(TradeSession(p, 1000)) for p in pairs]
    # all three sessions are active here
# all three __exit__ methods called here, in reverse order
```

---

## Common Interview Mistakes

1. **Forgetting to return `self` from `__enter__`** — without it, `as x` binds `None`
2. **Returning `True` accidentally** — suppresses ALL exceptions, not just the expected one
3. **Generator-based: exception before `yield`** — `__exit__` won't run if setup fails
4. **`asynccontextmanager` in sync code** — must be used with `async with`, not `with`
5. **Not handling exceptions in `finally`** — in `@contextmanager`, use `try/finally` not bare `yield`

---

## Quick Reference Cheatsheet

```python
# Class-based
class MyCtx:
    def __enter__(self):
        return self          # or whatever the caller needs
    def __exit__(self, exc_type, exc_val, exc_tb):
        # cleanup
        return False         # don't suppress exceptions

# Generator-based
from contextlib import contextmanager
@contextmanager
def my_ctx():
    # setup
    try:
        yield value
    except SomeError:
        pass               # suppress
    finally:
        pass               # cleanup (always runs)

# Async generator-based
from contextlib import asynccontextmanager
@asynccontextmanager
async def my_async_ctx():
    # setup
    try:
        yield value
    finally:
        pass               # cleanup

# Suppress exceptions silently
from contextlib import suppress
with suppress(FileNotFoundError):
    open("missing_file.txt")
```
