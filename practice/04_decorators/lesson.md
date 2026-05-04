# Topic 04 — Decorators

## The Four Interview Questions

---

### 1. WHAT is it?

A **decorator** is a callable that takes a function (or class), wraps it with new
behaviour, and returns the modified callable. The `@` syntax is shorthand for:

```python
@my_decorator
def foo(): pass
# exactly equivalent to:
def foo(): pass
foo = my_decorator(foo)
```

---

### 2. WHY does Python have them?

Decorators cleanly separate *cross-cutting concerns* (logging, timing, retrying,
authentication, validation) from business logic. Without decorators, you'd repeat
the same boilerplate in every function.

**ForexMind use cases:**
- `@retry(max_attempts=3)` — retry OANDA API calls on network failures
- `@lru_cache` — cache indicator calculations
- `@require_auth` — protect web endpoints (conceptual)
- `@log_execution` — log every signal generated

---

### 3. WHEN do you use each variant?

| Pattern | When |
|---------|------|
| Simple function wrapper | Cross-cutting concerns: logging, timing |
| Parametrized decorator | Need to configure the wrapper (`@retry(n=3)`) |
| Class-based decorator | Need to store state between calls (call count, cache) |
| `functools.wraps` | Always use — preserves `__name__`, `__doc__`, `__wrapped__` |
| Stacked decorators | Apply multiple behaviours — applied bottom-up |

---

### 4. SHOW ME — Annotated Examples

```python
import functools
import time

# ── Pattern 1: Simple wrapper ─────────────────────────────────────────────────
def log_execution(func):
    @functools.wraps(func)       # copies __name__, __doc__, __annotations__
    def wrapper(*args, **kwargs):
        print(f"[LOG] Calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"[LOG] {func.__name__} returned {result!r}")
        return result
    return wrapper

@log_execution
def generate_signal(pair: str) -> str:
    return "BUY"

generate_signal("EUR_USD")
# [LOG] Calling generate_signal
# [LOG] generate_signal returned 'BUY'


# ── Pattern 2: Parametrized decorator (factory → decorator → wrapper) ────────
def retry(max_attempts: int = 3, delay: float = 0.5):
    """
    @retry(max_attempts=3)  — try the function up to 3 times before raising.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt < max_attempts:
                        time.sleep(delay)
            raise last_exc
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.1)
def fetch_price(pair: str) -> float:
    ...   # calls OANDA API — might fail


# ── Pattern 3: Class-based decorator (preserves state) ───────────────────────
class CallCounter:
    """Counts how many times the decorated function has been called."""

    def __init__(self, func):
        functools.update_wrapper(self, func)    # same as @functools.wraps
        self.func  = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self.func(*args, **kwargs)

@CallCounter
def place_trade(pair: str, units: int) -> bool:
    return True

place_trade("EUR_USD", 1000)
place_trade("GBP_USD", 500)
print(place_trade.count)   # 2


# ── Pattern 4: Stacking decorators ───────────────────────────────────────────
# Applied BOTTOM-UP (retry first, then log wraps the result)
@log_execution
@retry(max_attempts=3)
def get_account_balance() -> float:
    return 10000.0

# Execution order: log_execution wraps retry wraps get_account_balance
# Call: log → retry → get_account_balance
```

---

## Key Concepts Deep-Dive

### Why `functools.wraps` is essential

```python
def bad_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def good_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@bad_decorator
def my_func():
    """My important docstring."""
    pass

@good_decorator
def my_func2():
    """My important docstring."""
    pass

print(my_func.__name__)   # "wrapper"  ← broken! debugging nightmare
print(my_func2.__name__)  # "my_func2" ← correct
print(my_func2.__wrapped__)  # <function my_func2> ← access original
```

---

### Decorator with optional arguments (the dual-mode pattern)

Advanced: make `@retry` work BOTH as `@retry` AND `@retry(max_attempts=3)`:

```python
def retry(_func=None, *, max_attempts=3, delay=0.5):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if attempt == max_attempts - 1:
                        raise
            if delay:
                time.sleep(delay)
        return wrapper

    if _func is not None:       # used as @retry (no parentheses)
        return decorator(_func)
    return decorator            # used as @retry(...) with args

@retry                          # works!
def foo(): pass

@retry(max_attempts=5)          # also works!
def bar(): pass
```

---

### Decorator order matters

```python
@A
@B
@C
def func(): pass
# equivalent to: func = A(B(C(func)))
# Call chain: A's wrapper → B's wrapper → C's wrapper → original func
```

---

## Common Interview Mistakes

1. **Forgetting `functools.wraps`** — the decorated function loses its identity
2. **Parametrized vs simple** — `@retry` (no parens) vs `@retry()` (parens) are different
3. **`*args, **kwargs` in wrapper** — forgetting these breaks decorators on functions
   with parameters
4. **Class decorators and `self`** — when decorating methods, the wrapper receives
   `self` as first arg in `*args`
5. **Stacking order** — `@log @retry` means log wraps retry — reversed from visual order

---

## Quick Reference Cheatsheet

```python
import functools

# Simple decorator
def my_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # before
        result = func(*args, **kwargs)
        # after
        return result
    return wrapper

# Parametrized decorator
def parametrized(param1, param2="default"):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Class-based decorator
class ClassDecorator:
    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

# Method decorator (must handle self)
def method_timer(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        import time
        t = time.perf_counter()
        result = func(self, *args, **kwargs)
        print(f"{func.__name__}: {time.perf_counter()-t:.4f}s")
        return result
    return wrapper
```
