# Topic 14 — Closures, `nonlocal`, and Free Variables

## The Four Interview Questions

---

### 1. WHAT is it?

A **closure** is a function that *captures* variables from its enclosing scope —
even after that outer function has returned. The captured variables are called
**free variables**.

```python
def outer(x):
    def inner(y):
        return x + y   # `x` is a free variable — captured from outer's scope
    return inner

add5 = outer(5)        # outer returns and is done, but `x=5` is preserved
print(add5(3))         # 8 — x is still accessible!
```

---

### 2. WHY does Python have them?

Closures enable:
- **Factories** — functions that generate other functions with baked-in configuration
- **Stateful callbacks** — attach context to a callback without a class
- **Decorator internals** — every decorator you write uses a closure
- **Partial application** — similar to `functools.partial` but more flexible

---

### 3. Key Concepts

| Concept | What it does |
|---------|-------------|
| **Free variable** | Variable used in inner function but defined in outer |
| **`__closure__`** | Tuple of `cell` objects holding free variable values |
| **`nonlocal`** | Declare that a variable in inner function refers to outer scope |
| **`global`** | Declare that a variable refers to module-level scope |
| **Late binding** | Python looks up free variables at CALL time, not definition time |

---

### 4. SHOW ME — Annotated Examples

```python
# ── Basic closure ─────────────────────────────────────────────────────────────
def make_multiplier(factor: float):
    def multiply(value: float) -> float:
        return value * factor   # `factor` is a free variable
    return multiply

double = make_multiplier(2.0)
triple = make_multiplier(3.0)

print(double(1.1050))   # 2.2100
print(triple(1.1050))   # 3.3150

# Inspect the closure
print(double.__closure__)  # (<cell at 0x...>,)
print(double.__closure__[0].cell_contents)  # 2.0


# ── nonlocal ─────────────────────────────────────────────────────────────────
def counter(start: int = 0):
    count = start

    def increment(step: int = 1) -> int:
        nonlocal count        # tells Python: `count` is in the enclosing scope
        count += step         # without nonlocal, this would create a new LOCAL variable
        return count

    def reset() -> None:
        nonlocal count
        count = start

    return increment, reset

inc, rst = counter(10)
print(inc())    # 11
print(inc(5))   # 16
rst()
print(inc())    # 11 (reset back to 10, then +1)


# ── Late binding trap (classic interview question!) ───────────────────────────
# WRONG — all lambdas capture the same `i` variable (late binding)
funcs = [lambda: i for i in range(3)]
print([f() for f in funcs])   # [2, 2, 2] ← ALL return 2 (the final i)

# FIX 1 — default argument captures the value at definition time
funcs = [lambda i=i: i for i in range(3)]
print([f() for f in funcs])   # [0, 1, 2] ← correct!

# FIX 2 — factory function forces immediate binding
def make_fn(x):
    return lambda: x
funcs = [make_fn(i) for i in range(3)]
print([f() for f in funcs])   # [0, 1, 2]


# ── Practical: stateful callback (no class needed) ───────────────────────────
def make_signal_logger(pair: str):
    signals = []

    def log(direction: str, confidence: float) -> None:
        signals.append({"pair": pair, "direction": direction, "confidence": confidence})

    def get_signals() -> list:
        return signals.copy()

    return log, get_signals

log_eur, get_eur = make_signal_logger("EUR_USD")
log_gbp, get_gbp = make_signal_logger("GBP_USD")

log_eur("BUY", 0.8)
log_eur("SELL", 0.7)
log_gbp("BUY", 0.6)

print(get_eur())   # 2 signals, all EUR_USD
print(get_gbp())   # 1 signal, GBP_USD
```

---

## `__closure__` Deep-Dive

```python
def outer():
    x = 10
    y = 20
    def inner():
        return x + y   # both x and y are free variables
    return inner

fn = outer()
print(fn.__code__.co_freevars)   # ('x', 'y') — names of free vars
print(fn.__closure__)             # (<cell x>, <cell y>)
for cell, name in zip(fn.__closure__, fn.__code__.co_freevars):
    print(f"  {name} = {cell.cell_contents}")
```

---

## Common Interview Mistakes

1. **The late binding trap** (`[lambda: i for i in range(3)]`) — all return last `i`
2. **`nonlocal` vs `global`** — `nonlocal` = enclosing function scope; `global` = module scope
3. **Trying to read a `nonlocal` variable without declaring it** — only needed for assignment
4. **Not knowing `__closure__`** — interviewers love this one
5. **Closures keeping objects alive** — the closure holds a reference, preventing GC

---

## Quick Reference Cheatsheet

```python
# Basic closure
def factory(config):
    def fn(data): return config + data
    return fn

# nonlocal counter
def counter():
    n = 0
    def inc():
        nonlocal n
        n += 1
        return n
    return inc

# Inspect closure
fn.__closure__                    # tuple of cells
fn.__code__.co_freevars           # names of free variables
fn.__closure__[0].cell_contents   # actual value

# Late binding fix
funcs = [lambda x=x: x for x in range(3)]  # capture by default arg
```
