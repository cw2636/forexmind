# Topic 03 — Descriptors

## The Four Interview Questions

---

### 1. WHAT is it?

A **descriptor** is any object that defines `__get__`, `__set__`, or `__delete__`.
When a descriptor is assigned as a **class attribute**, Python calls these methods
instead of doing normal attribute access. This is the mechanism behind `@property`,
`@classmethod`, `@staticmethod`, and `functools.cached_property`.

```
Descriptor Protocol:
  __get__(self, obj, objtype)   → called on attribute READ
  __set__(self, obj, value)     → called on attribute WRITE
  __delete__(self, obj)         → called on attribute DELETE
  __set_name__(self, owner, name) → called when class is DEFINED (Python 3.6+)
```

---

### 2. WHY does Python have them?

Descriptors let you attach *behaviour* to attribute access without putting logic
inside `__getattr__`/`__setattr__` in the class itself. This means you can:
- Reuse validation logic across multiple classes (DRY)
- Intercept reads/writes transparently
- Implement lazy loading, caching, type-coercion

**Real use in ForexMind**: `StrategySignal.confidence` should always be 0.0–1.0.
Instead of validating in `__post_init__`, a descriptor validates on every write.

---

### 3. WHEN do you use each?

| Type | Has `__set__`? | Behaviour |
|------|-----------|-----------|
| **Data descriptor** | Yes | Overrides instance `__dict__` |
| **Non-data descriptor** | No | Instance `__dict__` takes priority |
| `@property` | Yes (if setter defined) | Syntactic sugar over descriptor |
| `functools.cached_property` | No (non-data) | Instance dict cached after first access |

---

### 4. SHOW ME — Annotated Examples

```python
# ── A reusable range validator descriptor ────────────────────────────────────
class RangeValidator:
    """
    Descriptor that enforces min <= value <= max on any float field.
    Can be reused across multiple classes.
    """

    def __set_name__(self, owner, name):
        # Called when the class is defined.
        # `owner` = the class containing this descriptor (e.g. StrategySignal)
        # `name`  = the attribute name it's assigned to (e.g. "confidence")
        self.public_name  = name
        self.private_name = f"_{name}"   # store actual value here in instance dict

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self   # accessed from the CLASS itself → return descriptor
        return getattr(obj, self.private_name, 0.0)

    def __set__(self, obj, value: float):
        if not isinstance(value, (int, float)):
            raise TypeError(f"{self.public_name} must be numeric, got {type(value)}")
        if not (self.min_val <= value <= self.max_val):
            raise ValueError(
                f"{self.public_name} must be in [{self.min_val}, {self.max_val}], "
                f"got {value}"
            )
        setattr(obj, self.private_name, float(value))

    def __init__(self, min_val: float, max_val: float):
        self.min_val = min_val
        self.max_val = max_val


class StrategySignal:
    confidence = RangeValidator(0.0, 1.0)   # descriptor as class attribute
    risk_pct   = RangeValidator(0.1, 5.0)

    def __init__(self, instrument, direction, confidence, risk_pct=1.0):
        self.instrument = instrument
        self.direction  = direction
        self.confidence = confidence   # triggers RangeValidator.__set__
        self.risk_pct   = risk_pct


sig = StrategySignal("EUR_USD", "BUY", confidence=0.8)
sig.confidence = 1.5   # ValueError: confidence must be in [0.0, 1.0], got 1.5
```

---

## Key Concepts Deep-Dive

### `__set_name__` (Python 3.6+)

Before 3.6, descriptors had no way to know what name they were assigned to.
You had to pass the name explicitly: `confidence = RangeValidator("confidence", 0, 1)`.

`__set_name__` is called automatically when the class body is executed:

```python
class MyDesc:
    def __set_name__(self, owner, name):
        print(f"Assigned to {owner.__name__}.{name}")
        self.name = name

class Foo:
    x = MyDesc()   # prints: "Assigned to Foo.x"
```

---

### Data vs Non-Data Descriptors

This distinction determines priority vs instance `__dict__`:

```python
class DataDesc:
    def __get__(self, obj, t): return "from descriptor"
    def __set__(self, obj, v): pass    # makes it a DATA descriptor

class NonDataDesc:
    def __get__(self, obj, t): return "from descriptor"
    # No __set__ → non-data descriptor

class MyClass:
    data     = DataDesc()
    non_data = NonDataDesc()

obj = MyClass()
obj.__dict__["data"]     = "from dict"    # instance dict set directly
obj.__dict__["non_data"] = "from dict"    # instance dict set directly

print(obj.data)      # "from descriptor"  ← DATA desc wins over instance dict
print(obj.non_data)  # "from dict"        ← instance dict wins over non-data desc
```

---

### `@property` is a Descriptor

`@property` is simply a built-in descriptor class. Writing:

```python
class Foo:
    @property
    def name(self): return self._name
```

Is equivalent to:

```python
class Foo:
    def _get_name(self): return self._name
    name = property(_get_name)
```

---

### `functools.cached_property` (Non-Data Descriptor)

```python
from functools import cached_property

class StrategySignal:
    def __init__(self, entry, stop, tp):
        self.entry = entry
        self.stop  = stop
        self.tp    = tp

    @cached_property
    def risk_reward(self):
        """Computed once, then stored in instance __dict__."""
        risk   = abs(self.entry - self.stop)
        reward = abs(self.tp    - self.entry)
        return reward / risk if risk > 0 else 0.0

sig = StrategySignal(1.1050, 1.1020, 1.1110)
print(sig.risk_reward)          # computed
print(sig.__dict__)             # {"risk_reward": 2.0, ...} ← cached in dict
print(sig.risk_reward)          # returned from dict, NOT recomputed
```

---

## Common Interview Mistakes

1. **Forgetting `if obj is None: return self`** — without this, accessing the
   descriptor from the class (not an instance) raises `AttributeError`
2. **Storing value in `self` instead of the instance** — storing value in the
   descriptor object itself means ALL instances share the same value!
3. **Confusing data and non-data priority** — data descriptors override instance dict,
   non-data descriptors don't
4. **`__set_name__` not called for dynamically added descriptors** — only fired during
   class creation (class body execution)

```python
# BUG: storing value in the descriptor object (not the instance)
class BuggyDesc:
    def __get__(self, obj, t): return self.value   # WRONG: self is shared!
    def __set__(self, obj, v): self.value = v      # all instances share this!

class Foo:
    x = BuggyDesc()

a, b = Foo(), Foo()
a.x = 10
print(b.x)   # 10 ← contaminated! b sees a's value

# FIX: store in the INSTANCE's __dict__
class GoodDesc:
    def __set_name__(self, owner, name): self.name = f"_{name}"
    def __get__(self, obj, t):
        if obj is None: return self
        return obj.__dict__.get(self.name, None)
    def __set__(self, obj, v): obj.__dict__[self.name] = v
```

---

## Quick Reference Cheatsheet

```python
class MyDescriptor:
    # Called during class definition
    def __set_name__(self, owner: type, name: str) -> None:
        self.public_name  = name
        self.private_name = f"_{name}"

    # Called on attribute read
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self           # class-level access → return descriptor itself
        return getattr(obj, self.private_name, None)

    # Called on attribute write (makes this a DATA descriptor)
    def __set__(self, obj, value):
        # validate / transform value here
        setattr(obj, self.private_name, value)

    # Called on attribute delete
    def __delete__(self, obj):
        delattr(obj, self.private_name)
```
