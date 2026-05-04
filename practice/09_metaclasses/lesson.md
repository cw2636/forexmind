# Topic 09 — Metaclasses & `__init_subclass__`

## The Four Interview Questions

---

### 1. WHAT is it?

A **metaclass** is the class of a class. Just as a regular class defines how instances
behave, a metaclass defines how classes behave. By default, every class's metaclass
is `type`. A custom metaclass lets you intercept and modify class creation.

**`__init_subclass__`** (Python 3.6+) is a simpler alternative to metaclasses for
the most common use case: executing code when a subclass is defined.

---

### 2. WHY does Python have them?

Metaclasses power frameworks like Django ORM, SQLAlchemy, Pydantic, and `dataclasses`.
They enable:
- Auto-registration of plugins/strategies (no manual registry updates)
- Validation of class definitions at definition time
- Auto-generating methods based on annotations
- Singleton pattern enforcement

---

### 3. WHEN to use metaclass vs `__init_subclass__`?

| Need | Use |
|------|-----|
| Intercept class creation, modify `__dict__` | Metaclass |
| Run code when a subclass is defined | `__init_subclass__` (simpler) |
| Auto-register plugins | `__init_subclass__` |
| Enforce class-level constraints | Either |
| ABC-like enforcement | `__init_subclass__` |

> **Rule**: Use `__init_subclass__` unless you specifically need to modify
> the class `__dict__` or control `__new__`/`__init__` at the class level.

---

### 4. SHOW ME — Annotated Examples

```python
# ── Pattern 1: __init_subclass__ auto-registry ─────────────────────────────
class BaseStrategy:
    _registry: dict[str, type] = {}

    def __init_subclass__(cls, name: str = "", **kwargs):
        """
        Called automatically when BaseStrategy is subclassed.
        `cls`  = the new subclass being defined
        `name` = keyword argument from class definition line
        """
        super().__init_subclass__(**kwargs)
        strategy_name = name or cls.__name__.lower()
        BaseStrategy._registry[strategy_name] = cls
        print(f"  Registered: {strategy_name!r} → {cls.__name__}")

class RuleBasedStrategy(BaseStrategy, name="rule_based"):
    def generate_signal(self): return "BUY"

class MLStrategy(BaseStrategy, name="ml"):
    def generate_signal(self): return "SELL"

# No manual registration needed:
print(BaseStrategy._registry)
# {"rule_based": RuleBasedStrategy, "ml": MLStrategy}

# Factory function
def get_strategy(name: str) -> BaseStrategy:
    cls = BaseStrategy._registry.get(name)
    if cls is None:
        raise KeyError(f"Unknown strategy: {name!r}")
    return cls()


# ── Pattern 2: Metaclass for validation ──────────────────────────────────────
class StrategyMeta(type):
    """
    Metaclass that enforces every strategy class has a `name` class attribute.
    """

    def __new__(mcs, class_name, bases, namespace):
        # namespace = the class body dict
        cls = super().__new__(mcs, class_name, bases, namespace)

        # Skip validation for the base class itself
        if bases and "name" not in namespace:
            raise TypeError(
                f"{class_name} must define a 'name' class attribute"
            )
        return cls


class BaseStrategy2(metaclass=StrategyMeta):
    name = "base"

class GoodStrategy(BaseStrategy2):
    name = "good"           # required attribute present → OK

try:
    class BadStrategy(BaseStrategy2):  # no `name` → TypeError at class definition time
        pass
except TypeError as e:
    print(e)   # "BadStrategy must define a 'name' class attribute"


# ── Pattern 3: Singleton metaclass ──────────────────────────────────────────
class SingletonMeta(type):
    _instances: dict = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class RiskManager(metaclass=SingletonMeta):
    def __init__(self):
        self.balance = 10000.0

rm1 = RiskManager()
rm2 = RiskManager()
assert rm1 is rm2   # same instance
```

---

## Key Concepts Deep-Dive

### `type.__new__` vs `type.__init__`

```
Class creation flow:
1. Python executes the class body → builds namespace dict
2. Calls metaclass.__new__(mcs, name, bases, namespace)
3. Calls metaclass.__init__(cls, name, bases, namespace)
4. Returns the class object

Class instantiation flow (normal):
1. Calls cls.__new__(cls, *args)  → creates the instance
2. Calls cls.__init__(instance, *args) → initialises it
```

### `__init_subclass__` keyword arguments

```python
class Plugin:
    def __init_subclass__(cls, *, version: str = "1.0", **kwargs):
        super().__init_subclass__(**kwargs)
        cls.version = version   # inject class attribute

class MyPlugin(Plugin, version="2.5"):
    pass

print(MyPlugin.version)   # "2.5"
```

---

## Quick Reference Cheatsheet

```python
# __init_subclass__ registry
class Base:
    _registry = {}
    def __init_subclass__(cls, name="", **kwargs):
        super().__init_subclass__(**kwargs)
        Base._registry[name or cls.__name__] = cls

class Child(Base, name="child"): pass

# Metaclass
class Meta(type):
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        # modify cls here
        return cls
    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)

class MyClass(metaclass=Meta): pass

# Singleton
class Singleton(type):
    _inst = {}
    def __call__(cls, *a, **kw):
        if cls not in cls._inst:
            cls._inst[cls] = super().__call__(*a, **kw)
        return cls._inst[cls]
```
