"""
Topic 09 — Metaclasses & __init_subclass__
============================================
EXERCISES + SOLUTIONS

Run: python 09_metaclasses/exercises.py
"""

from __future__ import annotations


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 1 — __init_subclass__ auto-registry
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Build `StrategyBase` with:
#   - Class-level _registry: dict
#   - __init_subclass__ that registers each subclass by its `name` kwarg
#   - Class method `get(name)` that returns the class from registry
#   - Class method `list_all()` -> list of registered names

# TODO: class StrategyBase: ...

# These should register automatically:
# class RuleBased(StrategyBase, name="rule_based"): pass
# class MLStrat(StrategyBase, name="ml"): pass


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 2 — Metaclass validation
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Write metaclass `RequireDocstring` that raises TypeError at CLASS
# DEFINITION time if the class has no docstring.
# Apply it to `DocumentedStrategy`.

# TODO: class RequireDocstring(type): ...
# TODO: class DocumentedStrategy(metaclass=RequireDocstring): ...


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 3 — Singleton metaclass
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Write `SingletonMeta` metaclass.
# Apply it to `ConfigManager` (stores a `data: dict`).
# Verify that two calls to ConfigManager() return the SAME object.

# TODO: class SingletonMeta(type): ...
# TODO: class ConfigManager(metaclass=SingletonMeta): ...


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 4 — __init_subclass__ with kwargs injection
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Build `Versioned` base class where subclasses can declare their version:
#   class MyTool(Versioned, version="3.1"): ...
# __init_subclass__ should inject `cls.version` as a class attribute.
# If version not provided, default to "1.0".

# TODO: class Versioned: ...


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 5 — Metaclass auto-generating methods
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Write `AutoPropertyMeta` metaclass that:
#   Looks at the class namespace for any attribute starting with "_auto_"
#   and creates a @property named without the "_auto_" prefix.
#
# Example:
#   class Foo(metaclass=AutoPropertyMeta):
#       _auto_label = "EUR/USD"   → creates `label` property returning "EUR/USD"

# TODO: class AutoPropertyMeta(type): ...


# ─────────────────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_exercise_1():
    print("\n--- Exercise 1: __init_subclass__ registry ---")

    class RuleBased(StrategyBase, name="rule_based"):
        pass

    class MLStrat(StrategyBase, name="ml"):
        pass

    assert StrategyBase.get("rule_based") is RuleBased
    assert StrategyBase.get("ml") is MLStrat
    assert "rule_based" in StrategyBase.list_all()
    print(f"  registry: {StrategyBase.list_all()}  PASS")


def test_exercise_2():
    print("\n--- Exercise 2: RequireDocstring metaclass ---")

    class GoodClass(DocumentedStrategy):
        """I have a docstring."""
        pass

    try:
        class BadClass(DocumentedStrategy):
            pass   # no docstring
        print("  FAIL: should raise TypeError")
    except TypeError as e:
        print(f"  PASS: {e}")


def test_exercise_3():
    print("\n--- Exercise 3: Singleton ---")
    c1 = ConfigManager()
    c1.data = {"key": "value"}
    c2 = ConfigManager()
    assert c1 is c2, "ConfigManager should be singleton"
    assert c2.data["key"] == "value"
    print(f"  c1 is c2: {c1 is c2}  PASS")


def test_exercise_4():
    print("\n--- Exercise 4: Versioned ---")

    class ToolA(Versioned, version="2.3"):
        pass

    class ToolB(Versioned):
        pass

    assert ToolA.version == "2.3"
    assert ToolB.version == "1.0"
    print(f"  ToolA.version={ToolA.version}, ToolB.version={ToolB.version}  PASS")


def test_exercise_5():
    print("\n--- Exercise 5: AutoPropertyMeta ---")

    class Quote(metaclass=AutoPropertyMeta):
        _auto_pair  = "EUR_USD"
        _auto_price = 1.1050

    q = Quote()
    assert q.pair  == "EUR_USD"
    assert q.price == 1.1050
    print(f"  q.pair={q.pair}, q.price={q.price}  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("Topic 09 — Metaclasses — Exercise Runner")
    print("=" * 60)

    for fn in [test_exercise_1, test_exercise_2, test_exercise_3,
               test_exercise_4, test_exercise_5]:
        try:
            fn()
        except (NameError, AssertionError, TypeError) as e:
            print(f"  INCOMPLETE: {e}")

    print("\n" + "=" * 60)


# ── SOLUTIONS ─────────────────────────────────────────────────────────────────
"""
# Ex1
class StrategyBase:
    _registry: dict = {}
    def __init_subclass__(cls, name="", **kwargs):
        super().__init_subclass__(**kwargs)
        StrategyBase._registry[name or cls.__name__] = cls
    @classmethod
    def get(cls, name): return cls._registry[name]
    @classmethod
    def list_all(cls): return list(cls._registry.keys())

# Ex2
class RequireDocstring(type):
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        if bases and not namespace.get("__doc__"):
            raise TypeError(f"{name} must have a docstring")
        return cls

class DocumentedStrategy(metaclass=RequireDocstring):
    \"\"\"Base for documented strategies.\"\"\"

# Ex3
class SingletonMeta(type):
    _instances: dict = {}
    def __call__(cls, *a, **kw):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*a, **kw)
        return cls._instances[cls]

class ConfigManager(metaclass=SingletonMeta):
    def __init__(self): self.data = {}

# Ex4
class Versioned:
    def __init_subclass__(cls, version="1.0", **kwargs):
        super().__init_subclass__(**kwargs)
        cls.version = version

# Ex5
class AutoPropertyMeta(type):
    def __new__(mcs, name, bases, namespace):
        auto = {k: v for k, v in namespace.items() if k.startswith("_auto_")}
        for attr, value in auto.items():
            prop_name = attr[len("_auto_"):]
            namespace[prop_name] = property(lambda self, v=value: v)
        return super().__new__(mcs, name, bases, namespace)
"""
