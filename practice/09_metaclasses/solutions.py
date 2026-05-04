"""
Topic 09 — Metaclasses & __init_subclass__
============================================
SOLUTIONS
"""

from __future__ import annotations


class StrategyBase:
    _registry: dict = {}
    def __init_subclass__(cls, name: str = "", **kwargs):
        super().__init_subclass__(**kwargs)
        StrategyBase._registry[name or cls.__name__] = cls
    @classmethod
    def get(cls, name: str): return cls._registry[name]
    @classmethod
    def list_all(cls): return list(cls._registry.keys())


class RequireDocstring(type):
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        if bases and not namespace.get("__doc__"):
            raise TypeError(f"{name} must have a docstring")
        return cls

class DocumentedStrategy(metaclass=RequireDocstring):
    """Base class for documented strategies."""


class SingletonMeta(type):
    _instances: dict = {}
    def __call__(cls, *a, **kw):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*a, **kw)
        return cls._instances[cls]

class ConfigManager(metaclass=SingletonMeta):
    def __init__(self): self.data = {}


class Versioned:
    def __init_subclass__(cls, version: str = "1.0", **kwargs):
        super().__init_subclass__(**kwargs)
        cls.version = version


class AutoPropertyMeta(type):
    def __new__(mcs, name, bases, namespace):
        auto = {k: v for k, v in namespace.items() if k.startswith("_auto_")}
        for attr, value in auto.items():
            prop_name = attr[len("_auto_"):]
            namespace[prop_name] = property(lambda self, v=value: v)
        return super().__new__(mcs, name, bases, namespace)


if __name__ == "__main__":
    print("=" * 60)
    print("Topic 09 — Metaclasses — SOLUTIONS")
    print("=" * 60)

    class RuleBased(StrategyBase, name="rule_based"): pass
    class MLStrat(StrategyBase, name="ml"): pass
    assert StrategyBase.get("rule_based") is RuleBased
    print("\n  Ex1 registry: PASS")

    class Good(DocumentedStrategy):
        """Has docstring."""
    try:
        class Bad(DocumentedStrategy): pass
    except TypeError:
        print("  Ex2 RequireDocstring: PASS")

    c1, c2 = ConfigManager(), ConfigManager()
    assert c1 is c2
    print("  Ex3 Singleton: PASS")

    class ToolA(Versioned, version="2.3"): pass
    class ToolB(Versioned): pass
    assert ToolA.version == "2.3" and ToolB.version == "1.0"
    print("  Ex4 Versioned: PASS")

    class Quote(metaclass=AutoPropertyMeta):
        _auto_pair  = "EUR_USD"
        _auto_price = 1.1050
    q = Quote()
    assert q.pair == "EUR_USD" and q.price == 1.1050
    print("  Ex5 AutoPropertyMeta: PASS")

    print("\nAll solutions verified!")
