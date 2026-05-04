"""
Topic 10 — Design Patterns — SOLUTIONS (standalone)
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable


class SizingStrategy(ABC):
    @abstractmethod
    def calculate(self, risk_pct: float, balance: float) -> float: ...

class FixedRisk(SizingStrategy):
    def calculate(self, risk_pct, balance): return balance * (risk_pct / 100)

class KellyHalf(SizingStrategy):
    def __init__(self, win_rate, avg_win, avg_loss):
        self.win_rate=win_rate; self.avg_win=avg_win; self.avg_loss=avg_loss
    def calculate(self, risk_pct, balance):
        k = self.win_rate - (1-self.win_rate)*(self.avg_loss/self.avg_win)
        return balance * max(0,k) * 0.5

class PositionSizer:
    def __init__(self, s): self._s = s
    def set_strategy(self, s): self._s = s
    def size(self, risk_pct, balance): return self._s.calculate(risk_pct, balance)


class EventBus:
    def __init__(self): self._l: dict = {}
    def subscribe(self, e, cb): self._l.setdefault(e,[]).append(cb)
    def publish(self, e, **d): [cb(**d) for cb in self._l.get(e,[])]


class _RuleBased: name="rule_based"
class _ML: name="ml"
class StrategyFactory:
    _reg = {"rule_based":_RuleBased,"ml":_ML}
    @classmethod
    def create(cls,name):
        if name not in cls._reg: raise ValueError(f"Unknown:{name!r}. Valid:{list(cls._reg)}")
        return cls._reg[name]()


class Command(ABC):
    @abstractmethod
    def execute(self): ...
    @abstractmethod
    def undo(self): ...

class PlaceTrade(Command):
    def __init__(self,pair,units): self.pair=pair;self.units=units
    def execute(self): print(f"    Place {self.pair} x{self.units}")
    def undo(self): print(f"    Cancel {self.pair}")
    def __str__(self): return f"Place({self.pair})"

class CloseTrade(Command):
    def __init__(self,tid): self.tid=tid
    def execute(self): print(f"    Close {self.tid}")
    def undo(self): print(f"    Reopen {self.tid}")
    def __str__(self): return f"Close({self.tid})"

class TradeHistory:
    def __init__(self): self.done=[]
    def execute(self,cmd): cmd.execute(); self.done.append(cmd)
    def undo_last(self):
        if self.done: self.done.pop().undo()


class AnalysisPipeline(ABC):
    def __init__(self,pair): self.pair=pair
    def run(self):
        data = self.fetch_data()
        inds = self.compute_indicators(data)
        result = self.generate_signal(inds)
        self.log_result(result)
        return result
    def fetch_data(self): return [1.1050,1.1060,1.1055]
    def compute_indicators(self,d): return {"macd":d[-1]-d[0]}
    @abstractmethod
    def generate_signal(self,indicators): ...
    def log_result(self,r): print(f"    Signal:{r}")

class MACDPipeline(AnalysisPipeline):
    def generate_signal(self,inds):
        return {"direction":"BUY" if inds["macd"]>0 else "SELL"}


if __name__ == "__main__":
    print("=" * 60)
    print("Topic 10 — Design Patterns — SOLUTIONS")
    print("=" * 60)

    s = PositionSizer(FixedRisk())
    assert s.size(2.0,10000)==200.0
    print("\n  Ex1 Strategy: PASS")

    bus,log = EventBus(),[]
    bus.subscribe("sig",lambda **d: log.append(d))
    bus.publish("sig",direction="BUY",pair="EUR_USD")
    assert len(log)==1
    print("  Ex2 EventBus: PASS")

    s1=StrategyFactory.create("rule_based")
    assert s1.name=="rule_based"
    print("  Ex3 Factory: PASS")

    h=TradeHistory()
    h.execute(PlaceTrade("EUR_USD",1000))
    h.execute(PlaceTrade("GBP_USD",500))
    h.undo_last()
    assert len(h.done)==1
    print("  Ex4 Command: PASS")

    p=MACDPipeline("EUR_USD")
    r=p.run()
    assert r["direction"] in ("BUY","SELL")
    print("  Ex5 Template: PASS")
    print("\nAll solutions verified!")
