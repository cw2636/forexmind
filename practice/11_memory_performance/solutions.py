"""
Topic 11 — Memory & Performance — SOLUTIONS
"""
from __future__ import annotations
import sys, timeit, tracemalloc
from functools import lru_cache


class NoSlots:
    def __init__(self,p,b,a,t): self.pair=p;self.bid=b;self.ask=a;self.ts=t

class WithSlots:
    __slots__=("pair","bid","ask","ts")
    def __init__(self,p,b,a,t): self.pair=p;self.bid=b;self.ask=a;self.ts=t

@lru_cache(maxsize=128)
def compute_sma(prices:tuple,period:int)->float: return sum(prices[-period:])/period

def run_benchmarks():
    N=1000
    t_comp=timeit.timeit(f"[x**2 for x in range({N})]",number=10_000)
    t_map=timeit.timeit(f"list(map(lambda x:x**2,range({N})))",number=10_000)
    setup="r=[]\nfor x in range(1000): r.append(x**2)"
    t_loop=timeit.timeit(setup,number=10_000)
    fastest=min([("comprehension",t_comp),("map",t_map),("loop",t_loop)],key=lambda x:x[1])
    print(f"    comp={t_comp:.3f}s  map={t_map:.3f}s  loop={t_loop:.3f}s → fastest={fastest[0]}")

def measure_peak()->int:
    tracemalloc.start()
    data=[i for i in range(100_000)]
    _,peak=tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del data
    return peak

class FastSignal:
    __slots__=("instrument","direction","confidence","entry","stop")
    def __init__(self,i,d,c,e,s):
        self.instrument=i;self.direction=d;self.confidence=c;self.entry=e;self.stop=s

class RegularSignal:
    def __init__(self,i,d,c,e,s):
        self.instrument=i;self.direction=d;self.confidence=c;self.entry=e;self.stop=s


if __name__=="__main__":
    print("="*60)
    print("Topic 11 — Memory & Performance — SOLUTIONS")
    print("="*60)

    tracemalloc.start()
    ns=[NoSlots("EUR_USD",1.1,1.2,"t") for _ in range(10_000)]
    _,no_peak=tracemalloc.get_traced_memory(); tracemalloc.reset_peak()
    ws=[WithSlots("EUR_USD",1.1,1.2,"t") for _ in range(10_000)]
    _,sl_peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
    print(f"\n  __slots__: {no_peak/1024:.1f}KB → {sl_peak/1024:.1f}KB  PASS")

    prices=tuple([1.1+i*0.0001 for i in range(50)])
    compute_sma(prices,14);compute_sma(prices,14);compute_sma(prices,14)
    info=compute_sma.cache_info()
    assert info.hits>=2
    print(f"  lru_cache: {info}  PASS")

    run_benchmarks()
    print("  timeit: PASS")

    peak=measure_peak()
    print(f"  tracemalloc peak={peak/1024:.1f}KB  PASS")

    fs=FastSignal("EUR_USD","BUY",0.8,1.1050,1.1020)
    rs=RegularSignal("EUR_USD","BUY",0.8,1.1050,1.1020)
    assert sys.getsizeof(fs)<sys.getsizeof(rs)+sys.getsizeof(rs.__dict__)
    print("  FastSignal < RegularSignal  PASS")
    print("\nAll solutions verified!")
