"""
Topic 14 — Closures — SOLUTIONS (standalone)
"""
from __future__ import annotations


def make_pip_calculator(pair: str):
    multiplier = 100 if pair == "USD_JPY" else 10_000
    def calc(price_diff: float) -> float:
        return abs(price_diff) * multiplier
    return calc


def make_trade_tracker():
    position = 0
    def add(units: int) -> None:
        nonlocal position
        position += units
    def remove(units: int) -> None:
        nonlocal position
        position -= units
    def total() -> int:
        return position
    return add, remove, total


def inspect_closure(fn) -> None:
    names = fn.__code__.co_freevars
    cells = fn.__closure__ or ()
    print(f"  Free variables: {', '.join(names)}")
    for name, cell in zip(names, cells):
        print(f"    {name} = {cell.cell_contents}")


def memoize(fn):
    cache: dict = {}
    def wrapper(*args):
        if args not in cache:
            cache[args] = fn(*args)
        return cache[args]
    wrapper.cache = cache   # expose cache for inspection
    return wrapper


if __name__ == "__main__":
    print("=" * 60)
    print("Topic 14 — Closures — SOLUTIONS")
    print("=" * 60)

    # Ex1
    eur = make_pip_calculator("EUR_USD")
    jpy = make_pip_calculator("USD_JPY")
    assert abs(eur(0.003) - 30.0) < 0.01
    assert abs(jpy(0.30)  - 30.0) < 0.01
    print("\n  Ex1 pip calculator: PASS")
    print(f"    __closure__ cell: {eur.__closure__[0].cell_contents}")

    # Ex2
    add, remove, total = make_trade_tracker()
    add(1000); add(500); remove(300)
    assert total() == 1200
    print("  Ex2 trade tracker: PASS")

    # Ex3
    bad = [lambda: i for i in range(5)]
    assert all(f() == 4 for f in bad)
    good = [lambda i=i: i for i in range(5)]
    assert [f() for f in good] == [0,1,2,3,4]
    print("  Ex3 late binding: PASS")

    # Ex4
    def outer(x,y):
        def inner(z): return x+y+z
        return inner
    inspect_closure(outer(10,20))
    print("  Ex4 inspect_closure: PASS")

    # Ex5
    calls = [0]
    @memoize
    def sq(n):
        calls[0] += 1
        return n*n
    sq(5); sq(5); sq(5)
    assert calls[0] == 1
    print(f"  Ex5 memoize: {calls[0]} real call  PASS")

    print("\nAll solutions verified!")
