"""
Topic 02 — Dataclasses Deep-Dive
=================================
EXERCISES — Fill in every section marked TODO.

Run this file:
    python 02_dataclasses/exercises.py
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, asdict
from datetime import datetime, timezone
from typing import ClassVar, Optional


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 1 — Basic dataclass + mutable default trap
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Create a `TradeTag` dataclass with:
#   - `label: str`       — required
#   - `priority: int`     — default 1
#   - `notes: list`       — mutable default (use field!) starts empty
#   - `created_at: datetime` — auto-set to current UTC time, NOT in __init__
#
# Then show the mutable default trap by creating two instances and
# demonstrating that their `notes` lists are independent.

# TODO: Define TradeTag here


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 2 — __post_init__ validation
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Create `StrategySignal` dataclass with:
#   - instrument: str  (normalise "/" → "_" in __post_init__)
#   - direction: str   (validate: must be "BUY", "SELL", or "HOLD")
#   - confidence: float (validate: 0.0 to 1.0 inclusive)
#   - entry_price: float
#   - stop_loss: float = 0.0
#   - take_profit: float = 0.0
#   - source: str = "unknown"
#   - risk_reward: float — computed field (init=False), calculated in __post_init__
#                          as abs(take_profit - entry_price) / abs(entry_price - stop_loss)
#                          use 0.0 if stop_loss is 0
#
# __post_init__ must raise ValueError for invalid direction or confidence.

# TODO: Define StrategySignal here


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 3 — frozen dataclass
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Create `TradingPair` as a FROZEN dataclass with:
#   - base: str    (e.g. "EUR")
#   - quote: str   (e.g. "USD")
#   - A @property `symbol` that returns "EUR_USD"
#
# Demonstrate that:
#   1. You cannot reassign base or quote after creation
#   2. It can be used as a dictionary key
#   3. Two TradingPairs with the same base/quote are equal

# TODO: Define TradingPair here


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 4 — field() metadata + introspection
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Create `RiskProposal` dataclass where:
#   - entry_price: float  — metadata={"unit": "price"}
#   - stop_loss: float    — metadata={"unit": "price"}
#   - take_profit: float  — metadata={"unit": "price"}
#   - units: int          — metadata={"unit": "contracts"}
#   - risk_pct: float     — metadata={"unit": "percent"}, default 1.0
#   - risk_usd: float     — init=False, repr=False, computed as risk_pct/100 * 10000
#                           (simplified: assume 10k account)
#   - VERSION: ClassVar[str] = "2.0"
#
# Write a function `describe_fields(cls)` that:
#   - Takes a dataclass class
#   - Prints each field's name, type, and unit metadata (if present)

# TODO: Define RiskProposal here

# TODO: Define describe_fields(cls) here


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 5 — InitVar (advanced)
# ─────────────────────────────────────────────────────────────────────────────
# Goal: Create `FormattedPrice` dataclass with:
#   - raw: float     — stored
#   - precision: InitVar[int] = 5  — passed to __post_init__ but NOT stored
#   - display: str   — init=False, set in __post_init__ as f"{raw:.{precision}f}"
#
# Example: FormattedPrice(1.10503, precision=4).display == "1.1050"

# TODO: Define FormattedPrice here (requires `from dataclasses import InitVar`)


# ─────────────────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_exercise_1():
    print("\n--- Exercise 1: TradeTag ---")

    t1 = TradeTag(label="EURUSD_entry")
    t2 = TradeTag(label="GBPUSD_entry", priority=3)

    # Mutable defaults must be independent
    t1.notes.append("first note")
    assert t2.notes == [], f"t2.notes should be empty, got {t2.notes}"
    assert t1.priority == 1
    assert t2.priority == 3
    assert isinstance(t1.created_at, datetime)
    print(f"  t1 = {t1}")
    print("  PASS: TradeTag works, mutable defaults are independent")


def test_exercise_2():
    print("\n--- Exercise 2: StrategySignal ---")

    # Valid signal
    sig = StrategySignal(
        instrument="EUR/USD",   # slash should be normalised
        direction="BUY",
        confidence=0.75,
        entry_price=1.1050,
        stop_loss=1.1020,
        take_profit=1.1110,
    )
    assert sig.instrument == "EUR_USD", f"Expected EUR_USD, got {sig.instrument}"
    assert sig.risk_reward > 0, "risk_reward should be computed"
    print(f"  signal = {sig}")

    # Invalid confidence
    try:
        StrategySignal("EUR_USD", "BUY", confidence=1.5, entry_price=1.1)
        print("  FAIL: should raise ValueError")
    except (ValueError, TypeError) as e:
        print(f"  PASS: ValueError for bad confidence: {e}")

    # Invalid direction
    try:
        StrategySignal("EUR_USD", "LONG", confidence=0.7, entry_price=1.1)
        print("  FAIL: should raise ValueError")
    except (ValueError, TypeError) as e:
        print(f"  PASS: ValueError for bad direction: {e}")


def test_exercise_3():
    print("\n--- Exercise 3: TradingPair ---")

    eur_usd = TradingPair(base="EUR", quote="USD")
    assert eur_usd.symbol == "EUR_USD"

    # Frozen — cannot reassign
    try:
        eur_usd.base = "GBP"
        print("  FAIL: should raise FrozenInstanceError")
    except Exception as e:
        print(f"  PASS: Immutable: {type(e).__name__}")

    # Hashable — can be dict key
    prices = {eur_usd: 1.1050}
    assert prices[TradingPair("EUR", "USD")] == 1.1050
    print("  PASS: TradingPair is hashable and equal by value")


def test_exercise_4():
    print("\n--- Exercise 4: RiskProposal + describe_fields ---")

    rp = RiskProposal(entry_price=1.1050, stop_loss=1.1020, take_profit=1.1110,
                      units=1000, risk_pct=2.0)
    assert rp.risk_usd == 200.0, f"Expected 200.0, got {rp.risk_usd}"
    assert RiskProposal.VERSION == "2.0"
    print(f"  risk_usd = {rp.risk_usd}")
    print("\n  Field descriptions:")
    describe_fields(RiskProposal)
    print("  PASS")


def test_exercise_5():
    print("\n--- Exercise 5: FormattedPrice (InitVar) ---")

    p = FormattedPrice(1.10503, precision=4)
    assert p.display == "1.1050", f"Expected '1.1050', got '{p.display}'"
    assert p.raw == 1.10503
    try:
        _ = p.precision   # should NOT exist as an attribute
        print("  FAIL: precision should not be stored as attribute")
    except AttributeError:
        print("  PASS: precision not stored as attribute")
    print(f"  p = {p}")


if __name__ == "__main__":
    print("=" * 60)
    print("Topic 02 — Dataclasses — Exercise Runner")
    print("=" * 60)

    for fn in [test_exercise_1, test_exercise_2, test_exercise_3,
               test_exercise_4, test_exercise_5]:
        try:
            fn()
        except (NameError, AssertionError, TypeError) as e:
            print(f"  INCOMPLETE: {e}")

    print("\n" + "=" * 60)
