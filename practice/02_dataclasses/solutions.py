"""
Topic 02 — Dataclasses Deep-Dive
=================================
SOLUTIONS — Complete working implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, asdict, InitVar
from datetime import datetime, timezone
from typing import ClassVar, Optional


# ─────────────────────────────────────────────────────────────────────────────
# SOLUTION 1 — TradeTag
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeTag:
    label:      str                                           # required, no default
    priority:   int  = 1                                      # scalar default — OK
    notes:      list = field(default_factory=list)            # mutable default — MUST use field()
    created_at: datetime = field(                             # auto-set, not in __init__
        default_factory=lambda: datetime.now(timezone.utc),
        init=False,                                           # caller cannot set this
    )

    # Why field(default_factory=list)?
    # If you wrote `notes: list = []`, that SAME list object would be shared
    # across all TradeTag instances — a classic Python bug.
    # default_factory=list calls list() freshly for every new instance.


# ─────────────────────────────────────────────────────────────────────────────
# SOLUTION 2 — StrategySignal with __post_init__
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StrategySignal:
    instrument:  str
    direction:   str
    confidence:  float
    entry_price: float
    stop_loss:   float = 0.0
    take_profit: float = 0.0
    source:      str   = "unknown"

    # init=False: this field is NOT part of __init__ signature.
    # You must set it in __post_init__.
    risk_reward: float = field(init=False, default=0.0)

    def __post_init__(self):
        """
        Called automatically by the generated __init__ AFTER all fields are set.
        Use for:
         - Validation
         - Normalisation
         - Computing derived fields (init=False)
        """
        # Normalise instrument
        self.instrument = self.instrument.replace("/", "_")

        # Validate direction
        valid_directions = {"BUY", "SELL", "HOLD"}
        if self.direction not in valid_directions:
            raise ValueError(
                f"direction must be one of {valid_directions}, got {self.direction!r}"
            )

        # Validate confidence range
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {self.confidence}"
            )

        # Compute derived field (because init=False, we set it here)
        risk   = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.entry_price)
        self.risk_reward = reward / risk if risk > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# SOLUTION 3 — TradingPair (frozen)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)    # immutable — any mutation raises FrozenInstanceError
class TradingPair:
    base:  str             # e.g. "EUR"
    quote: str             # e.g. "USD"

    @property
    def symbol(self) -> str:
        return f"{self.base}_{self.quote}"

    # Because frozen=True, Python auto-generates __hash__
    # (based on base + quote), making TradingPair usable as dict keys and in sets.
    # Without frozen=True, @dataclass sets __hash__ = None (unhashable)
    # unless you explicitly set unsafe_hash=True.


# ─────────────────────────────────────────────────────────────────────────────
# SOLUTION 4 — RiskProposal with metadata + describe_fields
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RiskProposal:
    entry_price: float = field(metadata={"unit": "price"})
    stop_loss:   float = field(metadata={"unit": "price"})
    take_profit: float = field(metadata={"unit": "price"})
    units:       int   = field(metadata={"unit": "contracts"})
    risk_pct:    float = field(default=1.0, metadata={"unit": "percent"})

    # init=False and repr=False: computed internally, hidden from repr
    risk_usd: float = field(init=False, repr=False)

    VERSION: ClassVar[str] = "2.0"   # ClassVar → excluded from fields() entirely

    def __post_init__(self):
        # Simplified: assume $10,000 account
        self.risk_usd = (self.risk_pct / 100) * 10_000


def describe_fields(cls) -> None:
    """
    Print all dataclass fields with their name, type annotation, and metadata.

    `fields(cls)` returns a tuple of Field objects with attributes:
      .name, .type, .default, .default_factory, .metadata, .init, .repr
    """
    print(f"  Fields of {cls.__name__}:")
    for f in fields(cls):
        unit = f.metadata.get("unit", "—")
        print(f"    {f.name}: {f.type}  [unit={unit}]")


# ─────────────────────────────────────────────────────────────────────────────
# SOLUTION 5 — FormattedPrice with InitVar
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FormattedPrice:
    """
    InitVar: an argument passed to __post_init__ that is NOT stored as an attribute.

    Use case: configuration that influences field setup but doesn't need to
    persist after construction (e.g. precision, rounding mode, locale).
    """
    raw:       float                    # stored
    precision: InitVar[int] = 5         # passed to __post_init__ only — NOT stored
    display:   str = field(init=False)  # computed in __post_init__

    def __post_init__(self, precision: int):
        # `precision` is passed as argument here, then discarded
        self.display = f"{self.raw:.{precision}f}"


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_exercise_1():
    print("\n--- Exercise 1: TradeTag ---")
    t1 = TradeTag(label="EURUSD_entry")
    t2 = TradeTag(label="GBPUSD_entry", priority=3)
    t1.notes.append("first note")
    assert t2.notes == []
    assert t1.priority == 1
    assert t2.priority == 3
    assert isinstance(t1.created_at, datetime)
    print(f"  t1 = {t1}")
    print("  PASS")


def test_exercise_2():
    print("\n--- Exercise 2: StrategySignal ---")
    sig = StrategySignal("EUR/USD", "BUY", 0.75, 1.1050, 1.1020, 1.1110)
    assert sig.instrument == "EUR_USD"
    assert sig.risk_reward > 0
    print(f"  signal.risk_reward = {sig.risk_reward:.2f}")

    try:
        StrategySignal("EUR_USD", "BUY", confidence=1.5, entry_price=1.1)
    except ValueError as e:
        print(f"  PASS: {e}")

    try:
        StrategySignal("EUR_USD", "LONG", confidence=0.7, entry_price=1.1)
    except ValueError as e:
        print(f"  PASS: {e}")


def test_exercise_3():
    print("\n--- Exercise 3: TradingPair ---")
    pair = TradingPair("EUR", "USD")
    assert pair.symbol == "EUR_USD"
    try:
        pair.base = "GBP"
    except Exception as e:
        print(f"  PASS: {type(e).__name__}: {e}")
    prices = {pair: 1.1050}
    assert prices[TradingPair("EUR", "USD")] == 1.1050
    print("  PASS: hashable + equal by value")


def test_exercise_4():
    print("\n--- Exercise 4: RiskProposal ---")
    rp = RiskProposal(1.1050, 1.1020, 1.1110, 1000, risk_pct=2.0)
    assert rp.risk_usd == 200.0
    assert RiskProposal.VERSION == "2.0"
    print(f"  risk_usd = {rp.risk_usd}")
    describe_fields(RiskProposal)
    print("  PASS")


def test_exercise_5():
    print("\n--- Exercise 5: FormattedPrice ---")
    p = FormattedPrice(1.10503, precision=4)
    assert p.display == "1.1050"
    assert p.raw == 1.10503
    try:
        _ = p.precision
        print("  FAIL: precision should not exist")
    except AttributeError:
        print("  PASS: precision not stored")
    print(f"  p = {p}")


if __name__ == "__main__":
    print("=" * 60)
    print("Topic 02 — Dataclasses — SOLUTIONS")
    print("=" * 60)
    test_exercise_1()
    test_exercise_2()
    test_exercise_3()
    test_exercise_4()
    test_exercise_5()
    print("\n" + "=" * 60)
    print("All solutions verified!")
