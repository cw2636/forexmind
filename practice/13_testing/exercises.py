"""
Topic 13 — Testing Mastery
============================
EXERCISES + SOLUTIONS

Run:  pytest 13_testing/exercises.py -v
"""

from __future__ import annotations
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM UNDER TEST (the code we'll be testing)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Signal:
    pair: str
    direction: str
    confidence: float
    stop_loss:  float
    take_profit: float
    entry_price: float

    @property
    def risk_reward(self):
        risk   = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.entry_price)
        return reward / risk if risk > 0 else 0.0

    @property
    def is_valid(self):
        return (self.direction in ("BUY","SELL","HOLD")
                and 0 <= self.confidence <= 1
                and self.risk_reward >= 1.5)


class OandaClient:
    """Real API client — we'll mock this in tests."""

    def get_price(self, pair: str) -> float:
        raise NotImplementedError("Real API")

    def place_order(self, pair: str, units: int) -> dict:
        raise NotImplementedError("Real API")


class TradingEngine:
    def __init__(self, client: OandaClient, min_confidence: float = 0.6):
        self.client = client
        self.min_confidence = min_confidence
        self.trades: list = []

    def process_signal(self, signal: Signal) -> dict:
        if not signal.is_valid:
            return {"status": "rejected", "reason": "invalid signal"}
        if signal.confidence < self.min_confidence:
            return {"status": "rejected", "reason": "low confidence"}
        price = self.client.get_price(signal.pair)
        result = self.client.place_order(signal.pair, units=1000)
        self.trades.append(signal)
        return {"status": "placed", "price": price, "order": result}


# ─────────────────────────────────────────────────────────────────────────────
# LESSON: Key pytest features used below
#
# @pytest.fixture          — reusable setup/teardown
# @pytest.mark.parametrize — run one test with multiple inputs
# MagicMock                — replace an object with a controllable fake
# patch()                  — temporarily replace a module-level object
# monkeypatch              — pytest's built-in patching tool (fixture)
# capfd / capsys           — capture stdout/stderr
# pytest.raises            — assert that an exception is raised
# pytest.approx            — approximate float comparison


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 1 — @pytest.fixture
# ─────────────────────────────────────────────────────────────────────────────
# Create a fixture `valid_signal` that returns a Signal with:
#   pair="EUR_USD", direction="BUY", confidence=0.8,
#   entry_price=1.1050, stop_loss=1.1020, take_profit=1.1140

# TODO: @pytest.fixture
# def valid_signal(): ...

def test_signal_risk_reward(valid_signal):
    """Uses the fixture — no setup code in the test body."""
    assert valid_signal.risk_reward == pytest.approx(3.0, rel=0.01)

def test_signal_is_valid(valid_signal):
    assert valid_signal.is_valid is True


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 2 — @pytest.mark.parametrize
# ─────────────────────────────────────────────────────────────────────────────
# Write ONE test that validates is_valid for multiple cases:
#   (direction, confidence, expected_valid)
#   ("BUY",  0.8, ...valid...)
#   ("SELL", 0.3, ...but confidence may still be valid — check the property)
#   ("HOLD", 0.5, False)  — HOLD has direction in set but rr=0
#   ("LONG", 0.7, False)  — invalid direction

# TODO: @pytest.mark.parametrize(...)
# def test_signal_validity(direction, confidence, expected): ...


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 3 — MagicMock
# ─────────────────────────────────────────────────────────────────────────────
# Test TradingEngine.process_signal with a mocked OandaClient:
# - Set client.get_price.return_value = 1.1050
# - Set client.place_order.return_value = {"order_id": "T123"}
# - Assert engine returns status="placed"
# - Assert client.get_price.called is True
# - Assert client.place_order.called is True

# TODO: def test_process_valid_signal(): ...


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 4 — patch() context manager
# ─────────────────────────────────────────────────────────────────────────────
# Use `with patch("__main__.OandaClient.get_price") as mock_price:` to
# patch the class method in the engine and test rejection of low-confidence signals.

# TODO: def test_low_confidence_rejection(): ...


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 5 — pytest.raises + error message checking
# ─────────────────────────────────────────────────────────────────────────────
# Our Signal.risk_reward uses abs() so it won't raise, but let's test
# a function that SHOULD raise. Write `calculate_position_size(balance, risk_pct)`
# that raises ValueError if risk_pct > 10 or balance <= 0.
# Use pytest.raises to verify both conditions.

def calculate_position_size(balance: float, risk_pct: float) -> float:
    if balance <= 0:
        raise ValueError(f"balance must be positive, got {balance}")
    if risk_pct > 10:
        raise ValueError(f"risk_pct must be <= 10, got {risk_pct}")
    return balance * (risk_pct / 100)

# TODO: def test_position_size_errors(): ...
# Hint: use pytest.raises(ValueError, match="...")


# ─────────────────────────────────────────────────────────────────────────────
# SOLUTIONS
# ─────────────────────────────────────────────────────────────────────────────

# Exercise 1
@pytest.fixture
def valid_signal():
    return Signal(
        pair="EUR_USD", direction="BUY", confidence=0.8,
        entry_price=1.1050, stop_loss=1.1020, take_profit=1.1140,
    )

# Exercise 2
@pytest.mark.parametrize("direction,confidence,sl,tp,entry,expected", [
    ("BUY",  0.8, 1.1020, 1.1140, 1.1050, True),   # valid: rr=3.0
    ("SELL", 0.8, 1.1080, 1.0980, 1.1050, True),   # valid short
    ("HOLD", 0.5, 0.0, 0.0, 1.1050, False),         # rr=0 → invalid
    ("LONG", 0.7, 1.1020, 1.1140, 1.1050, False),  # bad direction
])
def test_signal_validity(direction, confidence, sl, tp, entry, expected):
    sig = Signal(pair="EUR_USD", direction=direction, confidence=confidence,
                 stop_loss=sl, take_profit=tp, entry_price=entry)
    assert sig.is_valid is expected

# Exercise 3
def test_process_valid_signal(valid_signal):
    mock_client = MagicMock(spec=OandaClient)
    mock_client.get_price.return_value = 1.1050
    mock_client.place_order.return_value = {"order_id": "T123"}

    engine = TradingEngine(mock_client, min_confidence=0.6)
    result = engine.process_signal(valid_signal)

    assert result["status"] == "placed"
    mock_client.get_price.assert_called_once_with("EUR_USD")
    mock_client.place_order.assert_called_once()

# Exercise 4
def test_low_confidence_rejection():
    mock_client = MagicMock(spec=OandaClient)
    engine = TradingEngine(mock_client, min_confidence=0.7)

    low_conf_signal = Signal(
        pair="EUR_USD", direction="BUY", confidence=0.5,
        entry_price=1.1050, stop_loss=1.1020, take_profit=1.1140,
    )
    result = engine.process_signal(low_conf_signal)
    assert result["status"] == "rejected"
    assert result["reason"] == "low confidence"
    mock_client.get_price.assert_not_called()

# Exercise 5
def test_position_size_errors():
    with pytest.raises(ValueError, match="positive"):
        calculate_position_size(balance=-100, risk_pct=2.0)

    with pytest.raises(ValueError, match="<= 10"):
        calculate_position_size(balance=10000, risk_pct=15.0)

    # Valid case
    result = calculate_position_size(10000, 2.0)
    assert result == pytest.approx(200.0)


# ── Lesson notes shown when running directly ─────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Topic 13 — Testing")
    print("Run with: pytest 13_testing/exercises.py -v")
    print("=" * 60)
    print("""
Key pytest concepts used:
  @pytest.fixture        — shared setup (valid_signal)
  @pytest.mark.parametrize — data-driven tests (test_signal_validity)
  MagicMock(spec=X)      — mock with type checking (test_process_valid_signal)
  .return_value          — control what mock returns
  .assert_called_once_with() — verify mock was called correctly
  pytest.raises(E, match="regex") — assert exception + message
  pytest.approx(v, rel=0.01)    — float comparison with tolerance
""")
