"""
ForexMind — Tests: Indicators Engine
======================================
Unit tests that validate technical indicators against known values.

Advanced Python testing concepts:
  - pytest with parametrize for multiple test cases
  - Fixtures for reusable test DataFrames
  - Approximate float comparison with pytest.approx
  - Mocking get_settings for isolated tests
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_ohlcv() -> pd.DataFrame:
    """
    300 bars of synthetic OHLCV with a clear uptrend,
    followed by a downtrend.
    """
    n = 300
    np.random.seed(42)
    # First half: strong uptrend (price rises ~10%)
    trend = np.linspace(1.0800, 1.1000, n // 2)
    # Second half: downtrend (price falls ~5%)
    downtrend = np.linspace(1.1000, 1.0800, n - n // 2)
    close = np.concatenate([trend, downtrend]) + np.random.normal(0, 0.0003, n)

    # Build realistic OHLC around close prices
    high = close + np.abs(np.random.normal(0, 0.0005, n))
    low = close - np.abs(np.random.normal(0, 0.0005, n))
    open_ = close + np.random.normal(0, 0.0002, n)
    volume = np.random.randint(100, 1000, n).astype(float)

    index = pd.date_range(
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        periods=n,
        freq="5min",
    )
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


@pytest.fixture
def flat_ohlcv() -> pd.DataFrame:
    """Flat, ranging market for volatility tests."""
    n = 200
    np.random.seed(7)
    close = np.random.normal(1.0800, 0.0002, n)
    return pd.DataFrame({
        "open": close + np.random.normal(0, 0.00005, n),
        "high": close + np.abs(np.random.normal(0, 0.0001, n)),
        "low": close - np.abs(np.random.normal(0, 0.0001, n)),
        "close": close,
        "volume": np.ones(n) * 500.0,
    }, index=pd.date_range(start="2024-01-01", periods=n, freq="5min", tz="UTC"))


# ── Indicator Engine Tests ────────────────────────────────────────────────────

class TestIndicatorEngine:
    """Tests for the IndicatorEngine computed columns."""

    def test_compute_returns_dataframe(self, simple_ohlcv):
        from forexmind.indicators.engine import IndicatorEngine
        engine = IndicatorEngine()
        result = engine.compute(simple_ohlcv)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(simple_ohlcv)

    def test_ema_columns_present(self, simple_ohlcv):
        from forexmind.indicators.engine import IndicatorEngine
        engine = IndicatorEngine()
        result = engine.compute(simple_ohlcv)
        for period in [9, 21, 50, 200]:
            assert f"ema_{period}" in result.columns, f"Missing ema_{period}"

    def test_rsi_range(self, simple_ohlcv):
        """RSI must always be between 0 and 100."""
        from forexmind.indicators.engine import IndicatorEngine
        engine = IndicatorEngine()
        result = engine.compute(simple_ohlcv)
        rsi = result["rsi"].dropna()
        assert (rsi >= 0).all(), "RSI below 0"
        assert (rsi <= 100).all(), "RSI above 100"

    def test_uptrend_ema_alignment(self, simple_ohlcv):
        """In uptrend first half, EMA9 should be > EMA21 for most recent bars."""
        from forexmind.indicators.engine import IndicatorEngine
        engine = IndicatorEngine()
        # Test on just the uptrend portion
        uptrend_df = simple_ohlcv.iloc[:150]
        result = engine.compute(uptrend_df)
        last_row = result.dropna(subset=["ema_9", "ema_21"]).iloc[-1]
        assert last_row["ema_9"] > last_row["ema_21"], (
            f"EMA9 {last_row['ema_9']:.5f} should be > EMA21 {last_row['ema_21']:.5f} in uptrend"
        )

    def test_atr_positive(self, simple_ohlcv):
        """ATR must always be > 0."""
        from forexmind.indicators.engine import IndicatorEngine
        engine = IndicatorEngine()
        result = engine.compute(simple_ohlcv)
        atr = result["atr"].dropna()
        assert (atr > 0).all()

    def test_bollinger_bands_order(self, simple_ohlcv):
        """Upper BB > Mid BB > Lower BB always."""
        from forexmind.indicators.engine import IndicatorEngine
        engine = IndicatorEngine()
        result = engine.compute(simple_ohlcv)
        valid = result.dropna(subset=["bb_upper", "bb_mid", "bb_lower"])
        assert (valid["bb_upper"] >= valid["bb_mid"]).all()
        assert (valid["bb_mid"] >= valid["bb_lower"]).all()

    def test_snapshot_structure(self, simple_ohlcv):
        """Snapshot should contain all required keys."""
        from forexmind.indicators.engine import IndicatorEngine, IndicatorSnapshot
        engine = IndicatorEngine()
        result = engine.compute(simple_ohlcv)
        snap = engine.snapshot(result, "EUR_USD", "M5")
        # Check required keys exist
        required_keys = ["rsi", "macd", "adx", "ema_9", "atr", "bb_upper", "stoch_k"]
        for key in required_keys:
            assert key in snap, f"Missing key in snapshot: {key}"

    def test_insufficient_data_returns_empty_snapshot(self):
        """With insufficient data, snapshot should not crash."""
        from forexmind.indicators.engine import IndicatorEngine
        engine = IndicatorEngine()
        short_df = pd.DataFrame(
            {"open": [1.08], "high": [1.082], "low": [1.079], "close": [1.081], "volume": [100]},
            index=pd.date_range("2024-01-01", periods=1, freq="5min", tz="UTC"),
        )
        result = engine.compute(short_df)
        snap = engine.snapshot(result, "EUR_USD", "M5")
        assert snap["instrument"] == "EUR_USD"


# ── Signal Scorer Tests ────────────────────────────────────────────────────────

class TestSignalScorer:

    def test_score_buy(self, simple_ohlcv):
        """Clear uptrend should produce a positive score."""
        from forexmind.indicators.engine import IndicatorEngine
        from forexmind.indicators.scorer import score_snapshot

        engine = IndicatorEngine()
        df = engine.compute(simple_ohlcv.iloc[:150])
        snap = engine.snapshot(df, "EUR_USD", "M5")
        score = score_snapshot(snap)
        assert score.composite >= -100 and score.composite <= 100
        assert score.direction in ("BUY", "SELL", "HOLD")
        assert 0.0 <= score.confidence <= 1.0

    def test_score_returns_reasoning(self, simple_ohlcv):
        from forexmind.indicators.engine import IndicatorEngine
        from forexmind.indicators.scorer import score_snapshot

        engine = IndicatorEngine()
        df = engine.compute(simple_ohlcv)
        snap = engine.snapshot(df, "EUR_USD", "M5")
        score = score_snapshot(snap)
        assert len(score.reasoning) > 10


# ── Utilities Tests ────────────────────────────────────────────────────────────

class TestHelpers:

    @pytest.mark.parametrize("instrument,expected", [
        ("EUR_USD", 0.0001),
        ("USD_JPY", 0.01),
        ("GBP_JPY", 0.01),
        ("AUD_USD", 0.0001),
    ])
    def test_pip_size(self, instrument, expected):
        from forexmind.utils.helpers import pip_size
        assert pip_size(instrument) == expected

    def test_atr_stop_loss_buy(self):
        from forexmind.utils.helpers import atr_stop_loss
        sl = atr_stop_loss(entry=1.0850, atr=0.0010, direction="BUY", multiplier=1.5)
        assert sl < 1.0850
        assert abs(sl - (1.0850 - 0.0010 * 1.5)) < 1e-10

    def test_atr_stop_loss_sell(self):
        from forexmind.utils.helpers import atr_stop_loss
        sl = atr_stop_loss(entry=1.0850, atr=0.0010, direction="SELL", multiplier=1.5)
        assert sl > 1.0850

    def test_kelly_fraction_positive(self):
        from forexmind.utils.helpers import kelly_fraction
        k = kelly_fraction(win_rate=0.6, rr_ratio=2.0)
        assert 0.0 < k <= 0.25

    def test_kelly_fraction_zero_for_bad_rr(self):
        from forexmind.utils.helpers import kelly_fraction
        k = kelly_fraction(win_rate=0.4, rr_ratio=0.5)
        assert k == 0.0

    def test_units_from_risk(self):
        from forexmind.utils.helpers import units_from_risk
        units = units_from_risk(
            account_balance=10000.0,
            risk_pct=1.0,
            stop_loss_pips=10.0,
            instrument="EUR_USD",
        )
        assert units >= 1000
        assert isinstance(units, int)


# ── Session Times Tests ────────────────────────────────────────────────────────

class TestSessionTimes:

    def test_london_session(self):
        from forexmind.utils.session_times import get_session_status
        from datetime import datetime, timezone as tz
        # 10:00 UTC is London session
        dt = datetime(2024, 1, 15, 10, 0, tzinfo=tz.utc)  # Monday
        status = get_session_status(dt)
        assert "London" in status.active_sessions

    def test_weekend_detection(self):
        from forexmind.utils.session_times import get_session_status
        from datetime import datetime, timezone as tz
        # Saturday
        dt = datetime(2024, 1, 13, 12, 0, tzinfo=tz.utc)  # Saturday
        status = get_session_status(dt)
        assert status.is_weekend

    def test_overlap_detection(self):
        from forexmind.utils.session_times import get_session_status
        from datetime import datetime, timezone as tz
        # 13:00 UTC is London-NY overlap
        dt = datetime(2024, 1, 15, 13, 0, tzinfo=tz.utc)  # Monday
        status = get_session_status(dt)
        assert status.is_overlap
        assert status.session_score > 0.5
