"""
ForexMind — Tests: Risk Management & Strategy
==============================================
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """200 bars of uptrending data with full indicator computation."""
    n = 200
    np.random.seed(99)
    close = np.linspace(1.08, 1.10, n) + np.random.normal(0, 0.0003, n)
    return pd.DataFrame({
        "open": close + np.random.normal(0, 0.0001, n),
        "high": close + np.abs(np.random.normal(0, 0.0005, n)),
        "low": close - np.abs(np.random.normal(0, 0.0005, n)),
        "close": close,
        "volume": np.random.randint(100, 500, n).astype(float),
    }, index=pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"))


# ── Risk Manager Tests ─────────────────────────────────────────────────────────

class TestRiskManager:

    def test_normal_proposal_approved(self):
        from forexmind.risk.manager import RiskManager
        rm = RiskManager()
        proposal = rm.calculate_risk(
            instrument="EUR_USD",
            direction="BUY",
            entry=1.0850,
            atr=0.0010,
            account_balance=10000.0,
        )
        # Should be approved: entry/SL/TP are all set, R:R >= 1
        assert proposal.stop_loss < 1.0850
        assert proposal.take_profit > 1.0850
        assert proposal.units > 0
        assert proposal.risk_usd > 0

    def test_stop_loss_on_correct_side_for_buy(self):
        from forexmind.risk.manager import RiskManager
        rm = RiskManager()
        proposal = rm.calculate_risk("EUR_USD", "BUY", 1.0850, 0.0010, 10000.0)
        assert proposal.stop_loss < proposal.entry_price

    def test_stop_loss_on_correct_side_for_sell(self):
        from forexmind.risk.manager import RiskManager
        rm = RiskManager()
        proposal = rm.calculate_risk("EUR_USD", "SELL", 1.0850, 0.0010, 10000.0)
        assert proposal.stop_loss > proposal.entry_price

    def test_ai_risk_pct_override_is_capped(self):
        from forexmind.risk.manager import RiskManager
        rm = RiskManager()
        # Even if AI asks for 10%, max is 2%
        proposal = rm.calculate_risk(
            "EUR_USD", "BUY", 1.0850, 0.0010, 10000.0, ai_risk_pct=10.0
        )
        assert proposal.risk_pct <= 2.0

    def test_max_concurrent_trades_kill_switch(self):
        from forexmind.risk.manager import RiskManager, RiskConfig, OpenTrade
        cfg = RiskConfig(max_concurrent_trades=1)
        rm = RiskManager(config=cfg)
        # Manually fill the slot
        rm._open_trades["fake_id"] = OpenTrade(
            trade_id="fake_id", instrument="EUR_USD",
            direction="BUY", entry_price=1.085,
            stop_loss=1.083, take_profit=1.089, units=1000,
        )
        proposal = rm.calculate_risk("GBP_USD", "BUY", 1.2700, 0.0012, 10000.0)
        assert not proposal.approved
        assert "concurrent" in proposal.rejection_reason.lower()

    def test_risk_proposal_summary(self):
        from forexmind.risk.manager import RiskManager
        rm = RiskManager()
        proposal = rm.calculate_risk("EUR_USD", "BUY", 1.0850, 0.0010, 10000.0)
        summary = proposal.summary()
        assert "EUR_USD" in summary
        assert "BUY" in summary


# ── Rule-Based Strategy Tests ─────────────────────────────────────────────────

class TestRuleBasedStrategy:

    def test_returns_strategy_signal(self, sample_df):
        from forexmind.indicators.engine import get_indicator_engine
        from forexmind.strategy.rule_based import RuleBasedStrategy

        engine = get_indicator_engine()
        df_ind = engine.compute(sample_df)
        strategy = RuleBasedStrategy()
        sig = strategy.generate_signal(df_ind, "EUR_USD", "M5", float(sample_df["close"].iloc[-1]))

        assert sig.direction in ("BUY", "SELL", "HOLD")
        assert 0.0 <= sig.confidence <= 1.0
        assert sig.instrument == "EUR_USD"
        assert sig.source == "rule_based"

    def test_hold_signal_on_short_data(self):
        from forexmind.strategy.rule_based import RuleBasedStrategy
        short_df = pd.DataFrame({
            "open": [1.08, 1.09], "high": [1.085, 1.095],
            "low": [1.075, 1.085], "close": [1.082, 1.092], "volume": [100, 100],
        }, index=pd.date_range("2024-01-01", periods=2, freq="5min", tz="UTC"))

        strategy = RuleBasedStrategy()
        sig = strategy.generate_signal(short_df, "EUR_USD", "M5", 1.092)
        assert sig.direction == "HOLD"

    def test_is_actionable_requires_confidence(self):
        from forexmind.strategy.base import StrategySignal
        sig = StrategySignal(
            instrument="EUR_USD", timeframe="M5",
            direction="BUY", confidence=0.30,  # below 0.45 threshold
            entry_price=1.085, stop_loss=1.083, take_profit=1.089,
        )
        assert not sig.is_actionable  # Confidence too low

    def test_is_actionable_requires_rr(self):
        from forexmind.strategy.base import StrategySignal
        sig = StrategySignal(
            instrument="EUR_USD", timeframe="M5",
            direction="BUY", confidence=0.70,
            entry_price=1.085,
            stop_loss=1.084,   # 1 pip SL
            take_profit=1.0855, # 0.5 pip TP → R:R = 0.5 (bad)
        )
        assert not sig.is_actionable  # R:R too low


# ── Backtester Tests ──────────────────────────────────────────────────────────

class TestBacktester:

    def test_backtest_runs_without_error(self, sample_df):
        from forexmind.backtest.engine import Backtester
        from forexmind.strategy.rule_based import RuleBasedStrategy

        bt = Backtester()
        strategy = RuleBasedStrategy()
        result = bt.run(sample_df, strategy, "EUR_USD", "M5", warmup_bars=60)
        assert result is not None
        assert result.total_trades >= 0

    def test_backtest_result_metrics_valid(self, sample_df):
        from forexmind.backtest.engine import Backtester
        from forexmind.strategy.rule_based import RuleBasedStrategy

        bt = Backtester()
        strategy = RuleBasedStrategy()
        result = bt.run(sample_df, strategy, "EUR_USD", "M5", warmup_bars=60)

        if result.total_trades > 0:
            assert 0.0 <= result.win_rate <= 1.0
            assert result.profit_factor >= 0

    def test_monte_carlo_with_no_trades(self, sample_df):
        from forexmind.backtest.engine import Backtester, BacktestResult, BacktestConfig
        bt = Backtester()
        result = BacktestResult(
            instrument="EUR_USD", timeframe="M5",
            start_date="2024-01-01", end_date="2024-12-31",
            trades=[], config=BacktestConfig(),
        )
        mc = bt.monte_carlo(result)
        assert mc == {}   # No trades → no simulation
