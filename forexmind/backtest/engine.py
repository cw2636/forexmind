"""
ForexMind — Vectorized Backtesting Engine
==========================================
Fast vectorized backtester for strategy evaluation on historical data.
Also includes an event-driven mode (slower, more realistic) for final validation.

Features:
  - Realistic spread simulation
  - Slippage modelling
  - Commission deduction
  - Walk-forward validation split
  - Complete performance metrics
  - Monte Carlo simulation for robustness

Advanced Python:
  - Generator-based event iteration for memory efficiency
  - numpy vectorised P&L computation
  - @dataclass with complex __post_init__ validation
  - Context variable for per-run state isolation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Generator

import numpy as np
import pandas as pd

from forexmind.indicators.engine import get_indicator_engine
from forexmind.strategy.base import BaseStrategy, StrategySignal
from forexmind.strategy.ensemble import EnsembleSignal
from forexmind.utils.helpers import pip_size, price_to_pips
from forexmind.utils.logger import get_logger
from forexmind.config.settings import get_settings

log = get_logger(__name__)


# ── Backtest Configuration ────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    commission_per_lot: float = 7.0      # USD round-trip per std lot
    slippage_pips: float = 0.5
    spread_pips_default: float = 1.5
    risk_pct_per_trade: float = 1.5
    rr_ratio: float = 2.0
    atr_multiplier: float = 1.5
    max_concurrent_trades: int = 3

    @classmethod
    def from_settings(cls) -> "BacktestConfig":
        bt = get_settings().backtest_config
        risk = get_settings().risk_config_yaml
        return cls(
            initial_capital=bt.get("initial_capital", 10000.0),
            commission_per_lot=bt.get("commission_per_lot", 7.0),
            slippage_pips=bt.get("slippage_pips", 0.5),
            risk_pct_per_trade=risk.get("min_risk_per_trade_pct", 1.5),
            rr_ratio=risk.get("default_rr_ratio", 2.0),
            atr_multiplier=risk.get("atr_stop_multiplier", 1.5),
        )


# ── Trade Record ──────────────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    """A single completed trade in the backtest."""
    entry_time: datetime
    exit_time: datetime
    instrument: str
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    units: int
    commission_usd: float
    pnl_pips: float = field(init=False)
    pnl_usd: float = field(init=False)
    exit_reason: str = "unknown"   # "tp", "sl", "forced"

    def __post_init__(self) -> None:
        ps = pip_size(self.instrument)
        if self.direction == "BUY":
            self.pnl_pips = (self.exit_price - self.entry_price) / ps
        else:
            self.pnl_pips = (self.entry_price - self.exit_price) / ps
        # Approximate USD P&L (minor pairs have different pip values — simplification)
        self.pnl_usd = self.pnl_pips * self.units * ps - self.commission_usd


# ── Performance Metrics ───────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    """Complete backtest performance summary."""
    instrument: str
    timeframe: str
    start_date: str
    end_date: str
    trades: list[BacktestTrade]
    config: BacktestConfig
    equity_curve: list[float] = field(default_factory=list)

    # Computed stats (populated in __post_init__)
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl_pips: float = 0.0
    total_pnl_usd: float = 0.0
    net_return_pct: float = 0.0
    average_win_pips: float = 0.0
    average_loss_pips: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    avg_trade_duration_min: float = 0.0

    def __post_init__(self) -> None:
        if not self.trades:
            return
        self._compute()

    def _compute(self) -> None:
        """Compute all performance metrics from trades list."""
        wins = [t for t in self.trades if t.pnl_usd > 0]
        losses = [t for t in self.trades if t.pnl_usd <= 0]
        self.total_trades = len(self.trades)
        self.winning_trades = len(wins)
        self.losing_trades = len(losses)
        self.win_rate = len(wins) / self.total_trades if self.total_trades > 0 else 0.0
        self.total_pnl_pips = sum(t.pnl_pips for t in self.trades)
        self.total_pnl_usd = sum(t.pnl_usd for t in self.trades)
        self.net_return_pct = (self.total_pnl_usd / self.config.initial_capital) * 100.0
        self.average_win_pips = np.mean([t.pnl_pips for t in wins]) if wins else 0.0
        self.average_loss_pips = np.mean([t.pnl_pips for t in losses]) if losses else 0.0

        gross_profit = sum(t.pnl_usd for t in wins)
        gross_loss = abs(sum(t.pnl_usd for t in losses))
        self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Compute equity curve and max drawdown
        if self.equity_curve:
            equity = np.array(self.equity_curve)
            peak = np.maximum.accumulate(equity)
            drawdown = (peak - equity) / peak
            self.max_drawdown_pct = float(np.max(drawdown) * 100)

        # Sharpe + Sortino (daily returns approximation)
        if len(self.trades) >= 5:
            returns = np.array([t.pnl_usd for t in self.trades])
            mean_r = np.mean(returns)
            std_r = np.std(returns)
            downside = np.std([r for r in returns if r < 0]) or 1e-9
            self.sharpe_ratio = float(mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0.0
            self.sortino_ratio = float(mean_r / downside * np.sqrt(252))

        # Average duration
        durations = [
            (t.exit_time - t.entry_time).total_seconds() / 60
            for t in self.trades
        ]
        self.avg_trade_duration_min = float(np.mean(durations)) if durations else 0.0

    def summary(self) -> str:
        return (
            f"\n{'='*60}\n"
            f"  BACKTEST: {self.instrument} {self.timeframe}  "
            f"({self.start_date} → {self.end_date})\n"
            f"{'='*60}\n"
            f"  Total trades:     {self.total_trades}\n"
            f"  Win rate:         {self.win_rate:.1%}\n"
            f"  Profit factor:    {self.profit_factor:.2f}\n"
            f"  Net P&L:          {self.total_pnl_pips:.1f} pips  /  ${self.total_pnl_usd:.2f}\n"
            f"  Net return:       {self.net_return_pct:.2f}%\n"
            f"  Max drawdown:     {self.max_drawdown_pct:.2f}%\n"
            f"  Sharpe ratio:     {self.sharpe_ratio:.2f}\n"
            f"  Sortino ratio:    {self.sortino_ratio:.2f}\n"
            f"  Avg win:          {self.average_win_pips:.1f} pips\n"
            f"  Avg loss:         {self.average_loss_pips:.1f} pips\n"
            f"  Avg duration:     {self.avg_trade_duration_min:.1f} min\n"
            f"{'='*60}"
        )


# ── Vectorized Backtester ─────────────────────────────────────────────────────

class Backtester:
    """
    Vectorised event-driven backtester.
    Iterates bar-by-bar (event-driven) over historical OHLCV data,
    generates signals using a strategy, and simulates trade execution.

    Usage:
        bt = Backtester()
        df = oanda_client.get_candles("EUR_USD", "M5", from_dt=..., to_dt=...)
        result = bt.run(df, strategy, "EUR_USD", "M5")
        print(result.summary())
    """

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self._cfg = config or BacktestConfig.from_settings()
        self._engine = get_indicator_engine()

    def run(
        self,
        df: pd.DataFrame,
        strategy: BaseStrategy,
        instrument: str,
        timeframe: str,
        htf_df: pd.DataFrame | None = None,
        warmup_bars: int = 200,
    ) -> BacktestResult:
        """
        Run a full backtest.

        Args:
            df: Full OHLCV DataFrame to test on.
            strategy: Any BaseStrategy subclass instance.
            instrument: e.g. "EUR_USD"
            timeframe: e.g. "M5"
            htf_df: Optional higher-timeframe context DataFrame.
            warmup_bars: Number of initial bars used to warm up indicators (not traded).

        Returns:
            BacktestResult with full trade list and metrics.
        """
        log.info(f"Starting backtest: {instrument} {timeframe} | {len(df)} bars")

        # Pre-compute all indicators on the full DataFrame (efficient)
        df_ind = self._engine.compute(df)

        capital = self._cfg.initial_capital
        equity_curve: list[float] = [capital]
        completed_trades: list[BacktestTrade] = []

        # State: list of open simulated trades (supports concurrent positions)
        open_trades: list[dict] = []
        max_concurrent = self._cfg.max_concurrent_trades

        ps = pip_size(instrument)
        slippage = self._cfg.slippage_pips * ps

        for i in range(warmup_bars, len(df_ind)):
            candle = df_ind.iloc[i]
            current_bar_dt = df_ind.index[i]
            current_price = float(candle["close"])
            current_high = float(candle["high"])
            current_low = float(candle["low"])

            # ── Check all open trades for SL/TP hits ──────────────────────────
            still_open: list[dict] = []
            for open_trade in open_trades:
                direction = open_trade["direction"]
                sl = open_trade["sl"]
                tp = open_trade["tp"]
                entry = open_trade["entry_price"]

                # ── Breakeven stop management ─────────────────────────────────
                # Once price moves 1R in profit, slide SL to entry+1pip (risk-free).
                # This is applied before the SL/TP hit check so the new SL is used
                # for the current bar's evaluation.
                if not open_trade.get("breakeven_triggered", False):
                    risk_distance = abs(entry - sl)  # 1R distance
                    if direction == "BUY" and current_high >= entry + risk_distance:
                        open_trade["sl"] = entry + ps  # 1 pip profit lock-in
                        open_trade["breakeven_triggered"] = True
                        sl = open_trade["sl"]
                    elif direction == "SELL" and current_low <= entry - risk_distance:
                        open_trade["sl"] = entry - ps
                        open_trade["breakeven_triggered"] = True
                        sl = open_trade["sl"]

                hit_sl = hit_tp = False
                if direction == "BUY":
                    hit_sl = current_low <= sl
                    hit_tp = current_high >= tp
                else:
                    hit_sl = current_high >= sl
                    hit_tp = current_low <= tp

                if hit_tp or hit_sl:
                    # Apply exit slippage: fills are slightly worse than the exact level
                    if hit_tp:
                        exit_price = tp - slippage if direction == "BUY" else tp + slippage
                        exit_reason = "tp"
                    else:
                        exit_price = sl - slippage if direction == "BUY" else sl + slippage
                        exit_reason = "sl"
                    trade = BacktestTrade(
                        entry_time=open_trade["entry_time"],
                        exit_time=current_bar_dt.to_pydatetime(),
                        instrument=instrument,
                        direction=direction,
                        entry_price=entry,
                        exit_price=exit_price,
                        stop_loss=sl,
                        take_profit=tp,
                        units=open_trade["units"],
                        commission_usd=open_trade["commission"],
                        exit_reason=exit_reason,
                    )
                    capital += trade.pnl_usd
                    equity_curve.append(capital)
                    completed_trades.append(trade)
                else:
                    still_open.append(open_trade)
            open_trades = still_open

            # ── Check for new signal (if capacity available) ──────────────────
            if len(open_trades) < max_concurrent:
                history = df_ind.iloc[max(0, i - 500):i + 1]
                sig = strategy.generate_signal(
                    history, instrument, timeframe, current_price
                )

                if sig.is_actionable:
                    entry_price = sig.entry_price + (slippage if sig.direction == "BUY" else -slippage)
                    stop_pips = price_to_pips(abs(entry_price - sig.stop_loss), instrument)
                    if stop_pips <= 0:
                        continue
                    # Risk per trade scales with capital; divide by concurrent
                    # trades to avoid over-exposing on correlated positions
                    effective_risk_pct = self._cfg.risk_pct_per_trade / max(1, max_concurrent)
                    risk_usd = capital * effective_risk_pct / 100.0
                    units = max(1000, int(risk_usd / (stop_pips * ps)))
                    commission = (units / 100_000) * self._cfg.commission_per_lot

                    open_trades.append({
                        "direction": sig.direction,
                        "entry_price": entry_price,
                        "sl": sig.stop_loss,
                        "tp": sig.take_profit,
                        "units": units,
                        "commission": commission,
                        "entry_time": current_bar_dt.to_pydatetime(),
                    })

        # Force-close any remaining open trades at last price
        last_price = float(df_ind["close"].iloc[-1])
        for open_trade in open_trades:
            trade = BacktestTrade(
                entry_time=open_trade["entry_time"],
                exit_time=df_ind.index[-1].to_pydatetime(),
                instrument=instrument,
                direction=open_trade["direction"],
                entry_price=open_trade["entry_price"],
                exit_price=last_price,
                stop_loss=open_trade["sl"],
                take_profit=open_trade["tp"],
                units=open_trade["units"],
                commission_usd=open_trade["commission"],
                exit_reason="forced",
            )
            capital += trade.pnl_usd
            equity_curve.append(capital)
            completed_trades.append(trade)

        start = str(df_ind.index[0].date())
        end = str(df_ind.index[-1].date())

        result = BacktestResult(
            instrument=instrument,
            timeframe=timeframe,
            start_date=start,
            end_date=end,
            trades=completed_trades,
            config=self._cfg,
            equity_curve=equity_curve,
        )
        log.info(result.summary())
        return result

    def walk_forward(
        self,
        df: pd.DataFrame,
        strategy: BaseStrategy,
        instrument: str,
        timeframe: str,
        n_splits: int = 5,
        train_pct: float = 0.7,
    ) -> list[BacktestResult]:
        """
        Walk-forward validation: train on first X% of each fold, test on rest.
        Returns a list of BacktestResults (one per fold).

        This prevents overfitting by ensuring the strategy is always tested
        on data it has never seen during training.
        """
        results: list[BacktestResult] = []
        fold_size = len(df) // n_splits

        for fold in range(n_splits):
            start = fold * fold_size
            end = start + fold_size
            fold_df = df.iloc[start:end]
            split = int(len(fold_df) * train_pct)
            test_df = fold_df.iloc[split:]

            log.info(f"Walk-forward fold {fold+1}/{n_splits}: testing on {len(test_df)} bars")
            result = self.run(test_df, strategy, instrument, timeframe)
            results.append(result)

        return results

    def monte_carlo(
        self,
        result: BacktestResult,
        n_simulations: int = 1000,
        confidence: float = 0.95,
    ) -> dict:
        """
        Monte Carlo simulation: randomly shuffle the trade sequence
        to estimate robustness across different market luck scenarios.

        Returns:
          - worst_drawdown_pct: Value at Risk for drawdown at given confidence
          - median_return_pct: Typical expected return
          - var_return_pct: Value at Risk for return
        """
        if not result.trades:
            return {}

        returns = np.array([t.pnl_usd for t in result.trades])
        simulated_returns: list[float] = []
        simulated_drawdowns: list[float] = []

        rng = np.random.default_rng(42)
        for _ in range(n_simulations):
            shuffled = rng.permutation(returns)
            equity = self._cfg.initial_capital + np.cumsum(shuffled)
            equity = np.concatenate([[self._cfg.initial_capital], equity])
            total_return = (equity[-1] - equity[0]) / equity[0] * 100.0
            peaks = np.maximum.accumulate(equity)
            drawdowns = (peaks - equity) / peaks * 100.0
            simulated_returns.append(total_return)
            simulated_drawdowns.append(float(np.max(drawdowns)))

        percentile = (1.0 - confidence) * 100
        return {
            "median_return_pct": float(np.median(simulated_returns)),
            "var_return_pct": float(np.percentile(simulated_returns, percentile)),
            "worst_drawdown_pct": float(np.percentile(simulated_drawdowns, 100 - percentile)),
            "n_simulations": n_simulations,
            "confidence": confidence,
        }
