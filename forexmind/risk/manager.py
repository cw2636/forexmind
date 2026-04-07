"""
ForexMind — Risk Management Engine
=====================================
Handles position sizing, stop-loss enforcement, daily P&L limits,
and real-time trade monitoring.

Key capabilities:
  1. Dynamic position sizing (Kelly Criterion + ATR-based risk)
  2. AI-overrideable risk parameters (Claude can tighten/loosen per trade)
  3. Daily loss kill-switch (stop trading if drawdown hits threshold)
  4. Trailing stop management
  5. Break-even stop escalation

Advanced Python:
  - dataclasses with validation via __post_init__
  - Context manager for trade lifecycle tracking
  - @property computed fields
  - asyncio.Lock for thread-safe state mutation
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from forexmind.config.settings import get_settings, RiskConfig
from forexmind.utils.helpers import (
    atr_stop_loss,
    atr_take_profit,
    kelly_fraction,
    price_to_pips,
    units_from_risk,
)
from forexmind.utils.logger import get_logger

log = get_logger(__name__)


# ── Trade risk proposal (output of risk manager) ─────────────────────────────

@dataclass
class RiskProposal:
    """
    Complete risk package for a single trade.
    Returned by RiskManager.calculate_risk() and sent to Claude
    for optional AI override before execution.
    """
    instrument: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    units: int
    risk_pct: float              # % of account being risked
    risk_usd: float              # Dollar amount at risk
    reward_usd: float            # Dollar reward if TP hit
    risk_reward_ratio: float     # reward / risk
    stop_loss_pips: float
    take_profit_pips: float
    atr: float
    kelly_fraction: float
    approved: bool = True        # Can be set False by kill-switch
    rejection_reason: str = ""

    def __post_init__(self) -> None:
        # Only apply the R:R sanity-check if no prior rejection reason exists.
        # (kill-switch and concurrent-trade rejections set rejection_reason before
        # stop/TP are computed, so risk_reward_ratio will be 0.0 — we must not
        # override those already-meaningful rejection messages.)
        if self.risk_reward_ratio < 1.0 and not self.rejection_reason:
            self.approved = False
            self.rejection_reason = f"R:R ratio {self.risk_reward_ratio:.2f} < 1.0 — not worth the risk"

    def summary(self) -> str:
        return (
            f"{'✅' if self.approved else '❌'} {self.direction} {self.instrument} "
            f"| Entry: {self.entry_price:.5f} "
            f"| SL: {self.stop_loss:.5f} ({self.stop_loss_pips:.1f} pips) "
            f"| TP: {self.take_profit:.5f} ({self.take_profit_pips:.1f} pips) "
            f"| Units: {self.units:,} "
            f"| Risk: {self.risk_pct:.2f}% (${self.risk_usd:.2f}) "
            f"| R:R {self.risk_reward_ratio:.1f}:1"
        )


@dataclass
class OpenTrade:
    """Tracks a live trade for trailing stop management."""
    trade_id: str
    instrument: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    units: int
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    highest_price: float = 0.0   # Track for trailing stop
    lowest_price: float = float("inf")
    breakeven_reached: bool = False


class RiskManager:
    """
    Central risk manager — the gatekeeper before any trade is placed.

    Usage:
        rm = RiskManager()
        proposal = rm.calculate_risk(
            instrument="EUR_USD", direction="BUY",
            entry=1.08500, atr=0.00080, account_balance=10000.0
        )
        if proposal.approved:
            # Place trade
    """

    def __init__(self, config: RiskConfig | None = None) -> None:
        self._cfg = config or get_settings().risk
        self._open_trades: dict[str, OpenTrade] = {}
        self._daily_pnl_usd: float = 0.0
        self._daily_pnl_date: str = ""
        self._lock = asyncio.Lock()
        # Live win-rate tracking — feeds real data into Kelly sizing
        self._wins: int = 0
        self._losses: int = 0

    def calculate_risk(
        self,
        instrument: str,
        direction: str,
        entry: float,
        atr: float,
        account_balance: float,
        ai_risk_pct: float | None = None,      # Optional override from Claude
        atr_multiplier: float | None = None,    # Optional override
        rr_ratio: float | None = None,          # Optional override
        win_rate_estimate: float = 0.55,        # Used for Kelly Criterion
    ) -> RiskProposal:
        """
        Calculate the complete risk proposal for a trade.

        Steps:
          1. Check kill-switch (daily loss limit)
          2. Check concurrent trade limit
          3. Compute stop-loss and take-profit using ATR
          4. Compute position size using risk %
          5. Apply Kelly criterion as upper bound
          6. Return RiskProposal for approval
        """
        # ── Kill-switch checks ─────────────────────────────────────────────────
        today = datetime.now(timezone.utc).date().isoformat()
        if today != self._daily_pnl_date:
            self._daily_pnl_usd = 0.0
            self._daily_pnl_date = today

        max_daily_loss = account_balance * self._cfg.max_daily_loss_pct / 100.0
        if self._daily_pnl_usd < -max_daily_loss:
            return _rejected(
                instrument, direction, entry, atr,
                f"Daily loss limit hit (${self._daily_pnl_usd:.2f} loss, limit ${max_daily_loss:.2f})"
            )

        if len(self._open_trades) >= self._cfg.max_concurrent_trades:
            return _rejected(
                instrument, direction, entry, atr,
                f"Max concurrent trades ({self._cfg.max_concurrent_trades}) already open"
            )

        # ── Stop-Loss & Take-Profit ────────────────────────────────────────────
        mult = atr_multiplier or self._cfg.atr_stop_multiplier
        rr = rr_ratio or self._cfg.default_rr_ratio

        stop_loss = atr_stop_loss(entry, atr, direction, multiplier=mult)
        take_profit = atr_take_profit(entry, stop_loss, direction, rr_ratio=rr)

        stop_pips = price_to_pips(abs(entry - stop_loss), instrument)
        tp_pips = price_to_pips(abs(take_profit - entry), instrument)

        # ── Position Sizing ────────────────────────────────────────────────────
        # Use measured win rate if we have enough trades; fall back to estimate
        measured_wr = self.measured_win_rate
        effective_win_rate = measured_wr if measured_wr is not None else win_rate_estimate
        kelly = kelly_fraction(effective_win_rate, rr)

        # Risk %: use AI override if provided, else Kelly-informed default
        if ai_risk_pct is not None:
            risk_pct = max(self._cfg.min_risk_per_trade_pct,
                           min(ai_risk_pct, self._cfg.max_risk_per_trade_pct))
        else:
            # Kelly suggests how much to risk; cap at our max
            kelly_pct = kelly * 100.0  # Convert fraction to percentage
            risk_pct = min(
                max(kelly_pct, self._cfg.min_risk_per_trade_pct),
                self._cfg.max_risk_per_trade_pct
            )

        risk_usd = account_balance * risk_pct / 100.0
        reward_usd = risk_usd * rr

        units = units_from_risk(
            account_balance=account_balance,
            risk_pct=risk_pct,
            stop_loss_pips=stop_pips,
            instrument=instrument,
            current_price=entry,
        )

        rr_actual = tp_pips / stop_pips if stop_pips > 0 else 0.0

        log.info(
            f"Risk proposal: {direction} {instrument} "
            f"risk={risk_pct:.2f}% kelly={kelly:.3f} "
            f"sl={stop_pips:.1f}pips tp={tp_pips:.1f}pips R:R={rr_actual:.1f}"
        )

        return RiskProposal(
            instrument=instrument,
            direction=direction,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            units=units,
            risk_pct=round(risk_pct, 3),
            risk_usd=round(risk_usd, 2),
            reward_usd=round(reward_usd, 2),
            risk_reward_ratio=round(rr_actual, 2),
            stop_loss_pips=round(stop_pips, 1),
            take_profit_pips=round(tp_pips, 1),
            atr=atr,
            kelly_fraction=round(kelly, 4),
        )

    async def register_trade(self, trade: OpenTrade) -> None:
        """Register an opened trade for tracking."""
        async with self._lock:
            self._open_trades[trade.trade_id] = trade
            log.info(f"Trade registered: {trade.trade_id} {trade.direction} {trade.instrument}")

    async def update_trailing_stop(
        self,
        trade_id: str,
        current_price: float,
        atr: float,
    ) -> float | None:
        """
        Check if the trailing stop should be moved.
        Returns the new stop-loss price if it should be updated, else None.
        """
        async with self._lock:
            trade = self._open_trades.get(trade_id)
            if trade is None:
                return None

            mult = self._cfg.trailing_stop_multiplier
            new_stop: float | None = None

            if trade.direction == "BUY":
                trade.highest_price = max(trade.highest_price, current_price)
                trailing_stop = trade.highest_price - (atr * mult)
                if trailing_stop > trade.stop_loss:
                    trade.stop_loss = trailing_stop
                    new_stop = trailing_stop
                    log.debug(f"Trailing stop moved UP to {trailing_stop:.5f} for {trade_id}")

            elif trade.direction == "SELL":
                trade.lowest_price = min(trade.lowest_price, current_price)
                trailing_stop = trade.lowest_price + (atr * mult)
                if trailing_stop < trade.stop_loss:
                    trade.stop_loss = trailing_stop
                    new_stop = trailing_stop
                    log.debug(f"Trailing stop moved DOWN to {trailing_stop:.5f} for {trade_id}")

            # Break-even activation
            if not trade.breakeven_reached:
                risk = abs(trade.entry_price - trade.stop_loss)
                if trade.direction == "BUY":
                    profit_pips = current_price - trade.entry_price
                    if profit_pips >= risk * self._cfg.breakeven_trigger_rr:
                        trade.stop_loss = trade.entry_price
                        trade.breakeven_reached = True
                        new_stop = trade.entry_price
                        log.info(f"Break-even activated for {trade_id}")
                elif trade.direction == "SELL":
                    profit_pips = trade.entry_price - current_price
                    if profit_pips >= risk * self._cfg.breakeven_trigger_rr:
                        trade.stop_loss = trade.entry_price
                        trade.breakeven_reached = True
                        new_stop = trade.entry_price

            return new_stop

    async def close_trade(self, trade_id: str, exit_price: float) -> float:
        """
        Deregister a closed trade and update daily P&L and win rate tracker.
        Returns P&L in pips.
        """
        async with self._lock:
            trade = self._open_trades.pop(trade_id, None)
            if trade is None:
                return 0.0
            ps = pip_size(trade.instrument)
            if trade.direction == "BUY":
                pnl_pips = (exit_price - trade.entry_price) / ps
            else:
                pnl_pips = (trade.entry_price - exit_price) / ps
            # Approximate USD P&L (assumes 1 pip ≈ $0.0001 per unit for major pairs)
            self._daily_pnl_usd += pnl_pips * trade.units * ps
            # Update live win rate counter
            if pnl_pips > 0:
                self._wins += 1
            else:
                self._losses += 1
            log.info(
                f"Trade closed: {trade_id} P&L = {pnl_pips:.1f} pips | "
                f"Live W/L: {self._wins}/{self._losses} "
                f"({self.measured_win_rate:.1%} WR)" if self.measured_win_rate else
                f"Trade closed: {trade_id} P&L = {pnl_pips:.1f} pips"
            )
            return pnl_pips

    @property
    def open_trade_count(self) -> int:
        return len(self._open_trades)

    @property
    def daily_pnl_usd(self) -> float:
        return self._daily_pnl_usd

    @property
    def is_kill_switch_active(self) -> bool:
        return self._daily_pnl_usd < 0 and abs(self._daily_pnl_usd) >= (
            10000 * self._cfg.max_daily_loss_pct / 100.0  # Rough check
        )

    @property
    def measured_win_rate(self) -> float | None:
        """
        Returns the live measured win rate once we have >= 30 closed trades.
        Returns None before that threshold — too few trades to be reliable.
        """
        total = self._wins + self._losses
        if total < 30:
            return None
        return self._wins / total

    @property
    def trade_stats(self) -> dict:
        """Summary of live trading statistics."""
        total = self._wins + self._losses
        return {
            "wins": self._wins,
            "losses": self._losses,
            "total_closed": total,
            "win_rate": round(self._wins / total, 4) if total > 0 else None,
            "using_measured_wr": self.measured_win_rate is not None,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rejected(
    instrument: str, direction: str, entry: float, atr: float, reason: str
) -> RiskProposal:
    """Return a rejected RiskProposal with zero sizing."""
    return RiskProposal(
        instrument=instrument, direction=direction,
        entry_price=entry, stop_loss=0.0, take_profit=0.0,
        units=0, risk_pct=0.0, risk_usd=0.0, reward_usd=0.0,
        risk_reward_ratio=0.0, stop_loss_pips=0.0, take_profit_pips=0.0,
        atr=atr, kelly_fraction=0.0, approved=False, rejection_reason=reason,
    )


# ── Singleton ─────────────────────────────────────────────────────────────────

_risk_manager: RiskManager | None = None


def get_risk_manager() -> RiskManager:
    global _risk_manager
    if _risk_manager is None:
        _risk_manager = RiskManager()
    return _risk_manager
