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
    confidence_scaled_risk,
    kelly_fraction,
    pip_size,
    price_to_pips,
    units_from_risk,
)
from forexmind.utils.session_times import get_tp_session_multiplier
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
        # Peak equity tracking for total drawdown circuit breaker
        self._peak_balance: float = 0.0     # highest account balance seen
        self._initial_balance: float = 0.0  # set on first calculate_risk call

    def calculate_risk(
        self,
        instrument: str,
        direction: str,
        entry: float,
        atr: float,
        account_balance: float,
        confidence: float | None = None,       # Signal confidence (0–1) — drives tiered sizing
        ai_risk_pct: float | None = None,      # Optional override from Claude
        atr_multiplier: float | None = None,   # Optional override
        rr_ratio: float | None = None,         # Optional override
        win_rate_estimate: float = 0.55,       # Used for Kelly Criterion
        skip_correlation: bool = False,        # Force bypass correlation block
    ) -> RiskProposal:
        """
        Calculate the complete risk proposal for a trade.

        Steps:
          1. Update peak equity; check total drawdown circuit breaker
          2. Check daily loss limit
          3. Confidence gate — reject if signal confidence is below minimum
          4. Check concurrent trade limit
          5. Compute stop-loss and take-profit using ATR
          6. Compute position size using confidence-tiered risk %
          7. Return RiskProposal for approval
        """
        # ── Peak equity tracking ───────────────────────────────────────────────
        if self._initial_balance == 0.0:
            self._initial_balance = account_balance
        if account_balance > self._peak_balance:
            self._peak_balance = account_balance
            log.info(f"New peak equity: ${self._peak_balance:.2f}")

        # ── Total drawdown circuit breaker ─────────────────────────────────────
        if self._peak_balance > 0:
            drawdown_pct = (self._peak_balance - account_balance) / self._peak_balance * 100.0
            if drawdown_pct >= self._cfg.max_total_drawdown_pct:
                return _rejected(
                    instrument, direction, entry, atr,
                    f"🚨 TOTAL DRAWDOWN HALT: account is {drawdown_pct:.1f}% below peak "
                    f"(${self._peak_balance:.2f} → ${account_balance:.2f}). "
                    f"Review system before resuming trading."
                )

        # ── Daily loss limit ───────────────────────────────────────────────────
        today = datetime.now(timezone.utc).date().isoformat()
        if today != self._daily_pnl_date:
            self._daily_pnl_usd = 0.0
            self._daily_pnl_date = today

        daily_loss_limit = account_balance * self._cfg.daily_loss_limit_pct / 100.0
        if self._daily_pnl_usd <= -daily_loss_limit:
            return _rejected(
                instrument, direction, entry, atr,
                f"🛑 Daily loss limit hit: ${self._daily_pnl_usd:.2f} loss "
                f"(limit ${daily_loss_limit:.2f}). No new trades today."
            )

        # Daily profit target — set very high so it only fires on extraordinary days
        daily_profit_target = account_balance * self._cfg.daily_profit_target_pct / 100.0
        if self._daily_pnl_usd >= daily_profit_target:
            return _rejected(
                instrument, direction, entry, atr,
                f"🎯 Daily profit target reached: +${self._daily_pnl_usd:.2f} "
                f"(target ${daily_profit_target:.2f}). Trading locked for today."
            )

        # ── Confidence gate ────────────────────────────────────────────────────
        # Reject trades with insufficient signal confidence before sizing
        if confidence is not None and confidence < self._cfg.min_signal_confidence:
            return _rejected(
                instrument, direction, entry, atr,
                f"⚠️ Signal confidence {confidence:.0%} below minimum "
                f"{self._cfg.min_signal_confidence:.0%} — skipping weak signal."
            )

        if len(self._open_trades) >= self._cfg.max_concurrent_trades:
            return _rejected(
                instrument, direction, entry, atr,
                f"Max concurrent trades ({self._cfg.max_concurrent_trades}) already open"
            )

        # ── Correlated pair protection ─────────────────────────────────────────
        # Trading two USD-base pairs in the same direction = double USD exposure.
        # e.g. SELL USD/JPY + SELL USD/CAD = both profit only if USD weakens.
        # One news event wipes both simultaneously.
        if not skip_correlation:
            _CORR_GROUPS: list[tuple[str, ...]] = [
                ("USD_JPY", "USD_CAD", "USD_CHF"),   # USD as base — all correlate on USD strength
                ("EUR_USD", "GBP_USD", "AUD_USD"),   # USD as quote — all correlate on USD weakness
            ]
            base, quote = (instrument.split("_") + [""])[:2]
            for group in _CORR_GROUPS:
                if instrument in group:
                    # Check if we already have a trade in the same correlation group
                    for _tid, open_trade in self._open_trades.items():
                        open_inst = open_trade.instrument
                        if open_inst in group and open_inst != instrument:
                            # Same group — check if directions imply the same USD bet
                            existing_dir = open_trade.direction
                            same_bet = (direction == existing_dir)
                            if same_bet:
                                log.warning(
                                    f"Correlated pair block: {instrument} {direction} correlates with "
                                    f"existing {open_inst} {existing_dir}"
                                )
                                # Both groups are USD plays — determine which USD direction
                                # Group (EUR/GBP/AUD)_USD: SELL = USD long, BUY = USD short
                                # Group USD_(JPY/CAD/CHF): BUY  = USD long, SELL = USD short
                                usd_quote_group = ("EUR_USD", "GBP_USD", "AUD_USD")
                                usd_dir = (
                                    "long" if (group == usd_quote_group and direction == "SELL")
                                           or (group != usd_quote_group and direction == "BUY")
                                    else "short"
                                )
                                return _rejected(
                                    instrument, direction, entry, atr,
                                    f"⚠️ Correlation block: already {existing_dir} {open_inst.replace('_','/')} "
                                    f"— adding {direction} {instrument.replace('_','/')} doubles USD-{usd_dir} exposure"
                                )

        # ── Stop-Loss & Take-Profit ────────────────────────────────────────────
        mult = atr_multiplier or self._cfg.atr_stop_multiplier

        stop_loss = atr_stop_loss(entry, atr, direction, multiplier=mult)
        sl_distance = abs(entry - stop_loss)

        # ATR-based dynamic TP: TP scales with volatility, not a fixed RR ratio.
        # Floor ensures we never accept worse than min_rr_floor:1 RR.
        atr_tp_distance = atr * self._cfg.atr_tp_multiplier
        min_tp_distance = sl_distance * self._cfg.min_rr_floor
        tp_distance = max(atr_tp_distance, min_tp_distance)

        # Session-aware scaling (Fix 6): reduce TP during low-liquidity sessions
        # so price is more likely to reach it before spread widens or momentum fades.
        session_mult = get_tp_session_multiplier()
        tp_distance = max(tp_distance * session_mult, min_tp_distance)

        take_profit = entry + tp_distance if direction.upper() == "BUY" else entry - tp_distance

        # RR used for Kelly / reward sizing — use actual achieved ratio
        rr = tp_distance / sl_distance if sl_distance > 0 else (rr_ratio or self._cfg.default_rr_ratio)

        stop_pips = price_to_pips(sl_distance, instrument)
        tp_pips = price_to_pips(tp_distance, instrument)

        log.info(
            f"ATR-TP: atr={atr:.5f} * {self._cfg.atr_tp_multiplier} = {atr_tp_distance:.5f} | "
            f"session_mult={session_mult:.2f} | final_tp_dist={tp_distance:.5f} | RR={rr:.2f}"
        )

        # ── Position Sizing ────────────────────────────────────────────────────
        measured_wr = self.measured_win_rate
        effective_win_rate = measured_wr if measured_wr is not None else win_rate_estimate
        kelly = kelly_fraction(effective_win_rate, rr)

        if ai_risk_pct is not None:
            # Claude override — honour it within hard bounds
            risk_pct = max(self._cfg.min_risk_per_trade_pct,
                           min(ai_risk_pct, self._cfg.max_risk_per_trade_pct))
        elif confidence is not None:
            # Confidence-tiered sizing — primary path
            tiered = confidence_scaled_risk(confidence, self._cfg.min_signal_confidence)
            if tiered == 0.0:
                return _rejected(
                    instrument, direction, entry, atr,
                    f"⚠️ confidence_scaled_risk returned 0 for confidence={confidence:.2f}"
                )
            risk_pct = tiered
            log.info(
                f"Confidence-tiered sizing: conf={confidence:.2f} → risk={risk_pct:.1f}% "
                f"(kelly={kelly:.3f}, kelly_pct={kelly*100:.1f}%)"
            )
        else:
            # No confidence provided — fall back to Kelly-informed sizing
            kelly_pct = kelly * 100.0
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

        conf_str = f"{confidence:.2f}" if confidence is not None else "n/a"
        log.info(
            f"Risk proposal: {direction} {instrument} "
            f"conf={conf_str} "
            f"risk={risk_pct:.2f}% (${risk_usd:.2f}) "
            f"sl={stop_pips:.1f}pips tp={tp_pips:.1f}pips R:R={rr_actual:.1f} "
            f"units={units:,}"
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

    def update_peak(self, account_balance: float) -> None:
        """Update peak balance — call after every account balance fetch."""
        if account_balance > self._peak_balance:
            self._peak_balance = account_balance

    @property
    def total_drawdown_pct(self) -> float:
        """Current drawdown from peak equity as a percentage."""
        if self._peak_balance <= 0:
            return 0.0
        # NOTE: this is a snapshot using cached peak — call update_peak() first for accuracy
        return (self._peak_balance - self._initial_balance) / self._peak_balance * 100.0

    def current_drawdown_pct(self, account_balance: float) -> float:
        """Live drawdown from peak using the current account balance."""
        if self._peak_balance <= 0:
            return 0.0
        return max(0.0, (self._peak_balance - account_balance) / self._peak_balance * 100.0)

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
            wr = self.measured_win_rate
            wr_str = f" | WR: {wr:.1%}" if wr is not None else ""
            log.info(
                f"Trade closed: {trade_id} P&L={pnl_pips:+.1f}pips | "
                f"W/L: {self._wins}/{self._losses}{wr_str}"
            )
            return pnl_pips

    async def sync_open_trades(self, oanda_open_ids: set[str]) -> None:
        """
        Reconcile _open_trades with OANDA's actual open trade IDs.
        Removes any stale entries that OANDA no longer reports as open.
        This prevents phantom trades from blocking correlation checks
        and concurrent trade limits.
        """
        async with self._lock:
            stale = [tid for tid in self._open_trades if tid not in oanda_open_ids]
            for tid in stale:
                trade = self._open_trades.pop(tid)
                log.warning(
                    f"Purged stale trade from risk manager: {tid} "
                    f"{trade.instrument} {trade.direction} (no longer open on OANDA)"
                )

    @property
    def open_trade_count(self) -> int:
        return len(self._open_trades)

    @property
    def daily_pnl_usd(self) -> float:
        return self._daily_pnl_usd

    @property
    def is_kill_switch_active(self) -> bool:
        ref_balance = self._peak_balance if self._peak_balance > 0 else 10_000.0
        daily_loss_limit = ref_balance * self._cfg.daily_loss_limit_pct / 100.0
        return self._daily_pnl_usd <= -daily_loss_limit

    def daily_status(self, account_balance: float) -> dict:
        """Return current daily P&L status relative to targets."""
        profit_target = account_balance * self._cfg.daily_profit_target_pct / 100.0
        loss_limit = account_balance * self._cfg.daily_loss_limit_pct / 100.0
        pnl = self._daily_pnl_usd

        if pnl >= profit_target:
            status = "🎯 TARGET HIT — Trading locked"
            locked = True
        elif pnl <= -loss_limit:
            status = "🛑 LOSS LIMIT HIT — Trading locked"
            locked = True
        elif pnl >= profit_target * 0.5:
            status = "✅ On track — halfway to target"
            locked = False
        elif pnl <= -loss_limit * 0.5:
            status = "⚠️ Caution — halfway to loss limit"
            locked = False
        else:
            status = "🟢 Open — no limits hit"
            locked = False

        return {
            "daily_pnl_usd": round(pnl, 2),
            "profit_target_usd": round(profit_target, 2),
            "loss_limit_usd": round(loss_limit, 2),
            "progress_pct": round((pnl / profit_target * 100) if profit_target > 0 else 0, 1),
            "status": status,
            "trading_locked": locked,
        }

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

    def record_close(self, pnl_usd: float) -> None:
        """
        Update in-memory win/loss counters from a closed trade's USD P&L.
        Always call this alongside close_trade_record() in trade_repo.
        Break-even (pnl == 0) is excluded from both win and loss counters.
        """
        if pnl_usd > 0:
            self._wins += 1
            result = "WIN"
        elif pnl_usd < 0:
            self._losses += 1
            result = "LOSS"
        else:
            result = "BREAKEVEN"   # not counted in either — doesn't distort win rate
        self._daily_pnl_usd += pnl_usd
        log.info(f"Trade recorded: {result} ${pnl_usd:+.2f} | W/L: {self._wins}/{self._losses}")

    async def load_stats_from_db(self) -> None:
        """
        Load win/loss counters and today's P&L from the database.
        Call once at application startup to restore state after restarts.
        """
        try:
            from forexmind.data.trade_repo import get_stats
            stats = await get_stats()
            async with self._lock:
                self._wins = stats["wins"]
                self._losses = stats["losses"]
                self._daily_pnl_usd = stats["today_pnl_usd"]
            log.info(
                f"Stats loaded from DB: {self._wins}W / {self._losses}L "
                f"| today P&L: ${self._daily_pnl_usd:+.2f}"
            )
        except Exception as e:
            log.error(f"load_stats_from_db error: {e}")

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
