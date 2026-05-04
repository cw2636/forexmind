"""
ForexMind — Trade Repository
=============================
Persists opened and closed trades to SQLite for durable win/loss tracking.

All stats survive service restarts and are the source of truth for /stats.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

from sqlalchemy import select

from forexmind.data.database import get_session
from forexmind.data.models import Direction, Trade, TradeStatus
from forexmind.utils.logger import get_logger

log = get_logger(__name__)


async def open_trade(
    oanda_trade_id: str,
    instrument: str,
    direction: str,
    units: int,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
) -> None:
    """Insert an OPEN trade record when a trade is placed."""
    try:
        async with get_session() as session:
            # Guard against duplicate inserts for the same OANDA trade ID
            existing = await session.execute(
                select(Trade).where(Trade.oanda_trade_id == str(oanda_trade_id)).limit(1)
            )
            if existing.scalar_one_or_none():
                log.debug(f"open_trade: {oanda_trade_id} already in DB — skipping duplicate insert")
                return
            trade = Trade(
                oanda_trade_id=str(oanda_trade_id),
                instrument=instrument,
                direction=Direction(direction),
                units=units,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                opened_at=datetime.now(timezone.utc),
                status=TradeStatus.OPEN,
            )
            session.add(trade)
    except Exception as e:
        log.error(f"open_trade DB error: {e}")


async def close_trade_record(
    oanda_trade_id: str,
    exit_price: float,
    pnl_usd: float,
    closed_at: datetime | None = None,
) -> bool:
    """
    Update an existing OPEN trade record to CLOSED.
    If the trade was never recorded (placed before DB wiring), inserts a minimal
    closed record so the win/loss count is still accurate.

    Returns True if the trade was newly closed, False if it was already CLOSED
    (duplicate-guard — callers should skip notifications when False).
    """
    try:
        async with get_session() as session:
            result = await session.execute(
                select(Trade)
                .where(Trade.oanda_trade_id == str(oanda_trade_id))
                .order_by(Trade.id.desc())
                .limit(1)
            )
            trade = result.scalar_one_or_none()
            now = closed_at or datetime.now(timezone.utc)
            if trade:
                if trade.status == TradeStatus.CLOSED:
                    log.debug(f"close_trade_record: {oanda_trade_id} already CLOSED — skipping")
                    return False
                trade.exit_price = exit_price
                trade.pnl_usd = pnl_usd
                trade.closed_at = now
                trade.status = TradeStatus.CLOSED
            else:
                # Trade opened before DB wiring — insert a closed-only stub
                session.add(Trade(
                    oanda_trade_id=str(oanda_trade_id),
                    instrument="UNKNOWN",
                    direction=Direction.BUY,
                    units=0,
                    entry_price=0.0,
                    stop_loss=0.0,
                    take_profit=0.0,
                    exit_price=exit_price,
                    pnl_usd=pnl_usd,
                    opened_at=now,
                    closed_at=now,
                    status=TradeStatus.CLOSED,
                ))
        return True
    except Exception as e:
        log.error(f"close_trade_record DB error: {e}")
        return False


async def get_stats() -> dict:
    """Return all-time and today's win/loss/pnl stats directly from OANDA.

    Uses OANDA's closed-trade list and account summary as the single source
    of truth — no reliance on the local DB which may have missed early trades.
    """
    try:
        from forexmind.data.oanda_client import get_oanda_client

        client = get_oanda_client()
        acc, closed_trades = await _async_gather(
            client.get_account(),
            client.get_recently_closed_trades(count=500),
        )

        total = len(closed_trades)
        pnls = [float(t.get("realizedPL", 0) or 0) for t in closed_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        decisive = len(wins) + len(losses)
        total_pnl = acc.balance - 100_000.0  # OANDA balance vs starting capital

        best_trade = max(closed_trades, key=lambda t: float(t.get("realizedPL", 0) or 0)) if closed_trades else None
        worst_trade = min(closed_trades, key=lambda t: float(t.get("realizedPL", 0) or 0)) if closed_trades else None
        best_pnl = float(best_trade.get("realizedPL", 0) or 0) if best_trade else 0.0
        worst_pnl = float(worst_trade.get("realizedPL", 0) or 0) if worst_trade else 0.0
        best_inst = best_trade.get("instrument", "") if best_trade else ""
        worst_inst = worst_trade.get("instrument", "") if worst_trade else ""

        today = date.today()
        today_pnls = []
        today_closed_count = 0
        for t in closed_trades:
            close_time = t.get("closeTime", "")
            if close_time:
                try:
                    ct = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
                    if ct.date() == today:
                        today_pnls.append(float(t.get("realizedPL", 0) or 0))
                        today_closed_count += 1
                except (ValueError, TypeError):
                    pass
        today_pnl = sum(today_pnls)

        return {
            "total_closed": total,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / decisive, 4) if decisive > 0 else None,
            "total_pnl_usd": round(total_pnl, 2),
            "today_pnl_usd": round(today_pnl, 2),
            "today_closed": today_closed_count,
            "avg_pnl_usd": round(total_pnl / total, 2) if total > 0 else 0.0,
            "best_trade_usd": round(best_pnl, 2),
            "best_trade_inst": best_inst,
            "worst_trade_usd": round(worst_pnl, 2),
            "worst_trade_inst": worst_inst,
            "using_measured_wr": total >= 30,
            # Pass through OANDA account data so cmd_stats doesn't need a
            # separate _get_account() call.
            "_account": {
                "balance": acc.balance,
                "nav": acc.nav,
                "unrealized_pnl": acc.unrealized_pnl,
                "open_trade_count": acc.open_trade_count,
            },
        }
    except Exception as e:
        log.error(f"get_stats OANDA error: {e}")
        return {
            "total_closed": 0, "wins": 0, "losses": 0,
            "win_rate": None, "total_pnl_usd": 0.0, "today_pnl_usd": 0.0,
            "today_closed": 0, "avg_pnl_usd": 0.0,
            "best_trade_usd": 0.0, "worst_trade_usd": 0.0,
            "using_measured_wr": False,
            "_account": None,
        }


async def _async_gather(*coros):
    """Thin wrapper so we can import asyncio only here."""
    import asyncio
    return await asyncio.gather(*coros)
