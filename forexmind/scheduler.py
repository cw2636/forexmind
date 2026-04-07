"""
ForexMind — Automated Signal Scheduler
========================================
Wakes up during the US trading session (12:00–21:00 UTC / 8AM–5PM ET),
scans all recommended pairs every 15 minutes, and sends a Telegram push
notification when a high-confidence BUY or SELL signal is detected.

Signal alert criteria:
  • action is BUY or SELL (not HOLD)
  • confidence ≥ MIN_CONFIDENCE (default 65%)
  • R:R ratio ≥ MIN_RR (default 1.5)
  • pair not already alerted in the last COOLDOWN_MINUTES

Usage:
  python main.py scheduler
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, time, timezone
from typing import NamedTuple

import pytz

from forexmind.config.settings import get_settings
from forexmind.utils.logger import get_logger
from forexmind.utils.session_times import get_session_status, best_pairs_for_session

log = get_logger(__name__)

UTC = pytz.utc

# ── Tunable parameters ────────────────────────────────────────────────────────
SCAN_INTERVAL_MINUTES = 15      # How often to scan in minutes

# Confidence thresholds (with RL weight cut to 0.05, 3 strategies agreeing at 70% = ~0.665):
#   2 strategies agreeing → ~55–65%   (MODERATE — alert only)
#   3 strategies agreeing → ~65–80%   (STRONG — tradeable)
MIN_CONFIDENCE = 55.0           # Alert threshold raised — no more noise alerts
AUTO_TRADE_CONFIDENCE = 65.0    # Auto-trade: 3 strong strategies agree
MIN_RR = 2.0                    # 2:1 R:R minimum — standard professional threshold
COOLDOWN_MINUTES = 90           # H1 candles close every hour — 90min prevents double-alert
SESSION_START = time(7, 0)      # London open (UTC)
SESSION_END = time(21, 0)       # US close (UTC)


class AlertRecord(NamedTuple):
    pair: str
    action: str
    alerted_at: datetime


def _is_us_session(dt: datetime) -> bool:
    """Return True if dt falls within the US trading window (UTC)."""
    t = dt.time().replace(second=0, microsecond=0)
    return SESSION_START <= t < SESSION_END


def _format_alert(data: dict) -> str:
    """Format signal data into a Telegram HTML alert message."""
    sig = data.get("signal", {})
    ind = data.get("indicators", {})
    session = data.get("session", {})
    sentiment = data.get("news_sentiment", {})

    pair = data.get("instrument", "").replace("_", "/")
    action = sig.get("action", "HOLD")
    emoji = "🟢" if action == "BUY" else "🔴"
    overlap = "🔥 OVERLAP" if session.get("is_overlap") else ""

    return (
        f"{emoji} <b>ALERT: {action} {pair}</b> {overlap}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Entry:       <code>{sig.get('entry', '-')}</code>\n"
        f"Stop Loss:   <code>{sig.get('stop_loss', '-')}</code> ({sig.get('stop_loss_pips', '-')} pips)\n"
        f"Take Profit: <code>{sig.get('take_profit', '-')}</code> ({sig.get('take_profit_pips', '-')} pips)\n"
        f"R:R Ratio:   <b>{sig.get('risk_reward', '-')}:1</b>\n"
        f"Confidence:  <b>{sig.get('confidence', 0)}%</b>\n"
        f"Strategies:  {sig.get('agreeing_strategies', '-')} in agreement\n"
        f"\n"
        f"📈 RSI: {ind.get('rsi', '-')} [{ind.get('rsi_zone', '-')}]  "
        f"EMA: {ind.get('ema_trend', '-')}\n"
        f"📰 News: {sentiment.get('impact', 'neutral').upper()} ({sentiment.get('score', 0):+.3f})\n"
        f"\n"
        f"<i>{data.get('reasoning', '')[:200]}</i>"
    ).strip()


async def _send_telegram(bot, chat_id: str, text: str) -> None:
    """Send a message via the Telegram bot."""
    try:
        from telegram.constants import ParseMode
        await bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode=ParseMode.HTML,
        )
    except Exception as e:
        log.error(f"Telegram send failed: {e}")


async def _save_signal_to_db(data: dict) -> int | None:
    """Persist a generated signal to the database. Returns the new signal DB id."""
    try:
        from forexmind.data.database import get_session
        from forexmind.data.models import Signal, SignalSource, Direction

        sig = data.get("signal", {})
        async with get_session() as session:
            record = Signal(
                instrument=data.get("instrument", ""),
                timeframe=data.get("timeframe", "M5"),
                generated_at=datetime.now(timezone.utc),
                direction=Direction(sig.get("action", "HOLD")),
                source=SignalSource.ENSEMBLE,
                confidence=sig.get("confidence", 0) / 100.0,
                entry_price=float(sig.get("entry", 0)),
                stop_loss=float(sig.get("stop_loss", 0)),
                take_profit=float(sig.get("take_profit", 0)),
                risk_pct=float(sig.get("risk_pct", 1.0)),
                reasoning=data.get("reasoning", ""),
            )
            session.add(record)
            await session.flush()   # populates record.id before commit
            signal_id = record.id
        log.info(f"Signal saved to DB (id={signal_id})")
        return signal_id
    except Exception as e:
        log.error(f"Failed to save signal to DB: {e}")
        return None


async def _save_trade_to_db(signal_id: int | None, data: dict, trade_result: dict) -> None:
    """Persist an executed trade to the database."""
    try:
        from forexmind.data.database import get_session
        from forexmind.data.models import Trade, TradeStatus, Direction

        sig = data.get("signal", {})
        async with get_session() as session:
            record = Trade(
                signal_id=signal_id,
                instrument=data.get("instrument", ""),
                direction=Direction(sig.get("action", "BUY")),
                units=trade_result.get("units", 0),
                entry_price=float(trade_result.get("filled_price", sig.get("entry", 0))),
                stop_loss=float(sig.get("stop_loss", 0)),
                take_profit=float(sig.get("take_profit", 0)),
                opened_at=datetime.now(timezone.utc),
                status=TradeStatus.OPEN,
                oanda_trade_id=str(trade_result.get("trade_id", "")),
            )
            session.add(record)
        log.info(f"Trade saved to DB (oanda_id={trade_result.get('trade_id')})")
    except Exception as e:
        log.error(f"Failed to save trade to DB: {e}")


async def _auto_place_trade(data: dict) -> tuple[bool, str, dict]:
    """
    Place a trade automatically using the risk manager for position sizing.
    Returns (success, message).
    """
    from forexmind.agents.tools import _place_trade
    from forexmind.data.oanda_client import get_oanda_client
    from forexmind.risk.manager import get_risk_manager

    sig = data.get("signal", {})
    ind = data.get("indicators", {})
    instrument = data.get("instrument", "")
    direction = sig.get("action")  # "BUY" or "SELL"
    entry = float(sig.get("entry", 0))
    stop_loss = float(sig.get("stop_loss", 0))
    take_profit = float(sig.get("take_profit", 0))
    atr = float(ind.get("atr", 0.0005))

    try:
        # Get account balance for position sizing
        client = get_oanda_client()
        acc = await client.get_account()
        rm = get_risk_manager()

        proposal = rm.calculate_risk(
            instrument=instrument,
            direction=direction,
            entry=entry,
            atr=atr,
            account_balance=acc.balance,
        )

        if not proposal.approved:
            return False, f"Risk manager rejected: {proposal.rejection_reason}"

        result_str = await _place_trade(
            instrument=instrument,
            direction=direction,
            units=proposal.units,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        result = json.loads(result_str)

        if "error" in result:
            return False, f"Order failed: {result['error']}", {}

        msg = (
            f"Trade #{result.get('trade_id', '?')} | "
            f"Filled @ {result.get('filled_price', entry)} | "
            f"{proposal.units:,} units | Risk: {proposal.risk_pct:.2f}%"
        )
        return True, msg, result

    except Exception as e:
        return False, str(e), {}


async def _scan_pairs(bot, chat_id: str, recent_alerts: dict[str, AlertRecord]) -> None:
    """Scan all session-recommended pairs and fire alerts for qualifying signals."""
    from forexmind.agents.tools import _get_signal

    pairs = best_pairs_for_session()
    if not pairs:
        log.info("No recommended pairs for current session — skipping scan")
        return

    now = datetime.now(UTC)
    log.info(f"Scanning {len(pairs)} pairs: {', '.join(pairs)}")

    for pair in pairs:
        # Check cooldown — skip if alerted recently
        if pair in recent_alerts:
            elapsed = (now - recent_alerts[pair].alerted_at).total_seconds() / 60
            if elapsed < COOLDOWN_MINUTES:
                log.info(f"{pair}: in cooldown ({elapsed:.0f}/{COOLDOWN_MINUTES} min)")
                continue

        try:
            result_str = await _get_signal(pair, "M5", 300)
            data = json.loads(result_str)

            if "error" in data:
                log.warning(f"{pair}: signal error — {data['error']}")
                continue

            sig = data.get("signal", {})
            action = sig.get("action", "HOLD")
            confidence = float(sig.get("confidence", 0))
            rr = float(sig.get("risk_reward", 0))

            log.info(f"{pair}: {action} | conf={confidence}% | R:R={rr}")

            # Must be actionable and meet minimum thresholds
            if action not in ("BUY", "SELL") or confidence < MIN_CONFIDENCE or rr < MIN_RR:
                log.info(f"{pair}: signal below threshold — skipping")
                continue

            # Save every qualifying signal to DB for historical analysis
            signal_id = await _save_signal_to_db(data)

            alert_text = _format_alert(data)

            if confidence >= AUTO_TRADE_CONFIDENCE:
                # HIGH CONFIDENCE — place trade automatically
                log.info(f"{pair}: confidence {confidence}% >= {AUTO_TRADE_CONFIDENCE}% — auto-trading")
                success, trade_msg, trade_result = await _auto_place_trade(data)

                if success:
                    await _save_trade_to_db(signal_id, data, trade_result)
                    full_text = (
                        alert_text +
                        f"\n\n⚡ <b>AUTO-TRADED</b>\n{trade_msg}"
                    )
                    log.info(f"✅ Auto-trade placed: {action} {pair} — {trade_msg}")
                else:
                    full_text = (
                        alert_text +
                        f"\n\n⚠️ <b>AUTO-TRADE FAILED</b>\n{trade_msg}\n"
                        f"<i>Please place manually.</i>"
                    )
                    log.error(f"❌ Auto-trade failed: {pair} — {trade_msg}")
            else:
                # MODERATE CONFIDENCE — alert only, user decides
                success, trade_msg, trade_result = False, "", {}
                full_text = alert_text + "\n\n👆 <i>Review and place manually if you agree.</i>"
                log.info(f"✅ Alert sent: {action} {pair} ({confidence}% conf, {rr} R:R)")

            await _send_telegram(bot, chat_id, full_text)
            recent_alerts[pair] = AlertRecord(pair=pair, action=action, alerted_at=now)

        except Exception as e:
            log.error(f"{pair}: scan error — {e}")

        # Small delay between pair requests to avoid rate limiting
        await asyncio.sleep(1)


# Hour (UTC) at which daily auto-retrain runs — 05:00 UTC (before London open)
RETRAIN_HOUR_UTC = 5


async def _run_daily_retrain(bot, chat_id: str) -> None:
    """
    Retrain LightGBM and LSTM models on fresh data.
    Runs in a thread executor to avoid blocking the event loop.
    Reports results to Telegram.
    """
    import concurrent.futures

    log.info("Starting daily auto-retrain...")
    await _send_telegram(bot, chat_id, "🔄 <b>Daily retrain started</b> — training LightGBM &amp; LSTM on latest data...")

    loop = asyncio.get_event_loop()

    def _retrain() -> dict:
        results: dict[str, str] = {}
        try:
            from forexmind.strategy.ml_strategy import LightGBMStrategy, LSTMStrategy

            lgbm = LightGBMStrategy()
            lgbm_result = lgbm.train()
            cv_acc = lgbm_result.get("cv_mean_accuracy", 0)
            ho_acc = lgbm_result.get("holdout_accuracy", 0)
            results["lightgbm"] = f"CV={cv_acc:.1%} | Holdout={ho_acc:.1%}"
        except Exception as e:
            results["lightgbm"] = f"FAILED: {e}"
            log.error(f"LightGBM retrain error: {e}")

        try:
            from forexmind.strategy.ml_strategy import LSTMStrategy

            lstm = LSTMStrategy()
            lstm_result = lstm.train()
            val_acc = lstm_result.get("best_val_accuracy", lstm_result.get("val_accuracy", 0))
            results["lstm"] = f"Val={val_acc:.1%}"
        except Exception as e:
            results["lstm"] = f"FAILED: {e}"
            log.error(f"LSTM retrain error: {e}")

        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        results = await loop.run_in_executor(pool, _retrain)

    lines = ["✅ <b>Daily retrain complete</b>"]
    for name, summary in results.items():
        icon = "✅" if "FAILED" not in summary else "❌"
        lines.append(f"{icon} <b>{name.upper()}</b>: {summary}")

    await _send_telegram(bot, chat_id, "\n".join(lines))
    log.info(f"Daily retrain done: {results}")


async def run_scheduler() -> None:
    """
    Main scheduler loop.

    - Outside US session: sleeps and checks every minute until session opens
    - Inside US session: scans all pairs every SCAN_INTERVAL_MINUTES
    - Daily retrain at RETRAIN_HOUR_UTC (05:00 UTC) before London open
    - Sends a wake-up and sleep message to Telegram
    """
    from telegram import Bot

    cfg = get_settings()

    if not cfg.telegram.is_configured:
        raise ValueError("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set in .env")

    bot = Bot(token=cfg.telegram.bot_token)
    chat_id = cfg.telegram.chat_id

    log.info("ForexMind Scheduler starting...")

    # Only notify on startup if markets are open — no weekend pings
    _startup_status = get_session_status(datetime.now(UTC))
    if not _startup_status.is_weekend:
        await _send_telegram(
            bot, chat_id,
            "🤖 <b>ForexMind Scheduler started</b>\n"
            f"Scanning every {SCAN_INTERVAL_MINUTES} min during London + US sessions (07:00–21:00 UTC)\n\n"
            f"📢 Alert only:  {MIN_CONFIDENCE}%–{AUTO_TRADE_CONFIDENCE - 1}% confidence\n"
            f"⚡ Auto-trade:  ≥{AUTO_TRADE_CONFIDENCE}% confidence + {MIN_RR}:1 R:R\n\n"
            f"🔄 Daily retrain: {RETRAIN_HOUR_UTC:02d}:00 UTC\n\n"
            f"<i>Paper trading mode — no real money at risk</i>"
        )
    else:
        log.info("Weekend — startup Telegram notification suppressed")

    recent_alerts: dict[str, AlertRecord] = {}
    session_open_announced = False
    last_retrain_date: datetime | None = None

    while True:
        now = datetime.now(UTC)
        status = get_session_status(now)

        # ── Daily retrain check ───────────────────────────────────────────────
        # Fire once per day at RETRAIN_HOUR_UTC, before London open
        today = now.date()
        if (
            now.hour == RETRAIN_HOUR_UTC
            and (last_retrain_date is None or last_retrain_date < today)
            and not status.is_weekend
        ):
            last_retrain_date = today
            # Run in background — don't block the scheduler loop
            asyncio.create_task(_run_daily_retrain(bot, chat_id))

        if status.is_weekend:
            log.info("Weekend — markets closed, sleeping 1 hour")
            await asyncio.sleep(3600)
            continue

        in_session = _is_us_session(now)

        if in_session:
            if not session_open_announced:
                overlap_note = ""
                if status.active_overlaps:
                    overlap_note = f"\n🔥 <b>OVERLAP ACTIVE: {', '.join(status.active_overlaps)}</b> — Peak liquidity!"
                await _send_telegram(
                    bot, chat_id,
                    f"🟢 <b>Session OPEN</b> — Scheduler scanning now{overlap_note}"
                )
                session_open_announced = True

            await _scan_pairs(bot, chat_id, recent_alerts)
            log.info(f"Scan complete. Next scan in {SCAN_INTERVAL_MINUTES} minutes.")
            await asyncio.sleep(SCAN_INTERVAL_MINUTES * 60)

        else:
            if session_open_announced:
                # Just closed
                await _send_telegram(
                    bot, chat_id,
                    "🔴 <b>Session CLOSED</b> — Scheduler sleeping until 07:00 UTC (London open)"
                )
                session_open_announced = False
                recent_alerts.clear()  # Reset cooldowns for next session

            # Sleep until next minute check
            await asyncio.sleep(60)
