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
MIN_CONFIDENCE = 62.0           # Matches MIN_ENSEMBLE_CONFIDENCE — only real signals pass
AUTO_TRADE_CONFIDENCE = 70.0    # Auto-trade: 3 strong strategies agree at ≥70% each
MIN_RR = 2.0                    # 2:1 R:R minimum — standard professional threshold
COOLDOWN_MINUTES = 45           # M15 timeframe — 3 candle cooldown prevents back-to-back noise
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

    # ATR status line — shows ratio vs 20-bar average with a caution label
    atr_line = ""
    atr_now = ind.get("atr")
    atr_avg = ind.get("atr_avg_20")
    if atr_now and atr_avg and atr_avg > 0:
        ratio = atr_now / atr_avg
        if ratio >= 2.0:
            atr_label = "⚠️ HIGH — news/event volatility"
        elif ratio >= 1.5:
            atr_label = "🟡 ELEVATED"
        else:
            atr_label = "🟢 normal"
        atr_line = f"📊 ATR: {atr_now:.5f}  ({ratio:.1f}x avg)  {atr_label}\n"

    # News line — flag non-neutral sentiment clearly
    news_score = float(sentiment.get("score", 0))
    news_impact = sentiment.get("impact", "neutral").upper()
    if abs(news_score) >= 0.3:
        news_label = "⚠️ OPPOSING DIRECTION" if (
            (action == "BUY" and news_score < 0) or (action == "SELL" and news_score > 0)
        ) else "📢 SUPPORTING"
    elif abs(news_score) >= 0.15:
        news_label = "🟡 moderate"
    else:
        news_label = ""
    news_line = f"📰 News: {news_impact} ({news_score:+.3f})" + (f"  {news_label}" if news_label else "") + "\n"

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
        f"{atr_line}"
        f"{news_line}"
        f"\n"
        f"<i>{data.get('reasoning', '')[:200]}</i>"
    ).strip()


# ── Economic calendar ─────────────────────────────────────────────────────────
_calendar_cache: dict = {"events": [], "fetched_at": None}
_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
_CALENDAR_TTL = 21600   # refresh every 6 hours
_NEWS_BLOCK_MINUTES = 30


async def _fetch_calendar() -> list[dict]:
    """
    Fetch ForexFactory high-impact events for the current week.
    Returns a list of dicts: {currency, title, dt (UTC datetime)}.
    Cached for 6 hours to avoid hammering the feed.
    """
    import aiohttp
    import xml.etree.ElementTree as ET
    from datetime import datetime as _dt

    now_ts = datetime.now(UTC).timestamp()
    if (
        _calendar_cache["fetched_at"]
        and now_ts - _calendar_cache["fetched_at"] < _CALENDAR_TTL
    ):
        return _calendar_cache["events"]

    events: list[dict] = []
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.get(_CALENDAR_URL) as resp:
                text = await resp.text()
        root = ET.fromstring(text)
        for ev in root.findall(".//event"):
            impact = (ev.findtext("impact") or "").strip()
            if impact.lower() != "high":
                continue
            currency = (ev.findtext("country") or "").strip().upper()
            title    = (ev.findtext("title")   or "").strip()
            date_str = (ev.findtext("date")    or "").strip()
            time_str = (ev.findtext("time")    or "").strip()
            if not date_str or not time_str or time_str.lower() == "all day":
                continue
            try:
                # e.g. "Apr 04, 2025" + "8:30am"
                dt_naive = _dt.strptime(f"{date_str} {time_str}", "%b %d, %Y %I:%M%p")
                # FF times are US Eastern — convert to UTC
                import pytz as _pytz
                et = _pytz.timezone("America/New_York")
                dt_utc = et.localize(dt_naive).astimezone(UTC)
                events.append({"currency": currency, "title": title, "dt": dt_utc})
            except Exception:
                continue
        _calendar_cache["events"] = events
        _calendar_cache["fetched_at"] = now_ts
        log.info(f"Economic calendar: loaded {len(events)} high-impact events")
    except Exception as e:
        log.warning(f"Economic calendar fetch failed: {e} — skipping news block")
    return events


def _is_near_high_impact_event(
    instrument: str,
    now: datetime,
    events: list[dict],
    window_minutes: int = _NEWS_BLOCK_MINUTES,
) -> tuple[bool, str]:
    """
    Return (True, reason) if a high-impact event for the instrument's currencies
    is within ±window_minutes of now, else (False, '').
    """
    if not events:
        return False, ""
    parts = instrument.replace("_", "/").split("/")
    currencies = {p.upper() for p in parts}
    window = window_minutes * 60
    for ev in events:
        if ev["currency"] not in currencies:
            continue
        diff = abs((ev["dt"] - now).total_seconds())
        if diff <= window:
            direction = "in" if ev["dt"] > now else "ago"
            mins = int(diff // 60)
            return True, f"{ev['currency']} {ev['title']} ({mins} min {direction})"
    return False, ""


async def _send_telegram(bot, chat_id: str, text: str) -> None:
    """Send a message via the Telegram bot (15 s timeout guard)."""
    try:
        from telegram.constants import ParseMode
        await asyncio.wait_for(
            bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.HTML),
            timeout=15.0,
        )
    except asyncio.TimeoutError:
        log.warning("Telegram send timed out after 15s — skipping")
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
                timeframe=data.get("timeframe", "M15"),
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
    """Persist an executed trade to the database. Skips if oanda_trade_id already exists."""
    try:
        from forexmind.data.database import get_session
        from forexmind.data.models import Trade, TradeStatus, Direction
        from sqlalchemy import select

        oanda_id = str(trade_result.get("trade_id", ""))
        sig = data.get("signal", {})

        async with get_session() as session:
            # Guard against duplicate inserts from concurrent code paths
            if oanda_id:
                existing = await session.execute(
                    select(Trade).where(Trade.oanda_trade_id == oanda_id)
                )
                if existing.scalar_one_or_none() is not None:
                    log.warning(f"Trade {oanda_id} already in DB — skipping duplicate insert")
                    return

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
                oanda_trade_id=oanda_id,
            )
            session.add(record)
        log.info(f"Trade saved to DB (oanda_id={oanda_id})")
    except Exception as e:
        log.error(f"Failed to save trade to DB: {e}")


async def _auto_place_trade(data: dict) -> tuple[bool, str, dict]:
    """
    Place a trade automatically using the risk manager for position sizing.
    Returns (success, message).
    """
    from forexmind.agents.tools import _place_trade
    from forexmind.data.oanda_client import get_oanda_client
    from forexmind.risk.manager import get_risk_manager, OpenTrade

    sig = data.get("signal", {})
    ind = data.get("indicators", {})
    instrument = data.get("instrument", "")
    direction = sig.get("action")  # "BUY" or "SELL"
    entry = float(sig.get("entry", 0))
    stop_loss = float(sig.get("stop_loss", 0))
    atr = float(ind.get("atr", 0.0005))
    # Confidence arrives as 0–100 from signal; normalise to 0–1 for risk manager
    confidence = float(sig.get("confidence", 0)) / 100.0

    try:
        # Get account balance for position sizing
        client = get_oanda_client()
        acc = await client.get_account()
        rm = get_risk_manager()
        rm.update_peak(acc.balance)   # keep peak equity current

        proposal = rm.calculate_risk(
            instrument=instrument,
            direction=direction,
            entry=entry,
            atr=atr,
            account_balance=acc.balance,
            confidence=confidence,
        )

        if not proposal.approved:
            return False, f"Risk manager rejected: {proposal.rejection_reason}", {}

        # ── S/R TP cap (Fix 2) ─────────────────────────────────────────────────
        # Risk manager returns ATR-based TP. Cap it at the nearest S/R level so
        # we never set TP past a resistance (BUY) or support (SELL) that price
        # is unlikely to break on M15.
        take_profit = proposal.take_profit
        sl_distance = abs(proposal.entry_price - proposal.stop_loss)
        min_rr_floor = rm._cfg.min_rr_floor
        spread_buffer = atr * 0.15  # 15% ATR buffer before the S/R level

        if direction == "BUY":
            resistance = float(ind.get("resistance", 0))
            if resistance > entry and resistance < take_profit:
                capped_tp = resistance - spread_buffer
                if (capped_tp - entry) >= (sl_distance * min_rr_floor):
                    log.info(
                        f"{instrument}: TP capped at resistance {resistance:.5f} "
                        f"(was {take_profit:.5f}, capped to {capped_tp:.5f})"
                    )
                    take_profit = capped_tp
                else:
                    log.info(
                        f"{instrument}: S/R too close ({resistance:.5f}), RR < {min_rr_floor} — skipping trade"
                    )
                    return False, f"S/R too close — RR below {min_rr_floor}:1 after cap", {}
        elif direction == "SELL":
            support = float(ind.get("support", 0))
            if support > 0 and support < entry and support > take_profit:
                capped_tp = support + spread_buffer
                if (entry - capped_tp) >= (sl_distance * min_rr_floor):
                    log.info(
                        f"{instrument}: TP capped at support {support:.5f} "
                        f"(was {take_profit:.5f}, capped to {capped_tp:.5f})"
                    )
                    take_profit = capped_tp
                else:
                    log.info(
                        f"{instrument}: S/R too close ({support:.5f}), RR < {min_rr_floor} — skipping trade"
                    )
                    return False, f"S/R too close — RR below {min_rr_floor}:1 after cap", {}

        result_str = await _place_trade(
            instrument=instrument,
            direction=direction,
            units=proposal.units,
            stop_loss=proposal.stop_loss,
            take_profit=take_profit,
        )
        result = json.loads(result_str)

        if "error" in result or result.get("success") is False:
            err_msg = result.get("error") or result.get("message") or "Unknown"
            return False, f"Order failed: {err_msg}", {}

        trade_id = result.get("trade_id", "")
        filled = float(result.get("filled_price", 0)) or entry
        if trade_id:
            await rm.register_trade(OpenTrade(
                trade_id=str(trade_id),
                instrument=instrument,
                direction=direction,
                entry_price=filled,
                stop_loss=proposal.stop_loss,
                take_profit=take_profit,
                units=proposal.units,
            ))

        # Inject risk-manager SL/TP into result so callers can use the actual values
        result["actual_stop_loss"] = proposal.stop_loss
        result["actual_take_profit"] = take_profit

        msg = (
            f"Trade #{trade_id or '?'} | "
            f"Filled @ {filled} | "
            f"{proposal.units:,} units | Risk: {proposal.risk_pct:.2f}%"
        )
        return True, msg, result

    except Exception as e:
        return False, str(e), {}


async def _scan_pairs(bot, chat_id: str, recent_alerts: dict[str, AlertRecord]) -> None:
    """Scan all session-recommended pairs and fire alerts for qualifying signals."""
    from forexmind.agents.tools import _get_signal
    from forexmind.config.settings import get_settings

    # Sync any SL/TP-closed trades every scan so /stats is always accurate
    await _sync_closed_trades(bot, chat_id)

    pairs = best_pairs_for_session()
    if not pairs:
        log.info("No recommended pairs for current session — skipping scan")
        return

    # Enforce max_concurrent_trades — count distinct OANDA trade IDs currently open
    try:
        from forexmind.data.oanda_client import get_oanda_client
        open_trades = await get_oanda_client().get_open_trades()
        open_count = len(open_trades)
        max_trades = get_settings().risk.max_concurrent_trades
    except Exception:
        open_count = 0
        max_trades = 3

    now = datetime.now(UTC)
    log.info(f"Scanning {len(pairs)} pairs: {', '.join(pairs)} | open trades: {open_count}/{max_trades}")

    for pair in pairs:
        # Check cooldown — skip if alerted recently
        if pair in recent_alerts:
            elapsed = (now - recent_alerts[pair].alerted_at).total_seconds() / 60
            if elapsed < COOLDOWN_MINUTES:
                log.info(f"{pair}: in cooldown ({elapsed:.0f}/{COOLDOWN_MINUTES} min)")
                continue

        try:
            result_str = await _get_signal(pair, "M15", 300)
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

            # Parse agreeing strategy count from "2/4" format
            agreeing_str = sig.get("agreeing_strategies", "0/0")
            try:
                agreeing_count = int(str(agreeing_str).split("/")[0])
            except (ValueError, IndexError):
                agreeing_count = 0

            # ATR/news caution check — block auto-trade if conditions are risky.
            # ATR ≥ 2.0x average means elevated volatility (news event likely in progress).
            # The ensemble blocks signals at 2.5x; auto-trade should stop at 2.0x.
            ind = data.get("indicators", {})
            sentiment = data.get("news_sentiment", {})
            atr_now = ind.get("atr") or 0
            atr_avg = ind.get("atr_avg_20") or 0
            atr_ratio = (atr_now / atr_avg) if atr_avg > 0 else 0
            news_score = float(sentiment.get("score", 0))
            has_caution = (
                atr_ratio >= 2.0
                or (action == "BUY" and news_score <= -0.3)
                or (action == "SELL" and news_score >= 0.3)
            )
            if has_caution:
                log.info(
                    f"{pair}: ATR/news caution — ATR={atr_ratio:.1f}x news={news_score:+.2f} "
                    f"— suppressing auto-trade, sending alert only"
                )

            # Liquidity check — mirrors the ≥40% gate in /signals
            session_score = get_session_status(now).session_score
            low_liquidity = session_score < 0.4
            if low_liquidity:
                log.info(
                    f"{pair}: low liquidity (score={session_score:.2f}) "
                    f"— suppressing auto-trade, sending alert only"
                )

            # EMA alignment check — block counter-trend auto-trades.
            # If price structure (EMA trend) opposes the signal direction,
            # the ML ensemble is fighting the trend and losses are much more likely.
            # "weak_bullish" / "weak_bearish" are treated as misaligned to be safe.
            ema_trend = ind.get("ema_trend", "").lower()
            ema_aligned = (
                (action == "BUY"  and ema_trend == "bullish")
                or (action == "SELL" and ema_trend == "bearish")
            )
            if not ema_aligned:
                log.info(
                    f"{pair}: EMA misaligned (ema={ema_trend}, signal={action}) "
                    f"— suppressing auto-trade, sending alert only"
                )

            # Minimum SL distance — reject signals with SL < 10 pips (noise-stop territory)
            sl_pips = float(sig.get("stop_loss_pips", 0))
            sl_too_tight = sl_pips < 10.0
            if sl_too_tight:
                log.info(
                    f"{pair}: SL too tight ({sl_pips} pips < 10) "
                    f"— suppressing auto-trade, sending alert only"
                )

            # Market regime filter — ADX < 20 means ranging market, trend signals fail
            adx_val = float(ind.get("adx", 0))
            is_ranging = adx_val < 20.0
            if is_ranging:
                log.info(
                    f"{pair}: ranging market (ADX={adx_val:.1f} < 20) "
                    f"— suppressing auto-trade, sending alert only"
                )

            # Rule-based must actively agree — if rule_based says HOLD or opposite,
            # price structure opposes the signal and the trade is higher risk.
            strategy_votes = data.get("strategy_votes", {})
            rule_based_vote = strategy_votes.get("rule_based", "HOLD")
            rule_based_agrees = rule_based_vote == action
            if not rule_based_agrees:
                log.info(
                    f"{pair}: rule_based={rule_based_vote} disagrees with {action} "
                    f"— suppressing auto-trade, sending alert only"
                )

            # Economic calendar block — no auto-trades within 30 min of high-impact events
            calendar_events = await _fetch_calendar()
            near_event, event_reason = _is_near_high_impact_event(
                pair.replace("/", "_"), now, calendar_events
            )
            if near_event:
                log.info(
                    f"{pair}: near high-impact event ({event_reason}) "
                    f"— suppressing auto-trade, sending alert only"
                )

            # VWAP alignment — BUY needs price above VWAP (institutional support),
            # SELL needs price below VWAP (institutional resistance).
            # None = VWAP unknown (first bar of day) → allow trade.
            above_vwap = ind.get("above_vwap")   # 1=above, 0=below, None=unknown
            vwap_misaligned = (
                above_vwap is not None
                and (
                    (action == "BUY"  and above_vwap == 0)
                    or (action == "SELL" and above_vwap == 1)
                )
            )
            if vwap_misaligned:
                log.info(
                    f"{pair}: VWAP misaligned (above_vwap={above_vwap}, signal={action}) "
                    f"— suppressing auto-trade, sending alert only"
                )

            # Retail sentiment — block if crowd is already leaning with us
            # (crowded positioning = contra-indicator; contrarian field tells us the fade direction)
            retail_sentiment = data.get("retail_sentiment", {})
            retail_contrarian = retail_sentiment.get("contrarian", "HOLD")
            retail_crowded_against = retail_contrarian not in ("HOLD", action)
            if retail_crowded_against:
                log.info(
                    f"{pair}: retail crowded against {action} "
                    f"(bias={retail_sentiment.get('bias','?')}, contrarian={retail_contrarian}) "
                    f"— suppressing auto-trade, sending alert only"
                )

            # COT bias — block if large speculator positioning opposes signal.
            # Net speculator longs = BUY bias; net shorts = SELL bias; NEUTRAL = ignore.
            try:
                from forexmind.data.cot_fetcher import get_cot_bias as _get_cot_bias
                cot = _get_cot_bias(pair.replace("/", "_"))
            except Exception:
                cot = {}
            cot_direction = cot.get("direction", "NEUTRAL")
            cot_opposes = cot_direction not in ("NEUTRAL", action)
            if cot_opposes:
                log.info(
                    f"{pair}: COT opposes {action} "
                    f"(cot_dir={cot_direction}, net={cot.get('net_position', '?')}) "
                    f"— suppressing auto-trade, sending alert only"
                )

            if (
                confidence >= AUTO_TRADE_CONFIDENCE
                and agreeing_count >= 3
                and not has_caution
                and not low_liquidity
                and ema_aligned
                and not sl_too_tight
                and not is_ranging
                and rule_based_agrees
                and not near_event
                and not vwap_misaligned
                and not retail_crowded_against
                and not cot_opposes
            ):
                # Check concurrent trade cap before placing
                if open_count >= max_trades:
                    log.info(f"{pair}: auto-trade skipped — at max concurrent trades ({open_count}/{max_trades})")
                    footer = f"\n\n⏸ <b>Auto-trade paused</b> — {open_count}/{max_trades} trades open (limit reached).\n<i>Will auto-trade once existing positions close.</i>"
                    full_text = alert_text + footer
                    await _send_telegram(bot, chat_id, full_text)
                    recent_alerts[pair] = AlertRecord(pair=pair, action=action, alerted_at=now)
                    continue

                # HIGH CONFIDENCE — 3+ strategies agree, clean conditions — place trade automatically
                log.info(f"{pair}: confidence {confidence}% >= {AUTO_TRADE_CONFIDENCE}% with {agreeing_count}/4 strategies — auto-trading")
                success, trade_msg, trade_result = await _auto_place_trade(data)

                if success:
                    await _save_trade_to_db(signal_id, data, trade_result)
                    # Register split-TP so auto-trades get partial close at 2R + BE SL
                    _trade_id = trade_result.get("trade_id", "")
                    _filled = float(trade_result.get("filled_price", 0)) or float(sig.get("entry", 0))
                    _units = trade_result.get("units", 0)
                    # Use risk-manager SL (ATR-based) not signal SL
                    _sl = float(trade_result.get("actual_stop_loss", 0)) or float(sig.get("stop_loss", 0))
                    if _trade_id and _units and _sl:
                        try:
                            from forexmind.interfaces.telegram_bot import register_split_tp
                            register_split_tp(
                                trade_id=str(_trade_id),
                                instrument=pair.replace("/", "_"),
                                direction=action,
                                entry=_filled,
                                sl=_sl,
                                units=_units,
                                chat_id=int(chat_id),
                            )
                        except Exception as stp_err:
                            log.warning(f"Split-TP registration failed (non-fatal): {stp_err}")
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
                # Manual-only: below confidence/strategy threshold OR caution/liquidity conditions
                success, trade_msg, trade_result = False, "", {}
                block_reasons = []
                if low_liquidity:
                    block_reasons.append(f"low liquidity ({session_score:.0%})")
                if not ema_aligned:
                    block_reasons.append(f"EMA {ema_trend} opposes {action}")
                if sl_too_tight:
                    block_reasons.append(f"SL too tight ({sl_pips} pips)")
                if is_ranging:
                    block_reasons.append(f"ranging market (ADX {adx_val:.0f})")
                if not rule_based_agrees:
                    block_reasons.append(f"rule-based {rule_based_vote}")
                if near_event:
                    block_reasons.append(f"news event: {event_reason}")
                if has_caution:
                    if atr_ratio >= 2.0:
                        block_reasons.append(f"ATR {atr_ratio:.1f}x avg")
                    if (action == "BUY" and news_score <= -0.3) or (action == "SELL" and news_score >= 0.3):
                        block_reasons.append(f"news {news_score:+.2f} opposing")
                if vwap_misaligned:
                    block_reasons.append(f"price {'below' if action == 'BUY' else 'above'} VWAP")
                if retail_crowded_against:
                    block_reasons.append(f"retail {retail_sentiment.get('bias', 'crowded')} contra")
                if cot_opposes:
                    block_reasons.append(f"COT {cot_direction} vs {action}")
                if block_reasons:
                    footer = f"\n\n⚠️ <b>Auto-trade blocked</b> — {', '.join(block_reasons)}\n<i>Review and place manually if you agree.</i>"
                else:
                    footer = "\n\n👆 <i>Review and place manually if you agree.</i>"
                full_text = alert_text + footer
                log.info(f"✅ Alert sent: {action} {pair} ({confidence}% conf, {agreeing_count}/4 strategies, {rr} R:R, caution={has_caution})")

            await _send_telegram(bot, chat_id, full_text)
            recent_alerts[pair] = AlertRecord(pair=pair, action=action, alerted_at=now)

        except Exception as e:
            log.error(f"{pair}: scan error — {e}")

        # Small delay between pair requests to avoid rate limiting
        await asyncio.sleep(1)


async def _sync_closed_trades(bot, chat_id: str) -> None:
    """
    Compare OPEN trades in our DB against OANDA's open trades.
    Any DB-OPEN trade that no longer appears in OANDA was closed by SL/TP —
    fetch the realised P&L from OANDA and record it so /stats counts it correctly.
    """
    try:
        from forexmind.data.oanda_client import get_oanda_client
        from forexmind.data.trade_repo import close_trade_record
        from forexmind.data.database import get_session
        from forexmind.data.models import Trade, TradeStatus
        from forexmind.risk.manager import get_risk_manager
        from sqlalchemy import select

        client = get_oanda_client()

        # Trades OANDA currently reports as open
        oanda_open = await client.get_open_trades()
        oanda_open_ids = {str(t.get("id", "")) for t in oanda_open}

        # Always reconcile the risk manager's _open_trades with OANDA reality.
        # This prevents phantom trades from blocking correlation checks.
        await get_risk_manager().sync_open_trades(oanda_open_ids)

        # All trades our DB thinks are still open
        async with get_session() as session:
            result = await session.execute(
                select(Trade).where(Trade.status == TradeStatus.OPEN)
            )
            db_open = result.scalars().all()

        if not db_open:
            return

        # Trades our DB has open but OANDA does not — they were closed by SL/TP
        orphans = [t for t in db_open if t.oanda_trade_id and t.oanda_trade_id not in oanda_open_ids]
        if not orphans:
            return

        # Fetch OANDA's recently closed trades to get exit price and P&L
        recently_closed = await client.get_recently_closed_trades(count=100)
        closed_by_id = {str(t.get("id", "")): t for t in recently_closed}

        for trade in orphans:
            oanda_trade = closed_by_id.get(trade.oanda_trade_id)
            if not oanda_trade:
                log.warning(f"Orphan trade {trade.oanda_trade_id} not found in recent OANDA closes — skipping")
                continue

            exit_price = float(oanda_trade.get("averageClosePrice", 0) or 0)
            realised_pnl = float(oanda_trade.get("realizedPL", 0) or 0)

            # Use OANDA's actual close time, not datetime.now()
            oanda_close_time = None
            if oanda_trade.get("closeTime"):
                try:
                    from datetime import datetime as _dt, timezone as _tz
                    _raw = oanda_trade["closeTime"].split(".")[0] + "+00:00"
                    oanda_close_time = _dt.fromisoformat(_raw)
                except Exception:
                    pass

            newly_closed = await close_trade_record(trade.oanda_trade_id, exit_price, realised_pnl, oanda_close_time)
            if not newly_closed:
                # Already recorded by a concurrent sync (e.g. monitor job or second instance)
                log.debug(f"Sync: {trade.oanda_trade_id} already closed in DB — skipping duplicate notification")
                continue

            await get_risk_manager().close_trade(trade.oanda_trade_id, exit_price)

            pair = trade.instrument.replace("_", "/")
            result_icon = "✅ WIN" if realised_pnl > 0 else "❌ LOSS (SL hit)"
            log.info(f"Sync: {pair} #{trade.oanda_trade_id} closed — {result_icon} ${realised_pnl:+.2f}")

            await _send_telegram(
                bot, chat_id,
                f"{'✅' if realised_pnl > 0 else '❌'} <b>{result_icon}  ·  {pair}</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Exit @ <code>{exit_price}</code>  ·  P&amp;L: <b>${realised_pnl:+,.2f}</b>\n"
                f"<i>Closed by OANDA (SL/TP) — recorded in stats.</i>"
            )

    except Exception as e:
        log.error(f"_sync_closed_trades error: {e}")


# Hour (UTC) at which daily auto-retrain runs — 05:00 UTC (before London open)
RETRAIN_HOUR_UTC = 5

# ── Thesis monitor constants ──────────────────────────────────────────────────
THESIS_CHECK_INTERVAL = 300   # Re-signal every 5 minutes
THESIS_FLIP_CONFIDENCE = 60.0 # Opposite signal must be ≥60% to trigger auto-exit
THESIS_TIME_STOP_MIN = 90     # Warn after 90 minutes of a losing trade

# Time-based profit capture — extended to let winners run
PROFIT_STALL_MIN    = 180   # Alert if barely moving (0 < R < 0.5) after 3h
PROFIT_CAPTURE_MIN  = 360   # "Take profits" alert at 6h if TP not yet hit
PROFIT_HARD_MAX_MIN = 480   # Auto-close any winning trade at 8h


def _r_multiple(direction: str, entry: float, sl: float, current: float) -> float:
    """
    Compute how many R-multiples of profit/loss the trade has achieved.
    Positive = profit, negative = loss. Zero if prices are invalid.
    """
    if entry <= 0 or sl <= 0 or current <= 0:
        return 0.0
    risk = abs(entry - sl)
    if risk == 0:
        return 0.0
    return (current - entry) / risk if direction == "BUY" else (entry - current) / risk


async def _thesis_monitor_loop(bot, chat_id: str) -> None:
    """
    Continuous background loop — independent of the 15-min signal scanner.

    Every 5 minutes, for each open trade:

    LOSING trades:
      1. Re-runs the signal (M15). If model flips with ≥60% confidence → auto-exit.
      2. If losing for ≥90 minutes → one-time warning so user can cut manually.

    WINNING trades (time-profit capture rules):
      1. 90 min, R < 0.5     → "Stalling" alert (capture partial or close)
      2. 90 min, 0.5 ≤ R < 1.5 → "Lock gains" alert (move SL to breakeven)
      3. 3h,  R > 0.1        → "3-hour exit window" alert (TP window missed)
      4. 4h,  R > 0          → Auto-close (M15 max hold time, protect gains)
      5. 20:30 UTC, R > 0    → "NY session closing" alert
    """
    time_stop_alerted: set[str] = set()          # trade IDs already warned (time-stop)
    profit_milestones: dict[str, set] = {}        # trade_id → set of fired milestone keys
    await asyncio.sleep(90)                       # brief delay so startup noise settles

    while True:
        try:
            from forexmind.agents.tools import _get_signal
            from forexmind.data.oanda_client import get_oanda_client
            from forexmind.data.trade_repo import close_trade_record
            from forexmind.risk.manager import get_risk_manager
            from forexmind.interfaces.telegram_bot import (
                _split_tp_trades, _save_monitors,
            )

            client = get_oanda_client()
            open_trades = await client.get_open_trades()

            if not open_trades:
                time_stop_alerted.clear()
                profit_milestones.clear()
                await asyncio.sleep(THESIS_CHECK_INTERVAL)
                continue

            now = datetime.now(UTC)
            open_ids = {str(t.get("id", "")) for t in open_trades}
            # Remove stale state for trades that are already closed
            time_stop_alerted &= open_ids
            for closed_id in list(profit_milestones):
                if closed_id not in open_ids:
                    profit_milestones.pop(closed_id, None)

            # Batch fetch current mid prices for R-multiple computation (one call per instrument)
            _instruments = list({t.get("instrument", "") for t in open_trades if t.get("instrument")})
            _current_prices: dict[str, float] = {}
            for _inst in _instruments:
                try:
                    _p = await client.get_price(_inst)
                    _current_prices[_inst] = _p.mid
                except Exception:
                    pass

            for trade in open_trades:
                trade_id      = str(trade.get("id", ""))
                instrument    = trade.get("instrument", "")
                pnl           = float(trade.get("unrealizedPL", 0))
                units         = int(float(trade.get("currentUnits", 0)))
                direction     = "BUY" if units > 0 else "SELL"
                open_time_str = trade.get("openTime", "")
                pair          = instrument.replace("_", "/")

                # Compute trade age (used by both winning and losing branches)
                age_min = 0.0
                if open_time_str:
                    try:
                        clean = open_time_str.split(".")[0] + "+00:00"
                        open_dt = datetime.fromisoformat(clean)
                        age_min = (now - open_dt).total_seconds() / 60
                    except Exception:
                        pass

                # ── WINNING TRADE: trailing stop + time-profit capture ────────
                if pnl > 0:
                    current_price = _current_prices.get(instrument)
                    entry_price   = float(trade.get("price", 0))
                    sl_price      = float((trade.get("stopLossOrder") or {}).get("price", 0))
                    r_mult = (
                        _r_multiple(direction, entry_price, sl_price, current_price)
                        if current_price else None
                    )
                    m = profit_milestones.setdefault(trade_id, set())

                    # ── Trailing stop: ratchet SL up as price moves in our favour ──
                    if current_price:
                        try:
                            rm = get_risk_manager()
                            # Estimate live ATR from recent price movement (or use last known)
                            _atr = abs(entry_price - sl_price) / max(rm._cfg.atr_stop_multiplier, 1.0)
                            new_sl = await rm.update_trailing_stop(trade_id, current_price, _atr)
                            if new_sl is not None:
                                moved = await client.modify_trade_sl(trade_id, new_sl, instrument)
                                if moved:
                                    sl_price = new_sl   # update for R-mult recalc below
                                    r_mult = _r_multiple(direction, entry_price, sl_price, current_price)
                                    if "trail" not in m:
                                        m.add("trail")
                                        await _send_telegram(bot, chat_id,
                                            f"📈 <b>Trailing Stop Moved  ·  {pair}</b>\n"
                                            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                                            f"Direction  {direction}\n"
                                            f"New SL     <code>{new_sl:.5f}</code>\n"
                                            f"Profit     <b>{r_mult:.2f}R  (${pnl:+,.2f})</b>\n\n"
                                            f"<i>SL trailing price — locking in gains.</i>"
                                        )
                        except Exception as trail_err:
                            log.debug(f"Trailing stop update failed for {trade_id}: {trail_err}")

                    # ── News proximity exit (Fix 7b) ──────────────────────────
                    # If a high-impact event is within 30 min and the trade is
                    # at least 30% of the way to TP, close now to protect gains.
                    if current_price and r_mult is not None and r_mult > 0 and "news_exit" not in m:
                        try:
                            _cal_events = await _fetch_calendar()
                            _near_news, _news_reason = _is_near_high_impact_event(
                                instrument, now, _cal_events, window_minutes=_NEWS_BLOCK_MINUTES
                            )
                            if _near_news:
                                tp_price = float((trade.get("takeProfitOrder") or {}).get("price", 0))
                                if tp_price > 0 and entry_price > 0:
                                    tp_dist = abs(tp_price - entry_price)
                                    tp_progress = abs(current_price - entry_price) / tp_dist if tp_dist else 0
                                else:
                                    tp_progress = 0
                                if tp_progress >= 0.30:
                                    m.add("news_exit")
                                    result = await client.close_trade(trade_id)
                                    if result.success:
                                        filled = result.filled_price or current_price
                                        await get_risk_manager().close_trade(trade_id, filled)
                                        from forexmind.data.trade_repo import close_trade_record
                                        await close_trade_record(trade_id, filled, pnl)
                                        profit_milestones.pop(trade_id, None)
                                        time_stop_alerted.discard(trade_id)
                                        await _send_telegram(bot, chat_id,
                                            f"📰 <b>News Exit  ·  {pair}</b>\n"
                                            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                                            f"Direction  {direction}\n"
                                            f"News       {_news_reason}\n"
                                            f"Progress   {tp_progress:.0%} to TP\n"
                                            f"Profit     <b>{r_mult:.2f}R  (${pnl:+,.2f})</b>\n\n"
                                            f"<i>Closed before high-impact event.</i>"
                                        )
                                        log.info(f"News exit: {trade_id} {instrument} {tp_progress:.0%} to TP — {_news_reason}")
                                        continue
                                else:
                                    # Not enough progress to close — tighten TP to lock in gains
                                    if tp_price > 0 and r_mult >= 0.10:
                                        await client.modify_trade_tp(trade_id, current_price, instrument)
                                        await _send_telegram(bot, chat_id,
                                            f"⚠️ <b>TP Tightened — News  ·  {pair}</b>\n"
                                            f"News: {_news_reason}\n"
                                            f"TP moved to <code>{current_price:.5f}</code>\n"
                                            f"<i>Locking in {r_mult:.2f}R before event.</i>"
                                        )
                        except Exception as _ne:
                            log.debug(f"News exit check failed for {trade_id}: {_ne}")

                    # ── Momentum decay detection (Fix 8) ─────────────────────
                    # Exit winning trades early if at least 2 of 3 momentum
                    # indicators flip against the trade direction. Only fires
                    # once per trade and only when price is already in profit.
                    if (
                        current_price
                        and r_mult is not None and 0 < r_mult < 1.5
                        and "momentum_decay" not in m
                    ):
                        try:
                            import pandas as pd
                            _candles = await client.get_candles(instrument, "M15", count=12)
                            if _candles and len(_candles) >= 5:
                                _df = pd.DataFrame(_candles)
                                if "close" in _df.columns:
                                    import pandas_ta as _ta
                                    _close = _df["close"].astype(float)
                                    _rsi = _ta.rsi(_close, length=14)
                                    _macd_df = _ta.macd(_close)
                                    _ema8  = _ta.ema(_close, length=8)
                                    _ema21 = _ta.ema(_close, length=21)
                                    decay = 0
                                    if _rsi is not None and len(_rsi) >= 2:
                                        if direction == "BUY"  and _rsi.iloc[-1] < 50 and _rsi.iloc[-2] >= 50:
                                            decay += 1
                                        if direction == "SELL" and _rsi.iloc[-1] > 50 and _rsi.iloc[-2] <= 50:
                                            decay += 1
                                    if _macd_df is not None and not _macd_df.empty:
                                        _mcol = [c for c in _macd_df.columns if "MACD_" in c and "h" not in c.lower() and "s" not in c.lower()]
                                        _scol = [c for c in _macd_df.columns if "MACDs" in c]
                                        if _mcol and _scol:
                                            _m = _macd_df[_mcol[0]]
                                            _s = _macd_df[_scol[0]]
                                            if direction == "BUY"  and _m.iloc[-1] < _s.iloc[-1] and _m.iloc[-2] >= _s.iloc[-2]:
                                                decay += 1
                                            if direction == "SELL" and _m.iloc[-1] > _s.iloc[-1] and _m.iloc[-2] <= _s.iloc[-2]:
                                                decay += 1
                                    if _ema8 is not None and _ema21 is not None:
                                        if direction == "BUY"  and _ema8.iloc[-1] < _ema21.iloc[-1]:
                                            decay += 1
                                        if direction == "SELL" and _ema8.iloc[-1] > _ema21.iloc[-1]:
                                            decay += 1
                                    if decay >= 2:
                                        m.add("momentum_decay")
                                        result = await client.close_trade(trade_id)
                                        if result.success:
                                            filled = result.filled_price or current_price
                                            await get_risk_manager().close_trade(trade_id, filled)
                                            from forexmind.data.trade_repo import close_trade_record
                                            await close_trade_record(trade_id, filled, pnl)
                                            profit_milestones.pop(trade_id, None)
                                            time_stop_alerted.discard(trade_id)
                                            await _send_telegram(bot, chat_id,
                                                f"⚠️ <b>Momentum Exit  ·  {pair}</b>\n"
                                                f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                                                f"Direction  {direction}\n"
                                                f"Signals    {decay}/3 indicators reversed\n"
                                                f"Profit     <b>{r_mult:.2f}R  (${pnl:+,.2f})</b>\n\n"
                                                f"<i>Momentum reversed — exited to protect gains.</i>"
                                            )
                                            log.info(f"Momentum decay exit: {trade_id} {instrument} {decay}/3 signals r={r_mult:.2f}")
                                            continue
                        except Exception as _me:
                            log.debug(f"Momentum decay check failed for {trade_id}: {_me}")

                    # ── Momentum hold: skip time-based exits if trend is strong ──
                    # If R > 1.5 and the trade is running well, let it ride to TP
                    momentum_hold = r_mult is not None and r_mult >= 1.5

                    # Rule 1 — Stalling: barely profitable after 3h
                    if (age_min >= PROFIT_STALL_MIN
                            and r_mult is not None and 0 < r_mult < 0.5
                            and "stall" not in m):
                        m.add("stall")
                        await _send_telegram(bot, chat_id,
                            f"🐌 <b>Trade Stalling  ·  {pair}</b>\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"Direction  {direction}\n"
                            f"Age        {age_min:.0f} min\n"
                            f"Profit     <b>{r_mult:.2f}R  (${pnl:+,.2f})</b>\n\n"
                            f"<i>Only {r_mult:.2f}R after {age_min:.0f} min — TP unlikely.\n"
                            f"Consider closing to capture partial gains.</i>"
                        )
                        log.info(f"Profit stall alert: {trade_id} {instrument} r={r_mult:.2f} age={age_min:.0f}min")

                    # Rule 2 — Breakeven lock: decent profit, protect it
                    if (age_min >= PROFIT_STALL_MIN
                            and r_mult is not None and 0.5 <= r_mult < 1.5
                            and "be_lock" not in m):
                        m.add("be_lock")
                        await _send_telegram(bot, chat_id,
                            f"🔒 <b>Lock Gains  ·  {pair}</b>\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"Direction  {direction}\n"
                            f"Age        {age_min:.0f} min\n"
                            f"Profit     <b>{r_mult:.2f}R  (${pnl:+,.2f})</b>\n\n"
                            f"<i>At {r_mult:.2f}R — move SL to breakeven if not already.\n"
                            f"Let remainder ride to the 2R TP.</i>"
                        )
                        log.info(f"BE lock alert: {trade_id} {instrument} r={r_mult:.2f} age={age_min:.0f}min")

                    # Rule 3 — 6-hour exit window: TP hasn't hit, momentum fading
                    # Skip if momentum_hold — trade is running strong, let it ride
                    if (age_min >= PROFIT_CAPTURE_MIN
                            and r_mult is not None and r_mult > 0.1
                            and not momentum_hold
                            and "6h" not in m):
                        m.add("6h")
                        await _send_telegram(bot, chat_id,
                            f"⏰ <b>6-Hour Exit Window  ·  {pair}</b>\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"Direction  {direction}\n"
                            f"Age        {age_min:.0f} min\n"
                            f"Profit     <b>{r_mult:.2f}R  (${pnl:+,.2f})</b>\n\n"
                            f"<i>TP not reached after 6 hours.\n"
                            f"Consider capturing {r_mult:.1f}R or tightening SL.</i>"
                        )
                        log.info(f"6h exit alert: {trade_id} {instrument} r={r_mult:.2f}")

                    # Rule 4 — 8-hour hard exit: auto-close, protect gains
                    # Skip if momentum_hold — trailing stop will protect, let it run to TP
                    if (age_min >= PROFIT_HARD_MAX_MIN
                            and r_mult is not None and r_mult > 0
                            and not momentum_hold
                            and "8h_close" not in m):
                        m.add("8h_close")
                        result = await client.close_trade(trade_id)
                        if result.success:
                            filled = result.filled_price or 0.0
                            await get_risk_manager().close_trade(trade_id, filled)
                            await close_trade_record(trade_id, filled, pnl)
                            _split_tp_trades.pop(trade_id, None)
                            _save_monitors()
                            profit_milestones.pop(trade_id, None)
                            time_stop_alerted.discard(trade_id)
                            await _send_telegram(bot, chat_id,
                                f"⏱ <b>8-Hour Time Exit  ·  {pair}</b>\n"
                                f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                                f"Direction  {direction}\n"
                                f"Age        {age_min:.0f} min\n"
                                f"Profit     <b>{r_mult:.2f}R  (${pnl:+,.2f})</b>\n"
                                f"Exit       <code>{filled}</code>\n\n"
                                f"<i>TP not reached in 8h — auto-closed to protect {r_mult:.1f}R.</i>"
                            )
                            log.info(f"8h auto-exit: {trade_id} {instrument} r={r_mult:.2f} pnl=${pnl:+.2f}")
                        continue  # trade closed — skip remaining checks

                    # Rule 4b — Momentum hold info: let user know trade is running strong
                    if (age_min >= PROFIT_HARD_MAX_MIN
                            and momentum_hold
                            and "hold_info" not in m):
                        m.add("hold_info")
                        await _send_telegram(bot, chat_id,
                            f"💪 <b>Momentum Hold  ·  {pair}</b>\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"Direction  {direction}\n"
                            f"Age        {age_min:.0f} min\n"
                            f"Profit     <b>{r_mult:.2f}R  (${pnl:+,.2f})</b>\n\n"
                            f"<i>Trade running strong at {r_mult:.1f}R — holding for TP.\n"
                            f"Trailing stop protects gains at SL <code>{sl_price:.5f}</code></i>"
                        )
                        log.info(f"Momentum hold: {trade_id} {instrument} r={r_mult:.2f} — skipping time exit")

                    # Rule 5 — Session end: 30 min before NY close (20:30–21:00 UTC)
                    now_t = now.time()
                    if (time(20, 30) <= now_t < time(21, 0)
                            and r_mult is not None and r_mult > 0
                            and "session_end" not in m):
                        m.add("session_end")
                        await _send_telegram(bot, chat_id,
                            f"🌙 <b>Session Close Alert  ·  {pair}</b>\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"Direction  {direction}\n"
                            f"Profit     <b>{r_mult:.2f}R  (${pnl:+,.2f})</b>\n\n"
                            f"<i>NY session closes at 21:00 UTC (~30 min).\n"
                            f"Consider closing to avoid post-session liquidity drop.</i>"
                        )
                        log.info(f"Session end alert: {trade_id} {instrument} r={r_mult:.2f}")

                    continue   # winning trade — skip model-flip (that's for losers only)

                # ── LOSING TRADE: time-stop + model flip ──────────────────────
                if pnl >= 0:
                    continue   # break-even — no action needed

                # Time-stop warning (one-time per trade)
                if age_min >= THESIS_TIME_STOP_MIN and trade_id not in time_stop_alerted:
                    time_stop_alerted.add(trade_id)
                    await _send_telegram(
                        bot, chat_id,
                        f"⏰ <b>Time-Stop Warning  ·  {pair}</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"Direction  {direction}\n"
                        f"Age        {age_min:.0f} min\n"
                        f"P&amp;L       <b>${pnl:+,.2f}</b>\n\n"
                        f"<i>Trade has been losing for {age_min:.0f} min — "
                        f"consider cutting manually.</i>",
                    )
                    log.info(f"Thesis: time-stop alert sent for {trade_id} {instrument} ({age_min:.0f} min, ${pnl:+.2f})")

                # Model-flip check → auto-exit
                try:
                    result_str = await _get_signal(instrument, "M15", 300)
                    data = json.loads(result_str)
                    if "error" in data:
                        continue

                    sig           = data.get("signal", {})
                    signal_action = sig.get("action", "HOLD")
                    confidence    = float(sig.get("confidence", 0))
                    opposite      = {"BUY": "SELL", "SELL": "BUY"}

                    log.info(
                        f"Thesis check {instrument}: trade={direction} "
                        f"signal={signal_action} conf={confidence:.1f}%"
                    )

                    if (
                        signal_action == opposite.get(direction)
                        and confidence >= THESIS_FLIP_CONFIDENCE
                    ):
                        result = await client.close_trade(trade_id)
                        if result.success:
                            filled = result.filled_price or 0.0
                            await get_risk_manager().close_trade(trade_id, filled)
                            await close_trade_record(trade_id, filled, pnl)
                            _split_tp_trades.pop(trade_id, None)
                            _save_monitors()
                            profit_milestones.pop(trade_id, None)
                            time_stop_alerted.discard(trade_id)
                            await _send_telegram(
                                bot, chat_id,
                                f"🔄 <b>Thesis Flip — Early Exit  ·  {pair}</b>\n"
                                f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                                f"Was        {direction}\n"
                                f"New signal <b>{signal_action}</b> @ {confidence:.0f}% confidence\n"
                                f"Exit       <code>{filled}</code>\n"
                                f"P&amp;L       <b>${pnl:+,.2f}</b>\n\n"
                                f"<i>Model reversed — closed before SL.</i>",
                            )
                            log.info(
                                f"Thesis flip exit: {trade_id} {instrument} "
                                f"{direction}→{signal_action} conf={confidence:.0f}% pnl=${pnl:+.2f}"
                            )
                except Exception as e:
                    log.warning(f"Thesis signal check failed for {instrument}: {e}")

        except Exception as e:
            log.error(f"Thesis monitor loop error: {e}")

        await asyncio.sleep(THESIS_CHECK_INTERVAL)


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

        # ── Fetch multi-pair training data ────────────────────────────────────
        # Pull H1 candles for each configured pair, add indicators, concatenate.
        # 10000 H1 bars ≈ 14 months — enough for multiple market regimes and
        # robust walk-forward CV without hitting OANDA's history limits.
        TRAIN_PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "XAU_USD"]
        # Desired number of training bars per instrument. We use paginated
        # requests below to assemble this many bars from OANDA (which has a
        # per-request 'count' maximum of 5000).
        TRAIN_BARS  = 10000
        # OANDA per-request maximum
        MAX_OANDA_COUNT = 5000
        TRAIN_TF    = "H1"

        def _fetch_df() -> "pd.DataFrame":
            # Call oandapyV20 synchronously — avoids asyncio.run() nesting issues
            # when running inside a ThreadPoolExecutor thread.
            import oandapyV20
            import oandapyV20.endpoints.instruments as _instruments
            import pandas as pd
            from forexmind.config.settings import get_settings
            from forexmind.indicators.engine import IndicatorEngine

            cfg = get_settings().oanda
            api = oandapyV20.API(
                access_token=cfg.api_key,
                environment=cfg.environment,
            )
            engine = IndicatorEngine()
            frames = []
            for pair in TRAIN_PAIRS:
                try:
                    # Paginated fetch: request up to MAX_OANDA_COUNT bars per
                    # call and iterate until we have TRAIN_BARS or no more data.
                    total_rows = []
                    to_time = None
                    while len(total_rows) < TRAIN_BARS:
                        need = min(MAX_OANDA_COUNT, TRAIN_BARS - len(total_rows))
                        params = {
                            "granularity": TRAIN_TF,
                            "price": "M",
                            "count": str(need),
                        }
                        if to_time is not None:
                            params["to"] = to_time

                        req = _instruments.InstrumentsCandles(pair, params=params)
                        data = api.request(req)
                        batch = []
                        for candle in data.get("candles", []):
                            if not candle.get("complete", True):
                                continue
                            mid = candle["mid"]
                            batch.append({
                                "time": pd.Timestamp(candle["time"]).tz_convert("UTC"),
                                "open": float(mid["o"]),
                                "high": float(mid["h"]),
                                "low": float(mid["l"]),
                                "close": float(mid["c"]),
                                "volume": int(candle.get("volume", 0)),
                            })

                        if not batch:
                            # No more data available for this pair
                            break

                        total_rows.extend(batch)

                        # Prepare 'to' param for next (older) page: use earliest
                        # candle time from this batch and step one microsecond.
                        earliest = pd.Timestamp(batch[0]["time"])
                        to_time = (earliest - pd.Timedelta(microseconds=1)).isoformat()

                        # Safety: stop if OANDA returned fewer than requested
                        if len(batch) < need:
                            break

                    if not total_rows:
                        log.warning(f"Retrain: no candles returned for {pair}")
                        continue

                    # Build DataFrame, sort chronologically and keep most recent
                    raw = (
                        pd.DataFrame(total_rows)
                        .set_index("time")
                        .sort_index()
                        .iloc[-TRAIN_BARS:]
                    )
                    ind_df = engine.compute(raw)
                    ind_df["instrument"] = pair
                    frames.append(ind_df)
                    log.info(f"Retrain: fetched {len(ind_df)} bars for {pair} (requested {TRAIN_BARS})")
                except Exception as e:
                    log.warning(f"Retrain: skipping {pair} — {e}")
            if not frames:
                raise RuntimeError("No training data fetched — all pairs failed")
            # Preserve DatetimeIndex — build_feature_matrix needs it for
            # add_session_flags (which calls ts.to_pydatetime() on each index entry).
            # ignore_index=True would reset to RangeIndex and break that call.
            return pd.concat(frames)

        try:
            train_df = _fetch_df()
            log.info(f"Retrain: total training rows = {len(train_df)}")
        except Exception as e:
            msg = f"FAILED to fetch training data: {e}"
            log.error(msg)
            return {"lightgbm": msg, "lstm": msg}

        # ── Train LightGBM ────────────────────────────────────────────────────
        try:
            from forexmind.strategy.ml_strategy import LightGBMStrategy
            from forexmind.strategy.feature_engineering import build_feature_matrix
            lgbm = LightGBMStrategy()

            # ── Pre-compute full feature matrix once (shared by all session splits)
            feat_df = build_feature_matrix(train_df, add_target=True)

            # ── Combined model (all sessions) ─────────────────────────────────
            lgbm_result = lgbm.train(feat_df)
            cv_acc = lgbm_result.get("cv_mean_accuracy", 0)
            ho_acc = lgbm_result.get("holdout_accuracy", 0)
            results["lightgbm"] = f"CV={cv_acc:.1%} | Holdout={ho_acc:.1%}"

            # ── Session-specific models ───────────────────────────────────────
            # London: 07:00–12:00 UTC  |  NY: 12:00–21:00 UTC
            # Use session_london / session_ny flags already in the feature matrix.
            for session_name, flag_col, lo_h, hi_h in [
                ("london", "session_london", 7, 12),
                ("ny",     "session_ny",     12, 21),
            ]:
                try:
                    if flag_col in feat_df.columns:
                        sess_df = feat_df[feat_df[flag_col] == 1].copy()
                    else:
                        # Fallback: filter by UTC hour range
                        if isinstance(feat_df.index, pd.DatetimeIndex):
                            hours = feat_df.index.hour
                            sess_df = feat_df[(hours >= lo_h) & (hours < hi_h)].copy()
                        else:
                            sess_df = pd.DataFrame()
                    if len(sess_df) < 500:
                        log.info(f"Retrain: skipping {session_name} model — only {len(sess_df)} rows")
                        continue
                    sess_result = lgbm.train(sess_df)
                    lgbm.save_session_model(lgbm._model, session_name)
                    s_cv = sess_result.get("cv_mean_accuracy", 0)
                    s_ho = sess_result.get("holdout_accuracy", 0)
                    results[f"lgbm_{session_name}"] = f"CV={s_cv:.1%} | Holdout={s_ho:.1%}"
                    log.info(f"LightGBM {session_name} model trained: CV={s_cv:.1%} Holdout={s_ho:.1%}")
                except Exception as e:
                    log.warning(f"LightGBM {session_name} retrain failed (non-fatal): {e}")
        except Exception as e:
            results["lightgbm"] = f"FAILED: {e}"
            log.error(f"LightGBM retrain error: {e}")

        # ── Train LSTM ────────────────────────────────────────────────────────
        try:
            from forexmind.strategy.ml_strategy import LSTMStrategy
            lstm = LSTMStrategy()
            lstm_result = lstm.train(train_df)
            val_acc = lstm_result.get("accuracy", lstm_result.get("best_val_accuracy", lstm_result.get("val_accuracy", 0)))
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

    from telegram.request import HTTPXRequest
    bot = Bot(
        token=cfg.telegram.bot_token,
        request=HTTPXRequest(connect_timeout=10, read_timeout=15),
    )
    await bot.initialize()   # PTB v20+ requires explicit init before first send
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

    # Seed RiskManager._open_trades from OANDA at startup so correlation checks
    # work even for trades that were open before this process started.
    try:
        from forexmind.data.oanda_client import get_oanda_client
        from forexmind.risk.manager import get_risk_manager, OpenTrade
        _client = get_oanda_client()
        _oanda_open = await _client.get_open_trades()
        _rm = get_risk_manager()
        for _t in _oanda_open:
            _tid = str(_t.get("id", ""))
            _inst = _t.get("instrument", "")
            _units = int(float(_t.get("currentUnits", _t.get("initialUnits", 0))))
            _direction = "BUY" if _units > 0 else "SELL"
            _entry = float(_t.get("price", 0))
            _sl = float((_t.get("stopLossOrder") or {}).get("price", 0))
            _tp = float((_t.get("takeProfitOrder") or {}).get("price", 0))
            if _tid and _inst:
                await _rm.register_trade(OpenTrade(
                    trade_id=_tid,
                    instrument=_inst,
                    direction=_direction,
                    entry_price=_entry,
                    stop_loss=_sl,
                    take_profit=_tp,
                    units=abs(_units),
                ))
        if _oanda_open:
            log.info(f"Seeded {len(_oanda_open)} open trade(s) into RiskManager from OANDA")
        # Restore daily P&L and win/loss counters from DB so /stats is
        # accurate immediately after a restart (not just after the first close).
        await _rm.load_stats_from_db()
        # Seed split-TP state so the 2R exit logic survives restarts
        from forexmind.interfaces.telegram_bot import seed_split_tp_from_oanda
        seed_split_tp_from_oanda(_oanda_open, chat_id)
    except Exception as _e:
        log.warning(f"Startup OANDA seed failed (non-fatal): {_e}")

    # Launch thesis monitor as a background task — runs independently of the
    # 15-min scanner, checking every 5 min for model flips and time-stops.
    asyncio.create_task(_thesis_monitor_loop(bot, chat_id))
    log.info("Thesis monitor started (5-min interval)")

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
