"""
ForexMind — Telegram Bot Interface
=====================================
Full-featured bot with real-time signals, auto-trade confirmation flow,
account monitoring, and honest confidence display.

Commands:
  /start          — Welcome + quick-start guide
  /signal EUR/USD — Live signal with entry/SL/TP
  /signals        — Top signals for current session
  /trade EUR/USD  — Signal + one-tap trade confirmation
  /autotrade on   — Enable auto-trading (uses scheduler thresholds)
  /autotrade off  — Disable auto-trading
  /trades         — Show open positions from OANDA
  /stats          — Win rate, P&L, trade count
  /account        — Full account snapshot
  /sessions       — Market session status
  /backtest EUR/USD — Quick 1-year backtest
  /help           — List commands
  [any text]      — Chat with Claude agent
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from forexmind.utils.logger import get_logger
from forexmind.utils.session_times import get_session_status, best_pairs_for_session
from forexmind.config.settings import get_settings

log = get_logger(__name__)

try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import (
        Application,
        CommandHandler,
        MessageHandler,
        CallbackQueryHandler,
        ContextTypes,
        filters,
    )
    from telegram.constants import ParseMode
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    log.warning("python-telegram-bot not installed. Run: pip install python-telegram-bot")


# ── Confidence helpers ────────────────────────────────────────────────────────

def confidence_label(conf_pct: float) -> str:
    """
    Return a calibrated label for a confidence percentage.

    Ensemble scores are structurally diluted across 4 weighted strategies
    (weights sum to 1.0), so realistic achievable ranges are:
      2 strategies agreeing → ~42–55%
      3 strategies agreeing → ~55–70%
      4 strategies agreeing → ~70–85%
    """
    if conf_pct >= 68:
        return "🔥 STRONG"
    elif conf_pct >= 55:
        return "✅ MODERATE"
    elif conf_pct >= 42:
        return "⚠️ WEAK"
    else:
        return "❌ LOW"


def confidence_note(conf_pct: float) -> str:
    """One-line context that sets realistic expectations."""
    if conf_pct >= 68:
        return "3–4 strategies in agreement — highest quality signal"
    elif conf_pct >= 55:
        return "Majority strategy agreement — tradeable with discipline"
    elif conf_pct >= 42:
        return "2 strategies agree — valid signal, size down"
    else:
        return "Below edge threshold — do not trade"


# ── Message formatters ────────────────────────────────────────────────────────

def format_signal_message(data: dict, include_trade_note: bool = False) -> str:
    """Format a signal dict into a Telegram HTML message."""
    sig = data.get("signal", {})
    ind = data.get("indicators", {})
    session = data.get("session", {})
    sentiment = data.get("news_sentiment", {})

    pair = data.get("instrument", "").replace("_", "/")
    action = sig.get("action", "HOLD")
    emoji = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "⚪"
    conf_pct = float(sig.get("confidence", 0))
    clabel = confidence_label(conf_pct)
    cnote = confidence_note(conf_pct)

    session_names = session.get("active", session.get("active_sessions", []))
    session_str = ", ".join(session_names) if session_names else "None"
    overlap_str = "🔥 YES — prime liquidity" if session.get("is_overlap") else "No"

    sentiment_score = float(sentiment.get("score", 0))
    sentiment_impact = sentiment.get("impact", "neutral").upper()
    news_str = f"{sentiment_impact} ({sentiment_score:+.3f})"

    rr = sig.get("risk_reward", "-")
    agreeing = sig.get("agreeing_strategies", "-")

    msg = (
        f"{emoji} <b>{action} {pair}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>Signal</b>\n"
        f"Entry:       <code>{sig.get('entry', '-')}</code>\n"
        f"Stop Loss:   <code>{sig.get('stop_loss', '-')}</code> "
        f"({sig.get('stop_loss_pips', '-')} pips)\n"
        f"Take Profit: <code>{sig.get('take_profit', '-')}</code> "
        f"({sig.get('take_profit_pips', '-')} pips)\n"
        f"R:R Ratio:   <b>{rr}:1</b>\n"
        f"Risk %:      {sig.get('risk_pct', '-')}%\n"
        f"\n"
        f"🎯 <b>Confidence: {conf_pct:.0f}%</b>  {clabel}\n"
        f"<i>{cnote}</i>\n"
        f"Strategies:  {agreeing} in agreement\n"
        f"\n"
        f"📈 <b>Indicators</b>\n"
        f"RSI:  {ind.get('rsi', '-')} [{ind.get('rsi_zone', '-')}]\n"
        f"MACD: {ind.get('macd_cross', '-')} | ADX: {ind.get('adx', '-')} "
        f"[{ind.get('adx_trend_strength', '-')}]\n"
        f"EMA:  {ind.get('ema_trend', '-')} | PSAR: {ind.get('psar_signal', '-')}\n"
        f"\n"
        f"🌍 <b>Context</b>\n"
        f"Sessions: {session_str}\n"
        f"Overlap:  {overlap_str}\n"
        f"News:     {news_str}"
    )

    if include_trade_note and action in ("BUY", "SELL"):
        msg += (
            f"\n\n"
            f"<i>Tap ✅ Confirm to place this trade on your OANDA "
            f"{'paper' if get_settings().app.paper_trading else 'LIVE'} account.</i>"
        )

    return msg.strip()


def format_account_message(data: dict) -> str:
    pnl = float(data.get("unrealized_pnl", 0))
    daily_pnl = float(data.get("daily_pnl_usd", 0))
    pnl_emoji = "📈" if pnl >= 0 else "📉"
    daily_emoji = "✅" if daily_pnl >= 0 else "⚠️"
    kill = "🔴 ACTIVE" if data.get("kill_switch_active") else "🟢 OK"
    mode = "🧪 Paper" if get_settings().app.paper_trading else "⚠️ LIVE"
    return (
        f"💼 <b>OANDA Account</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Balance:        <b>${float(data.get('balance', 0)):,.2f}</b>\n"
        f"NAV:            ${float(data.get('nav', 0)):,.2f}\n"
        f"{pnl_emoji} Unrealised P&L: ${pnl:,.2f}\n"
        f"{daily_emoji} Today's P&L:    ${daily_pnl:,.2f}\n"
        f"Open Trades:    {data.get('open_trade_count', 0)}\n"
        f"Margin Used:    ${float(data.get('margin_used', 0)):,.2f}\n"
        f"Kill Switch:    {kill}\n"
        f"Mode:           {mode}"
    ).strip()


def format_stats_message(stats: dict, trade_stats: dict) -> str:
    total = trade_stats.get("total_closed", 0)
    wr = trade_stats.get("win_rate")
    # Avoid raw < character — breaks Telegram HTML parser
    wr_str = f"{wr:.1%}" if wr is not None else f"Need 30 trades ({total} so far)"
    using_measured = trade_stats.get("using_measured_wr", False)
    kelly_note = "Using measured win rate for Kelly sizing ✓" if using_measured else "Using estimated 55% (need 30 trades)"

    return (
        f"📊 <b>ForexMind Performance</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Live Statistics</b>\n"
        f"Total Closed:   {total}\n"
        f"Wins:           {trade_stats.get('wins', 0)}\n"
        f"Losses:         {trade_stats.get('losses', 0)}\n"
        f"Win Rate:       <b>{wr_str}</b>\n"
        f"\n"
        f"<b>Today</b>\n"
        f"Daily P&L:      ${float(stats.get('daily_pnl_usd', 0)):,.2f}\n"
        f"Open Trades:    {stats.get('open_trades', 0)}\n"
        f"\n"
        f"<i>{kelly_note}</i>\n"
        f"\n"
        f"<b>Note:</b> <i>52-58% win rate is profitable at 2:1 R:R. "
        f"If above 60%, verify no data leakage.</i>"
    ).strip()


# ── Context keys ──────────────────────────────────────────────────────────────

AUTOTRADE_KEY = "autotrade_enabled"
PENDING_TRADE_KEY = "pending_trade"


# ── Handlers ──────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cfg = get_settings()
    mode = "🧪 Paper trading" if cfg.app.paper_trading else "⚠️ LIVE trading"
    await update.message.reply_html(
        f"👋 <b>Welcome to ForexMind!</b>\n\n"
        f"Your AI forex analyst. Mode: {mode}\n\n"
        f"<b>Quick start:</b>\n"
        f"/signal EUR/USD — analyse a pair\n"
        f"/trade EUR/USD — signal + one-tap trade button\n"
        f"/signals — top pairs right now\n"
        f"/help — all commands\n\n"
        f"<i>Realistic targets: 52-58% win rate, 2:1 R:R = profitable system.</i>"
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_html(
        "<b>ForexMind Commands</b>\n\n"
        "<b>Signals</b>\n"
        "/signal EUR/USD — Live signal analysis\n"
        "/signals — Top pairs for current session\n"
        "/trade EUR/USD — Signal + confirm trade button\n"
        "\n"
        "<b>Trading</b>\n"
        "/autotrade on — Enable auto-trading (scheduler)\n"
        "/autotrade off — Disable auto-trading\n"
        "/trades — Open positions (with ❌ Close buttons)\n"
        "/close EUR/USD — Close all positions on a pair\n"
        "/close all — Close every open trade\n"
        "\n"
        "<b>Account &amp; Stats</b>\n"
        "/account — Balance, P&amp;L, margin\n"
        "/stats — Win rate, trade count, performance\n"
        "/backtest EUR/USD — 1-year backtest results\n"
        "\n"
        "<b>Market</b>\n"
        "/sessions — Active sessions + recommended pairs\n"
        "\n"
        "<i>Type anything to chat with Claude AI agent.</i>"
    )


async def cmd_sessions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    status = get_session_status()
    pairs = best_pairs_for_session()[:6]

    if status.is_weekend:
        await update.message.reply_text("🚫 Weekend — Forex markets closed")
        return

    lines = []
    for s in ["Sydney", "Tokyo", "London", "New York"]:
        active = s in status.active_sessions
        lines.append(f"{'🟢' if active else '⚫'} {s}")

    overlap = (
        f"\n🔥 <b>OVERLAP: {', '.join(status.active_overlaps)} — Prime scalping time!</b>"
        if status.active_overlaps else ""
    )
    rec_pairs = ", ".join(p.replace("_", "/") for p in pairs)
    score = status.session_score

    await update.message.reply_html(
        f"<b>Market Sessions (UTC)</b>\n"
        f"{'  '.join(lines)}"
        f"{overlap}\n\n"
        f"Session score: {score:.1f}/1.0\n"
        f"Recommended: <code>{rec_pairs}</code>"
    )


async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /signal EUR/USD — shows signal without trade button."""
    args = context.args
    if not args:
        await update.message.reply_text("Usage: /signal EUR/USD")
        return

    pair = args[0].upper().replace("/", "_").replace("-", "_")
    msg = await update.message.reply_text(f"🔍 Analysing {pair}...")

    try:
        from forexmind.agents.tools import _get_signal
        result_str = await _get_signal(pair, "H1", 300)
        data = json.loads(result_str)

        if "error" in data:
            await msg.edit_text(f"❌ Error: {data['error']}")
            return

        text = format_signal_message(data)
        sig = data.get("signal", {})
        action = sig.get("action", "HOLD")

        keyboard_rows = [[
            InlineKeyboardButton("📰 News", callback_data=f"news_{pair}"),
            InlineKeyboardButton("🔄 Refresh", callback_data=f"signal_{pair}"),
        ]]
        if action in ("BUY", "SELL"):
            keyboard_rows.insert(0, [
                InlineKeyboardButton(
                    f"📝 Trade {action} {pair.replace('_', '/')}",
                    callback_data=f"prep_trade_{pair}"
                )
            ])

        await msg.edit_text(
            text,
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup(keyboard_rows),
        )

    except Exception as e:
        await msg.edit_text(f"❌ Failed to get signal: {str(e)[:200]}")


async def cmd_trade(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /trade EUR/USD — Fetch signal and show a confirm/cancel trade button.
    The confirmation button places the trade via OANDA.
    """
    args = context.args
    if not args:
        await update.message.reply_text("Usage: /trade EUR/USD")
        return

    pair = args[0].upper().replace("/", "_").replace("-", "_")
    msg = await update.message.reply_text(f"🔍 Analysing {pair} for trade...")

    try:
        from forexmind.agents.tools import _get_signal
        result_str = await _get_signal(pair, "H1", 300)
        data = json.loads(result_str)

        if "error" in data:
            await msg.edit_text(f"❌ Error: {data['error']}")
            return

        sig = data.get("signal", {})
        action = sig.get("action", "HOLD")
        conf_pct = float(sig.get("confidence", 0))

        if action not in ("BUY", "SELL"):
            await msg.edit_text(
                f"⚪ <b>HOLD — {pair.replace('_', '/')}</b>\n\n"
                f"No tradeable signal right now.\n"
                f"Confidence: {conf_pct:.0f}% ({confidence_label(conf_pct)})\n\n"
                f"<i>Try again during London/NY overlap for best conditions.</i>",
                parse_mode=ParseMode.HTML,
            )
            return

        # Store signal data in user context for confirmation callback
        context.user_data[PENDING_TRADE_KEY] = data

        text = format_signal_message(data, include_trade_note=True)
        rr = float(sig.get("risk_reward", 0))
        rr_ok = rr >= 1.8
        conf_ok = conf_pct >= 55.0

        # Show risk warnings in the buttons
        confirm_label = f"✅ Confirm {action}"
        if not conf_ok:
            confirm_label = f"⚠️ Low conf — Trade {action} anyway"
        if not rr_ok:
            confirm_label = f"⚠️ Low R:R — Trade {action} anyway"

        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton(confirm_label, callback_data=f"confirm_trade_{pair}"),
            InlineKeyboardButton("❌ Cancel", callback_data="cancel_trade"),
        ]])
        await msg.edit_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)

    except Exception as e:
        await msg.edit_text(f"❌ Failed: {str(e)[:200]}")


async def cmd_autotrade(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /autotrade on|off — Toggle auto-trading mode for the scheduler.
    When on, the scheduler places trades automatically at AUTO_TRADE_CONFIDENCE threshold.
    """
    args = context.args
    if not args or args[0].lower() not in ("on", "off"):
        current = context.bot_data.get(AUTOTRADE_KEY, False)
        status = "ON ✅" if current else "OFF ❌"
        await update.message.reply_html(
            f"Auto-trade is currently: <b>{status}</b>\n\n"
            f"Usage: /autotrade on  or  /autotrade off\n\n"
            f"<i>When ON, trades are placed automatically when ensemble confidence ≥ 62% "
            f"and R:R ≥ 1.8. You receive a Telegram notification for each trade.</i>"
        )
        return

    enable = args[0].lower() == "on"
    context.bot_data[AUTOTRADE_KEY] = enable

    if enable:
        from forexmind.scheduler import AUTO_TRADE_CONFIDENCE, MIN_RR
        await update.message.reply_html(
            "⚡ <b>Auto-trading ENABLED</b>\n\n"
            "Trades will be placed automatically when:\n"
            f"• Ensemble confidence ≥ {AUTO_TRADE_CONFIDENCE:.0f}% (3+ strategies agree)\n"
            f"• R:R ≥ {MIN_RR}:1\n"
            "• Active London or NY session\n\n"
            f"Mode: {'🧪 Paper trading — no real money' if get_settings().app.paper_trading else '⚠️ LIVE TRADING'}\n\n"
            "<i>Use /autotrade off to disable.</i>"
        )
    else:
        await update.message.reply_html(
            "🛑 <b>Auto-trading DISABLED</b>\n\n"
            "You'll still receive signal alerts, but no trades will be placed automatically.\n"
            "Use /trade EUR/USD to manually confirm trades."
        )


async def cmd_trades(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show open trades with close buttons."""
    msg = await update.message.reply_text("📋 Fetching open trades...")
    try:
        from forexmind.data.oanda_client import get_oanda_client
        client = get_oanda_client()
        trades = await client.get_open_trades()

        if not trades:
            await msg.edit_text("📭 No open trades right now.")
            return

        lines = ["📋 <b>Open Trades</b>\n━━━━━━━━━━━━━━━━━━━━"]
        keyboard_rows = []
        for t in trades:
            trade_id = t.get("id", "?")
            instrument = t.get("instrument", "?")
            units = float(t.get("currentUnits", t.get("initialUnits", 0)))
            direction = "BUY" if units > 0 else "SELL"
            entry = float(t.get("price", 0))
            pnl = float(t.get("unrealizedPL", 0))
            pnl_emoji = "📈" if pnl >= 0 else "📉"
            sl = float(t.get("stopLossOrder", {}).get("price", 0))
            tp = float(t.get("takeProfitOrder", {}).get("price", 0))
            sl_str = f" | SL: {sl}" if sl else ""
            tp_str = f" | TP: {tp}" if tp else ""

            lines.append(
                f"{pnl_emoji} <b>{instrument.replace('_', '/')}</b> {direction} #{trade_id}\n"
                f"   Units: {abs(int(units)):,} | Entry: {entry}{sl_str}{tp_str}\n"
                f"   P&L: <b>${pnl:,.2f}</b>"
            )
            keyboard_rows.append([
                InlineKeyboardButton(
                    f"❌ Close {instrument.replace('_', '/')} {direction} (${pnl:+,.2f})",
                    callback_data=f"close_trade_{trade_id}_{instrument}"
                )
            ])

        keyboard_rows.append([
            InlineKeyboardButton("❌ Close ALL trades", callback_data="close_all_trades")
        ])

        await msg.edit_text(
            "\n".join(lines),
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup(keyboard_rows),
        )

    except Exception as e:
        await msg.edit_text(f"❌ Error fetching trades: {str(e)[:200]}")


async def cmd_close(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /close EUR/USD — Close all positions on a pair immediately.
    /close all     — Close all open trades.
    """
    args = context.args
    if not args:
        await update.message.reply_text(
            "Usage:\n"
            "/close EUR/USD — close all positions on a pair\n"
            "/close all     — close every open trade\n\n"
            "Or use /trades and tap the ❌ Close button."
        )
        return

    target = args[0].upper().replace("/", "_")
    msg = await update.message.reply_text(f"⏳ Closing {target}...")
    try:
        from forexmind.data.oanda_client import get_oanda_client
        client = get_oanda_client()
        trades = await client.get_open_trades()

        if not trades:
            await msg.edit_text("📭 No open trades to close.")
            return

        to_close = trades if target == "ALL" else [
            t for t in trades if t.get("instrument", "") == target
        ]

        if not to_close:
            await msg.edit_text(f"📭 No open trades found for {target.replace('_', '/')}.")
            return

        results = []
        for t in to_close:
            trade_id = t.get("id")
            instrument = t.get("instrument", "?")
            units = float(t.get("currentUnits", 0))
            direction = "BUY" if units > 0 else "SELL"
            result = await client.close_trade(trade_id)
            if result.success:
                filled = result.filled_price or float(t.get("price", 0))
                entry = float(t.get("price", 0))
                pnl = float(t.get("unrealizedPL", 0))
                results.append(f"✅ Closed {instrument.replace('_', '/')} {direction} @ {filled} | P&L: ${pnl:+,.2f}")
            else:
                results.append(f"❌ Failed to close #{trade_id}: {result.message}")

        await msg.edit_text("\n".join(results), parse_mode=ParseMode.HTML)

    except Exception as e:
        await msg.edit_text(f"❌ Error: {str(e)[:200]}")


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show win rate, P&L and trade stats from the risk manager."""
    msg = await update.message.reply_text("📊 Fetching stats...")
    try:
        from forexmind.risk.manager import get_risk_manager
        from forexmind.agents.tools import _get_account

        rm = get_risk_manager()
        acc_str = await _get_account()
        acc = json.loads(acc_str)

        stats = {
            "daily_pnl_usd": acc.get("daily_pnl_usd", 0),
            "open_trades": acc.get("open_trade_count", 0),
        }
        text = format_stats_message(stats, rm.trade_stats)
        await msg.edit_text(text, parse_mode=ParseMode.HTML)

    except Exception as e:
        await msg.edit_text(f"❌ Error: {str(e)[:200]}")


async def cmd_signals(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Get signals for all recommended session pairs."""
    msg = await update.message.reply_text("🔍 Scanning session pairs...")
    try:
        from forexmind.agents.tools import _get_signal
        status = get_session_status()
        if status.is_weekend:
            await msg.edit_text(
                "🚫 <b>Weekend — Forex markets closed</b>\n\n"
                "Markets reopen Sunday 21:00 UTC.",
                parse_mode=ParseMode.HTML
            )
            return
        pairs = best_pairs_for_session()[:6]
        if not pairs:
            await msg.edit_text(
                "⚠️ <b>No active session</b>\n\nNo recommended pairs right now. Try during London or New York session.",
                parse_mode=ParseMode.HTML
            )
            return

        import asyncio
        # Scan H1 (trend) + M15 (entry timing) for each pair in parallel
        h1_tasks = [_get_signal(p, "H1", 300) for p in pairs]
        m15_tasks = [_get_signal(p, "M15", 200) for p in pairs]
        h1_results, m15_results = await asyncio.gather(
            asyncio.gather(*h1_tasks),
            asyncio.gather(*m15_tasks),
        )

        lines = ["<b>Session Signals</b>  <i>(H1 trend | M15 entry)</i>\n"]
        for pair, h1_res, m15_res in zip(pairs, h1_results, m15_results):
            h1 = json.loads(h1_res).get("signal", {})
            m15 = json.loads(m15_res).get("signal", {})

            # Use H1 as primary signal, M15 as entry confirmation
            action = h1.get("action", "HOLD")
            conf = float(h1.get("confidence", 0))
            rr = h1.get("risk_reward", "-")
            m15_action = m15.get("action", "HOLD")
            m15_conf = float(m15.get("confidence", 0))

            # Alignment bonus: both timeframes agree → stronger signal
            aligned = action == m15_action and action != "HOLD"

            if action == "HOLD" or conf < 42:
                emoji = "⚪"
                action = "HOLD"
            elif conf >= 65 and aligned:
                emoji = "🟢" if action == "BUY" else "🔴"  # Strong + aligned
            elif conf >= 55:
                emoji = "🟡"  # Moderate — tradeable
            else:
                emoji = "🟡"

            clabel = confidence_label(conf) if action != "HOLD" else ""
            align_str = " ✅" if aligned else ""
            rr_str = f"R:R {rr}" if action != "HOLD" else ""
            lines.append(
                f"{emoji} <b>{pair.replace('_', '/')}</b>: {action} "
                f"({conf:.0f}% {clabel}){align_str} {rr_str}".rstrip()
            )

        lines.append("\n<i>✅ = H1 + M15 aligned (higher quality)</i>")
        await msg.edit_text("\n".join(lines), parse_mode=ParseMode.HTML)
    except Exception as e:
        await msg.edit_text(f"❌ Error: {str(e)[:200]}")


async def cmd_account(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = await update.message.reply_text("💼 Fetching account...")
    try:
        from forexmind.agents.tools import _get_account
        result = await _get_account()
        data = json.loads(result)
        if "error" in data:
            await msg.edit_text(f"❌ {data['error']}")
        else:
            await msg.edit_text(format_account_message(data), parse_mode=ParseMode.HTML)
    except Exception as e:
        await msg.edit_text(f"❌ {str(e)[:200]}")


async def cmd_backtest(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Quick backtest command from Telegram."""
    args = context.args
    pair = (args[0].upper().replace("/", "_") if args else "EUR_USD")
    msg = await update.message.reply_text(f"⏳ Running backtest on {pair} (2024, H1)...")
    try:
        from forexmind.agents.tools import _run_backtest
        # Use H1 to match new primary timeframe; H1 has far fewer candles so no OANDA count limit
        result = await _run_backtest(pair, "H1", "2024-01-01", "2024-12-31")
        data = json.loads(result)
        if "error" in data:
            await msg.edit_text(f"❌ {data['error']}")
            return

        wr = float(data.get("win_rate", "0").replace("%", "")) if isinstance(data.get("win_rate"), str) else 0
        wr_str = data.get("win_rate", "-")
        pf = data.get("profit_factor", "-")
        sharpe = data.get("sharpe_ratio", "-")
        dd = data.get("max_drawdown_pct", "-")
        ret = data.get("net_return_pct", "-")
        n = data.get("total_trades", "-")

        quality = "✅ Positive edge" if float(pf or 0) > 1.1 else "⚠️ Break-even or loss"
        text = (
            f"📈 <b>Backtest: {pair.replace('_', '/')} M5 (2024)</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Trades:        {n}\n"
            f"Win Rate:      <b>{wr_str}</b>\n"
            f"Profit Factor: <b>{pf}</b>  {quality}\n"
            f"Net Return:    {ret}%\n"
            f"Max Drawdown:  {dd}%\n"
            f"Sharpe Ratio:  {sharpe}\n\n"
            f"<i>Note: backtest uses rule-based strategy only (H1 timeframe). "
            f"Ensemble results may differ.</i>"
        )
        await msg.edit_text(text, parse_mode=ParseMode.HTML)
    except Exception as e:
        await msg.edit_text(f"❌ {str(e)[:200]}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle free-text messages by forwarding to Claude agent."""
    user_text = update.message.text
    msg = await update.message.reply_text("🤔 Thinking...")
    try:
        from forexmind.agents.claude_agent import get_agent
        agent = get_agent()
        response = await agent.chat(user_text)
        if len(response) > 4000:
            response = response[:3997] + "..."
        await msg.edit_text(response, parse_mode=ParseMode.HTML)
    except Exception as e:
        await msg.edit_text(f"❌ Agent error: {str(e)[:200]}")


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle all inline keyboard button callbacks."""
    query = update.callback_query
    await query.answer()
    data = query.data

    # ── Confirm trade placement ───────────────────────────────────────────────
    if data.startswith("confirm_trade_"):
        pair = data[len("confirm_trade_"):]
        pending = context.user_data.get(PENDING_TRADE_KEY)

        if not pending:
            await query.edit_message_text("⚠️ Trade data expired. Please use /trade again.")
            return

        await query.edit_message_text(f"⏳ Placing {pair.replace('_', '/')} trade...")
        try:
            from forexmind.agents.tools import _place_trade
            from forexmind.data.oanda_client import get_oanda_client
            from forexmind.risk.manager import get_risk_manager
            import json as _json

            sig = pending.get("signal", {})
            ind = pending.get("indicators", {})
            direction = sig.get("action")
            entry = float(sig.get("entry", 0))
            stop_loss = float(sig.get("stop_loss", 0))
            take_profit = float(sig.get("take_profit", 0))
            atr = float(ind.get("atr", 0.0005))

            client = get_oanda_client()
            acc = await client.get_account()
            rm = get_risk_manager()
            proposal = rm.calculate_risk(
                instrument=pair,
                direction=direction,
                entry=entry,
                atr=atr,
                account_balance=acc.balance,
            )

            if not proposal.approved:
                await query.edit_message_text(
                    f"🚫 <b>Trade rejected by risk manager</b>\n\n"
                    f"Reason: {proposal.rejection_reason}",
                    parse_mode=ParseMode.HTML,
                )
                return

            result_str = await _place_trade(
                instrument=pair,
                direction=direction,
                units=proposal.units,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            result = _json.loads(result_str)

            if "error" in result:
                await query.edit_message_text(
                    f"❌ <b>Order failed</b>\n{result['error']}",
                    parse_mode=ParseMode.HTML,
                )
                return

            if result.get("status") == "pending_confirmation":
                await query.edit_message_text(
                    "⚠️ <b>Live trading requires explicit confirmation.</b>\n"
                    "Set PAPER_TRADING=True in .env for paper mode.",
                    parse_mode=ParseMode.HTML,
                )
                return

            cfg = get_settings()
            mode = "Paper" if cfg.app.paper_trading else "⚠️ LIVE"
            filled = float(result.get("filled_price", 0)) or entry
            trade_id = result.get("trade_id") or "pending"
            await query.edit_message_text(
                f"✅ <b>Trade Placed [{mode}]</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"Pair:      {pair.replace('_', '/')}\n"
                f"Direction: {direction}\n"
                f"Units:     {proposal.units:,}\n"
                f"Filled @   {filled}\n"
                f"Stop Loss: {stop_loss}\n"
                f"TP:        {take_profit}\n"
                f"Risk:      {proposal.risk_pct:.2f}% (${proposal.risk_usd:.2f})\n"
                f"R:R:       {proposal.risk_reward_ratio:.1f}:1\n"
                f"Trade ID:  {trade_id}",
                parse_mode=ParseMode.HTML,
            )
            context.user_data.pop(PENDING_TRADE_KEY, None)

        except Exception as e:
            await query.edit_message_text(f"❌ Trade error: {str(e)[:300]}")

    # ── Cancel trade ──────────────────────────────────────────────────────────
    elif data == "cancel_trade":
        context.user_data.pop(PENDING_TRADE_KEY, None)
        await query.edit_message_text("❌ Trade cancelled.")

    # ── Close individual trade ─────────────────────────────────────────────────
    elif data.startswith("close_trade_"):
        parts = data[len("close_trade_"):].split("_", 1)
        trade_id = parts[0]
        instrument = parts[1] if len(parts) > 1 else "?"
        await query.edit_message_text(f"⏳ Closing {instrument.replace('_', '/')}...")
        try:
            from forexmind.data.oanda_client import get_oanda_client
            client = get_oanda_client()
            result = await client.close_trade(trade_id)
            if result.success:
                filled = result.filled_price
                await query.edit_message_text(
                    f"✅ <b>Closed {instrument.replace('_', '/')}</b>\n"
                    f"Trade ID: {trade_id}\n"
                    f"Exit Price: {filled}",
                    parse_mode=ParseMode.HTML,
                )
            else:
                await query.edit_message_text(f"❌ Close failed: {result.message}")
        except Exception as e:
            await query.edit_message_text(f"❌ Error: {str(e)[:200]}")

    # ── Close all trades ───────────────────────────────────────────────────────
    elif data == "close_all_trades":
        await query.edit_message_text("⏳ Closing all trades...")
        try:
            from forexmind.data.oanda_client import get_oanda_client
            client = get_oanda_client()
            trades = await client.get_open_trades()
            if not trades:
                await query.edit_message_text("📭 No open trades.")
                return
            lines = ["<b>Closing all trades:</b>"]
            for t in trades:
                trade_id = t.get("id")
                instrument = t.get("instrument", "?")
                pnl = float(t.get("unrealizedPL", 0))
                result = await client.close_trade(trade_id)
                if result.success:
                    lines.append(f"✅ {instrument.replace('_', '/')} @ {result.filled_price} | P&L: ${pnl:+,.2f}")
                else:
                    lines.append(f"❌ {instrument.replace('_', '/')} failed: {result.message}")
            await query.edit_message_text("\n".join(lines), parse_mode=ParseMode.HTML)
        except Exception as e:
            await query.edit_message_text(f"❌ Error: {str(e)[:200]}")

    # ── Prepare trade from signal screen ─────────────────────────────────────
    elif data.startswith("prep_trade_"):
        pair = data[len("prep_trade_"):]
        await query.edit_message_text(f"🔍 Preparing trade for {pair.replace('_', '/')}...")
        try:
            from forexmind.agents.tools import _get_signal
            result_str = await _get_signal(pair, "H1", 300)
            signal_data = json.loads(result_str)
            context.user_data[PENDING_TRADE_KEY] = signal_data

            sig = signal_data.get("signal", {})
            action = sig.get("action", "HOLD")
            conf_pct = float(sig.get("confidence", 0))

            if action not in ("BUY", "SELL"):
                await query.edit_message_text("⚪ Signal changed to HOLD — no trade available now.")
                return

            text = format_signal_message(signal_data, include_trade_note=True)
            keyboard = InlineKeyboardMarkup([[
                InlineKeyboardButton(f"✅ Confirm {action}", callback_data=f"confirm_trade_{pair}"),
                InlineKeyboardButton("❌ Cancel", callback_data="cancel_trade"),
            ]])
            await query.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
        except Exception as e:
            await query.edit_message_text(f"❌ {str(e)[:200]}")

    # ── Refresh signal ────────────────────────────────────────────────────────
    elif data.startswith("signal_"):
        pair = data[7:]
        await query.edit_message_text("🔍 Refreshing...")
        try:
            from forexmind.agents.tools import _get_signal
            result = await _get_signal(pair, "H1", 300)
            signal_data = json.loads(result)
            text = format_signal_message(signal_data)
            sig = signal_data.get("signal", {})
            action = sig.get("action", "HOLD")

            keyboard_rows = [[
                InlineKeyboardButton("📰 News", callback_data=f"news_{pair}"),
                InlineKeyboardButton("🔄 Refresh", callback_data=f"signal_{pair}"),
            ]]
            if action in ("BUY", "SELL"):
                keyboard_rows.insert(0, [
                    InlineKeyboardButton(
                        f"📝 Trade {action}", callback_data=f"prep_trade_{pair}"
                    )
                ])
            await query.edit_message_text(
                text, parse_mode=ParseMode.HTML,
                reply_markup=InlineKeyboardMarkup(keyboard_rows),
            )
        except Exception as e:
            await query.edit_message_text(f"❌ Error: {str(e)[:200]}")

    # ── News ──────────────────────────────────────────────────────────────────
    elif data.startswith("news_"):
        pair = data[5:]
        try:
            from forexmind.agents.tools import _get_news
            result = await _get_news(pair, 4)
            news_data = json.loads(result)
            sentiment = news_data.get("sentiment", {})
            articles = news_data.get("articles", [])[:5]
            score = float(sentiment.get("score", 0))
            impact = sentiment.get("impact", "neutral").upper()
            lines = [
                f"📰 <b>News: {pair.replace('_', '/')}</b>",
                f"Sentiment: {impact} ({score:+.3f})",
                f"Coverage: {sentiment.get('article_count', 0)} articles\n",
            ]
            for art in articles:
                sent = float(art.get("sentiment", 0))
                s_emoji = "📈" if sent > 0.05 else "📉" if sent < -0.05 else "➖"
                lines.append(f"{s_emoji} {art['headline'][:100]}")
            await query.edit_message_text("\n".join(lines), parse_mode=ParseMode.HTML)
        except Exception as e:
            await query.edit_message_text(f"❌ {str(e)[:200]}")


# ── App builder ────────────────────────────────────────────────────────────────

def build_telegram_app() -> "Application":  # type: ignore[type-arg]
    if not TELEGRAM_AVAILABLE:
        raise ImportError("python-telegram-bot not installed")

    cfg = get_settings()
    if not cfg.telegram.is_configured:
        raise ValueError("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set in .env")

    app = Application.builder().token(cfg.telegram.bot_token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("signal", cmd_signal))
    app.add_handler(CommandHandler("signals", cmd_signals))
    app.add_handler(CommandHandler("trade", cmd_trade))
    app.add_handler(CommandHandler("autotrade", cmd_autotrade))
    app.add_handler(CommandHandler("trades", cmd_trades))
    app.add_handler(CommandHandler("close", cmd_close))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("account", cmd_account))
    app.add_handler(CommandHandler("backtest", cmd_backtest))
    app.add_handler(CommandHandler("sessions", cmd_sessions))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CallbackQueryHandler(handle_callback))

    log.info("Telegram bot configured with auto-trade support")
    return app


async def run_telegram_bot() -> None:
    import asyncio
    app = build_telegram_app()
    log.info("Telegram bot starting (long-polling)...")
    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)
    log.info("Telegram bot running.")
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    finally:
        await app.updater.stop()
        await app.stop()
        await app.shutdown()
