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
  /monitor EUR/USD 20 — Auto-close when profit hits $20
  /help           — List commands
  [any text]      — Chat with Claude agent
"""

from __future__ import annotations

import asyncio
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
        JobQueue,
    )
    from telegram.constants import ParseMode
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    log.warning("python-telegram-bot not installed. Run: pip install python-telegram-bot")


# ── Visual helpers ─────────────────────────────────────────────────────────────

def _bar(value: float, max_val: float = 100.0, width: int = 10) -> str:
    """Unicode block progress bar. ▓ filled · ░ empty."""
    filled = max(0, min(width, round(value / max_val * width)))
    return "▓" * filled + "░" * (width - filled)


def _pnl_icon(pnl: float) -> str:
    if pnl > 0:  return "📈"
    if pnl < 0:  return "📉"
    return "➖"


def _risk_tier(risk_pct: float) -> str:
    if risk_pct >= 5:  return "TIER 4  MAX"
    if risk_pct >= 4:  return "TIER 3  HIGH"
    if risk_pct >= 3:  return "TIER 2  MID"
    return "TIER 1  BASE"


# ── Confidence helpers ────────────────────────────────────────────────────────

def confidence_label(conf_pct: float) -> str:
    """
    Calibrated for diluted ensemble scores across 4 weighted strategies.
    Realistic achievable ranges:
      2 strategies agreeing → ~42–55%
      3 strategies agreeing → ~55–75%
      4 strategies agreeing → ~75–85%
    """
    if conf_pct >= 75:  return "🔥 STRONG"
    if conf_pct >= 65:  return "✅ HIGH"
    if conf_pct >= 55:  return "📶 MODERATE"
    if conf_pct >= 42:  return "⚡ DEVELOPING"
    return "❌ BELOW EDGE"


def confidence_note(conf_pct: float) -> str:
    """One-line context that sets realistic expectations."""
    if conf_pct >= 75:  return "All strategies aligned — premium quality"
    if conf_pct >= 65:  return "3–4 strategies agree — tradeable signal"
    if conf_pct >= 55:  return "Majority agreement — trade with discipline"
    if conf_pct >= 42:  return "2 strategies agree — size down, manage tightly"
    return "Below edge threshold — stay flat"


# ── Message formatters ────────────────────────────────────────────────────────

def signal_caution_warning(data: dict) -> str:
    """
    Return a warning string if the signal context is risky:
    - ATR spike (current ATR > 2x 20-bar average) → news event likely in progress
    - News sentiment strongly opposes the signal direction
    Returns empty string if no warning.
    """
    sig = data.get("signal", {})
    ind = data.get("indicators", {})
    sentiment = data.get("news_sentiment", {})
    action = sig.get("action", "HOLD")

    warnings = []

    # ATR spike check
    atr = ind.get("atr")
    atr_avg = ind.get("atr_avg_20")
    if atr and atr_avg and atr_avg > 0:
        ratio = atr / atr_avg
        if ratio >= 2.0:
            warnings.append(f"⚠️ ATR spike ({ratio:.1f}x normal) — news event likely, technicals unreliable")

    # News sentiment opposing direction
    if action in ("BUY", "SELL"):
        score = float(sentiment.get("score", 0))
        if action == "BUY" and score <= -0.3:
            warnings.append(f"⚠️ News sentiment NEGATIVE ({score:+.2f}) — opposes BUY signal")
        elif action == "SELL" and score >= 0.3:
            warnings.append(f"⚠️ News sentiment POSITIVE ({score:+.2f}) — opposes SELL signal")

    return "\n".join(warnings)


def format_signal_message(data: dict, include_trade_note: bool = False) -> str:
    """Format a signal dict into a premium Telegram HTML message."""
    sig = data.get("signal", {})
    ind = data.get("indicators", {})
    session = data.get("session", {})
    sentiment = data.get("news_sentiment", {})

    pair = data.get("instrument", "").replace("_", "/")
    action = sig.get("action", "HOLD")
    conf_pct = float(sig.get("confidence", 0))
    clabel = confidence_label(conf_pct)
    cnote = confidence_note(conf_pct)
    caution = signal_caution_warning(data)

    dir_emoji = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "⚪"
    dir_arrow = "↑" if action == "BUY" else "↓" if action == "SELL" else "—"

    # Confidence bar
    conf_bar_str = _bar(conf_pct, 100, 10)
    agreeing = sig.get("agreeing_strategies", 0)
    try:
        agreeing = int(agreeing)
    except (TypeError, ValueError):
        agreeing = 0

    # Session context
    session_names = session.get("active", session.get("active_sessions", []))
    overlaps = session.get("active_overlaps", session.get("overlaps", []))
    if overlaps:
        session_line = f"{', '.join(overlaps)} 🔥"
    elif session_names:
        session_line = ", ".join(session_names)
    else:
        session_line = "Off-hours"

    # News
    sentiment_score = float(sentiment.get("score", 0))
    sentiment_impact = sentiment.get("impact", "neutral").upper()
    news_line = f"{sentiment_impact} ({sentiment_score:+.2f})"

    # ATR in pips
    pip_mult = 100 if "JPY" in pair else 10000
    atr_raw = float(ind.get("atr", 0) or 0)
    atr_avg_raw = float(ind.get("atr_avg_20", 0) or 0)
    atr_pips = round(atr_raw * pip_mult, 1) if atr_raw else None
    atr_ratio = (atr_raw / atr_avg_raw) if atr_avg_raw > 0 else 1.0
    atr_state = "⚠️ SPIKE" if atr_ratio >= 2.0 else "Normal"
    atr_display = f"{atr_pips:.1f} pips  [{atr_state}]" if atr_pips else ind.get("atr", "—")

    # Indicators
    rsi_val = ind.get("rsi")
    rsi_display = f"{float(rsi_val):.0f}" if rsi_val else "—"
    rsi_zone = ind.get("rsi_zone", "—")
    rsi_bar_str = _bar(float(rsi_val), 100, 10) if rsi_val else "░" * 10
    macd_str = ind.get("macd_cross", "—")
    ema_str = (ind.get("ema_trend") or "—").replace("_", " ").title()
    psar_str = (ind.get("psar_signal") or "—").title()
    adx_val = ind.get("adx", "—")
    adx_str = ind.get("adx_trend_strength", "—")

    # HOLD — compact format
    if action == "HOLD":
        return (
            f"{dir_emoji} <b>HOLD  {pair}</b>  ·  {dir_arrow}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            + (f"{caution}\n━━━━━━━━━━━━━━━━━━━━━━━━━\n" if caution else "")
            + f"🎯 Confidence  {conf_bar_str}  {conf_pct:.0f}%\n"
            f"   {clabel}\n"
            f"   <i>{cnote}</i>\n\n"
            f"No entry setup at this time. Monitoring for conditions.\n\n"
            f"─────────────────────────\n"
            f"📊 <b>Technicals</b>\n"
            f"RSI(14)  {rsi_bar_str}  {rsi_display}  [{rsi_zone}]\n"
            f"EMA      {ema_str}\n"
            f"ADX      {adx_val}  [{adx_str}]\n"
            f"\n"
            f"🌍 <b>Context</b>\n"
            f"Session   {session_line}"
        ).strip()

    # BUY / SELL — full detail
    rr = sig.get("risk_reward", "—")
    risk_pct = float(sig.get("risk_pct", 0) or 0)
    tier = _risk_tier(risk_pct)

    msg = (
        f"{dir_emoji} <b>{action}  {pair}</b>  ·  {dir_arrow}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    )

    if caution:
        msg += f"{caution}\n━━━━━━━━━━━━━━━━━━━━━━━━━\n"

    msg += (
        f"🎯 <b>Confidence</b>  {conf_bar_str}  <b>{conf_pct:.0f}%</b>\n"
        f"   {clabel}  ·  {agreeing}/4 strategies\n"
        f"   <i>{cnote}</i>\n"
        f"\n"
        f"📍 Entry         <code>{sig.get('entry', '—')}</code>\n"
        f"🛡 Stop Loss     <code>{sig.get('stop_loss', '—')}</code>  "
        f"({sig.get('stop_loss_pips', '—')} pips)\n"
        f"🎯 Take Profit   <code>{sig.get('take_profit', '—')}</code>  "
        f"(+{sig.get('take_profit_pips', '—')} pips)\n"
        f"⚡ R:R           <b>{rr}:1</b>  ·  Risk: {risk_pct:.1f}%  [{tier}]\n"
        f"\n"
        f"─────────────────────────\n"
        f"📊 <b>Technicals</b>\n"
        f"RSI(14)  {rsi_bar_str}  {rsi_display}  [{rsi_zone}]\n"
        f"MACD     {macd_str}\n"
        f"EMA      {ema_str}\n"
        f"PSAR     {psar_str}\n"
        f"ADX      {adx_val}  [{adx_str}]\n"
        f"\n"
        f"─────────────────────────\n"
        f"🌍 <b>Context</b>\n"
        f"Session      {session_line}\n"
        f"News         {news_line}\n"
        f"Volatility   {atr_display}"
    )

    if include_trade_note and action in ("BUY", "SELL"):
        mode = "paper" if get_settings().app.paper_trading else "⚠️ LIVE"
        msg += (
            f"\n\n"
            f"<i>↓ Tap Confirm to place on your OANDA {mode} account.</i>"
        )

    return msg.strip()


def format_account_message(data: dict) -> str:
    balance = float(data.get("balance", 0))
    nav = float(data.get("nav", 0))
    pnl = float(data.get("unrealized_pnl", 0))
    daily_pnl = float(data.get("daily_pnl_usd", 0))
    margin_used = float(data.get("margin_used", 0))
    open_cnt = data.get("open_trade_count", 0)
    kill = data.get("kill_switch_active", False)
    mode = "🧪 Paper" if get_settings().app.paper_trading else "⚠️ LIVE"

    # Margin utilisation bar
    margin_pct = (margin_used / nav * 100) if nav > 0 else 0
    margin_bar = _bar(margin_pct, 100, 8)
    kill_str = "🔴 TRIGGERED" if kill else "🟢 OK"

    # Equity milestone progress
    milestones = [500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 7500, 10000, 15000, 20000]
    milestone_str = ""
    for m in milestones:
        if balance < m:
            pct = max(0, min(100, balance / m * 100))
            m_bar = _bar(pct, 100, 8)
            milestone_str = (
                f"\n─────────────────────────\n"
                f"🏆 <b>Next Milestone</b>  ${m:,.0f}\n"
                f"   {m_bar}  ${m - balance:,.2f} to go"
            )
            break

    return (
        f"💼 <b>Account Snapshot</b>  ·  {mode}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Balance       <b>${balance:,.2f}</b>\n"
        f"NAV           ${nav:,.2f}\n"
        f"{_pnl_icon(pnl)} Open P&amp;L     <b>${pnl:+,.2f}</b>\n"
        f"{_pnl_icon(daily_pnl)} Today's P&amp;L  <b>${daily_pnl:+,.2f}</b>\n"
        f"Open Trades   {open_cnt}\n"
        f"\n"
        f"Margin Used   ${margin_used:,.2f}\n"
        f"Utilisation   {margin_bar}  {margin_pct:.1f}%\n"
        f"Kill Switch   {kill_str}"
        f"{milestone_str}"
    ).strip()


def format_stats_message(account: dict, stats: dict) -> str:
    total = stats.get("total_closed", 0)
    wins = stats.get("wins", 0)
    losses = stats.get("losses", 0)
    wr = stats.get("win_rate")
    today_pnl = stats.get("today_pnl_usd", 0.0)
    best = stats.get("best_trade_usd", 0.0)
    best_inst = stats.get("best_trade_inst", "")
    worst = stats.get("worst_trade_usd", 0.0)
    worst_inst = stats.get("worst_trade_inst", "")
    today_closed = stats.get("today_closed", 0)

    # All P&L figures come directly from OANDA (single source of truth).
    total_pnl = stats.get("total_pnl_usd", 0.0)
    avg_pnl = stats.get("avg_pnl_usd", 0.0)

    # Win rate bar
    if wr is not None:
        wr_bar = _bar(wr * 100, 100, 10)
        wr_str = f"{wr:.1%}"
        kelly_note = "Kelly sizing active — live calibrated win rate ✓"
    else:
        remaining = max(0, 30 - total)
        wr_bar = "░" * 10
        wr_str = f"~55% est.  ({remaining} more trades to calibrate)"
        kelly_note = "Using estimated 55% until 30 trades completed"

    daily = account.get("daily_status", {})
    target = daily.get("profit_target_usd", 0)
    loss_lim = daily.get("loss_limit_usd", 0)
    locked = daily.get("trading_locked", False)
    acc_unrealized = float(account.get("unrealized_pnl", 0))

    # Use DB today_pnl as the authoritative source for progress
    # (rm._daily_pnl_usd resets to 0 on restart, so progress_pct from daily_status is unreliable)
    progress = round((today_pnl / target * 100) if target > 0 else 0, 1)
    prog_bar = _bar(max(0, progress), 100, 10)
    # Avoid Python's -0.0 formatting as "-0%" — show explicit sign only when meaningful
    if progress < -0.5:
        prog_str = f"{int(progress)}%"   # e.g. "-6%"
    elif progress >= 100:
        prog_str = "🎯 100%"
    else:
        prog_str = f"{int(max(0, progress))}%"   # "0%" through "99%"
    daily_loss_used = max(0, -today_pnl)
    dl_bar = _bar(min(daily_loss_used, loss_lim) if loss_lim > 0 else 0, max(loss_lim, 1), 10)

    # Edge summary
    if total >= 5:
        if wins > losses:
            edge_note = f"  ·  Positive edge: {wins}W / {losses}L"
        elif losses > wins:
            edge_note = f"  ·  ⚠️ Review entries: {losses}L / {wins}W"
        else:
            edge_note = f"  ·  Break-even: {wins}W / {losses}L"
    else:
        edge_note = "  ·  Building sample"

    return (
        f"📊 <b>ForexMind  ·  Performance</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Today</b>\n"
        f"{_pnl_icon(today_pnl)} P&amp;L         <b>${today_pnl:+,.2f}</b>  /  ${target:,.2f} target\n"
        f"Progress    {prog_bar}  {prog_str}\n"
        f"Closed      {today_closed} trade{'s' if today_closed != 1 else ''}\n"
        f"Loss Limit  {dl_bar}  -${loss_lim:,.2f}  "
        f"{'🔴 LOCKED' if locked else '🟢 OK'}\n"
        f"\n"
        f"─────────────────────────\n"
        f"<b>All-Time</b>\n"
        f"Closed      {total} trades{edge_note}\n"
        f"Win Rate    {wr_bar}  <b>{wr_str}</b>\n"
        f"{_pnl_icon(total_pnl)} Total P&amp;L  <b>${total_pnl:+,.2f}</b>\n"
        f"Per Trade   ${avg_pnl:+,.2f}\n"
        f"Best        ${best:+,.2f}  {best_inst}\n"
        f"Worst       ${worst:+,.2f}  {worst_inst}\n"
        f"\n"
        f"Open Trades {account.get('open_trades', 0)}  ·  "
        f"Account P&amp;L {_pnl_icon(acc_unrealized)} ${acc_unrealized:+,.2f}\n"
        f"\n"
        f"<i>{kelly_note}</i>"
    ).strip()


# ── Context keys ──────────────────────────────────────────────────────────────

AUTOTRADE_KEY = "autotrade_enabled"
PENDING_TRADE_KEY = "pending_trade"


# ── Handlers ──────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cfg = get_settings()
    if cfg.app.paper_trading:
        mode_line = "🧪 <b>Paper mode</b>  ·  No real capital at risk"
    else:
        mode_line = "⚠️ <b>LIVE mode</b>  ·  Real capital active"
    await update.message.reply_html(
        f"⚡ <b>ForexMind  ·  AI Trading Agent</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{mode_line}\n\n"
        f"<b>What I do</b>\n"
        f"• Scan 8 pairs every 15 min on M15 + H1\n"
        f"• 4-strategy ensemble: Rules + LightGBM + LSTM + RL\n"
        f"• Auto-trade at ≥65% confidence with 2:1 R:R minimum\n"
        f"• Split-TP: close 50% at 2R, move SL to breakeven\n"
        f"• 90-second monitor with automatic profit close\n\n"
        f"<b>Quick Start</b>\n"
        f"/signals        — Top setups right now\n"
        f"/trade EUR/USD  — Signal + one-tap trade\n"
        f"/account        — Balance + open P&amp;L\n"
        f"/autotrade on   — Enable auto-trading\n"
        f"/help           — Full command reference\n\n"
        f"<i>Edge: 52–58% win rate at 2:1 R:R = profitable compounding system.</i>"
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_html(
        "⚡ <b>ForexMind  ·  Command Reference</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "\n"
        "📡 <b>Signals</b>\n"
        "/signal EUR/USD    Full H1 signal + technicals\n"
        "/signals           Best pairs for this session\n"
        "/trade EUR/USD     Signal + one-tap trade confirm\n"
        "\n"
        "🤖 <b>Auto-Trade</b>\n"
        "/autotrade on      Auto-place at ≥65% conf + 2:1 R:R\n"
        "/autotrade off     Disable auto-trading\n"
        "\n"
        "📋 <b>Positions</b>\n"
        "/trades            Open positions + ❌ close buttons\n"
        "/close EUR/USD     Close all positions on a pair\n"
        "/close all         Close every open trade\n"
        "/monitor EUR/USD 20  Auto-close when P&amp;L ≥ $20\n"
        "/monitor all 10    Watch all trades, close each at $10\n"
        "/monitor stop      Cancel all monitoring\n"
        "/monitor status    Show active monitors\n"
        "\n"
        "📊 <b>Performance</b>\n"
        "/stats             Win rate, P&amp;L, trade history\n"
        "/account           Balance, margin, kill switch\n"
        "/risk              Live risk exposure dashboard\n"
        "/backtest EUR/USD  1-year H1 backtest results\n"
        "\n"
        "🌍 <b>Market</b>\n"
        "/sessions          Session clock + best pairs now\n"
        "\n"
        "<i>Type anything else to chat with the Claude AI agent.</i>"
    )


async def cmd_sessions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    status = get_session_status()
    pairs = best_pairs_for_session()[:6]

    if status.is_weekend:
        await update.message.reply_html(
            "🚫 <b>Weekend</b>  ·  Forex markets closed\n\n"
            "Reopen  <b>Sunday 21:00 UTC</b>\n"
            "<i>Use downtime to review setups and plan the week.</i>"
        )
        return

    session_defs = [
        ("Sydney",   "🌏", "21:00–06:00"),
        ("Tokyo",    "🗼", "00:00–09:00"),
        ("London",   "🏦", "07:00–16:00"),
        ("New York", "🗽", "12:00–21:00"),
    ]
    lines = []
    for name, icon, hours in session_defs:
        dot = "🟢" if name in status.active_sessions else "⚫"
        lines.append(f"{dot} {icon} {name:<9}  {hours} UTC")

    score = status.session_score
    score_bar = _bar(score * 100, 100, 10)

    msg = (
        f"🌍 <b>Market Sessions</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        + "\n".join(lines) + "\n"
        f"\n"
        f"Liquidity   {score_bar}  {score:.0%}\n"
    )

    if status.active_overlaps:
        msg += (
            f"\n🔥 <b>OVERLAP ACTIVE</b>\n"
            f"   {', '.join(status.active_overlaps)}\n"
            f"   Prime window — highest volume &amp; tightest spreads\n"
        )

    if pairs:
        rec = "  ".join(f"<code>{p.replace('_', '/')}</code>" for p in pairs)
        msg += f"\n<b>Best Pairs Now</b>\n{rec}"

    await update.message.reply_html(msg)


def _low_liquidity_message(score: float) -> str:
    score_bar = _bar(score * 100, 100, 8)
    return (
        f"⏸ <b>Low Liquidity — Signals Unavailable</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Liquidity  {score_bar}  {score:.0%}\n\n"
        f"Signals and trading are only available at <b>≥40% liquidity</b> "
        f"(active session).\n\n"
        f"Next windows:\n"
        f"  🗼 Tokyo      00:00–09:00 UTC\n"
        f"  🏦 London     07:00–16:00 UTC\n"
        f"  🗽 New York   12:00–21:00 UTC\n\n"
        f"<i>Use /sessions to see current status.</i>"
    )


async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /signal EUR/USD — shows signal without trade button."""
    args = context.args
    if not args:
        await update.message.reply_text("Usage: /signal EUR/USD")
        return

    status = get_session_status()
    if status.session_score < 0.4:
        await update.message.reply_html(_low_liquidity_message(status.session_score))
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
        await update.message.reply_text("Usage: /trade EUR/USD [force]")
        return

    # Check for 'force' flag to bypass liquidity gate
    force = len(args) > 1 and args[-1].lower() == "force"
    status = get_session_status()
    if status.session_score < 0.4 and not force:
        await update.message.reply_html(_low_liquidity_message(status.session_score))
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
        context.user_data["trade_force"] = force

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
        status_line = "🤖 ON  ·  Placing trades automatically" if current else "⏸ OFF  ·  Signals only"
        await update.message.reply_html(
            f"<b>Auto-Trade Status</b>\n\n"
            f"{status_line}\n\n"
            f"<code>/autotrade on</code>   — enable\n"
            f"<code>/autotrade off</code>  — disable\n\n"
            f"<i>When ON: confidence ≥ 65%, R:R ≥ 1.8, active session required.</i>"
        )
        return

    enable = args[0].lower() == "on"
    context.bot_data[AUTOTRADE_KEY] = enable

    if enable:
        from forexmind.scheduler import AUTO_TRADE_CONFIDENCE, MIN_RR
        mode_str = "🧪 Paper — no real money at risk" if get_settings().app.paper_trading else "⚠️ LIVE TRADING"
        await update.message.reply_html(
            f"🤖 <b>Auto-Trading ENABLED</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Trades placed automatically when:\n"
            f"  ✅ Confidence ≥ {AUTO_TRADE_CONFIDENCE:.0f}%  (3+ strategies agree)\n"
            f"  ✅ R:R ≥ {MIN_RR}:1\n"
            f"  ✅ Active London or NY session\n\n"
            f"Split-TP: 50% at 2R, SL → breakeven\n"
            f"Mode: {mode_str}\n\n"
            f"<i>You receive a notification for every trade placed.</i>\n"
            f"<i>Use /autotrade off to disable.</i>"
        )
    else:
        await update.message.reply_html(
            f"⏸ <b>Auto-Trading DISABLED</b>\n\n"
            f"Signal alerts continue — no trades placed automatically.\n\n"
            f"Use <code>/trade EUR/USD</code> to manually confirm individual setups."
        )


async def cmd_trades(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show open trades with close buttons."""
    msg = await update.message.reply_text("📋 Fetching open trades...")
    try:
        from forexmind.data.oanda_client import get_oanda_client
        client = get_oanda_client()
        trades = await client.get_open_trades()

        if not trades:
            await msg.edit_text(
                "📭 <b>No open positions</b>\n\n"
                "<i>Use /signals to find the next setup.</i>",
                parse_mode=ParseMode.HTML,
            )
            return

        total_pnl = sum(float(t.get("unrealizedPL", 0)) for t in trades)
        lines = [
            f"📋 <b>Open Positions</b>  ·  {len(trades)} active  "
            f"({_pnl_icon(total_pnl)} ${total_pnl:+,.2f})\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━"
        ]
        keyboard_rows = []
        for t in trades:
            trade_id = t.get("id", "?")
            instrument = t.get("instrument", "?")
            units = float(t.get("currentUnits", t.get("initialUnits", 0)))
            direction = "BUY" if units > 0 else "SELL"
            entry = float(t.get("price", 0))
            pnl = float(t.get("unrealizedPL", 0))
            sl = float((t.get("stopLossOrder") or {}).get("price", 0))
            tp = float((t.get("takeProfitOrder") or {}).get("price", 0))
            sl_str = f"  SL <code>{sl}</code>" if sl else ""
            tp_str = f"  TP <code>{tp}</code>" if tp else ""
            dir_emoji = "🟢" if direction == "BUY" else "🔴"

            lines.append(
                f"\n{_pnl_icon(pnl)} {dir_emoji} <b>{instrument.replace('_', '/')}</b>  "
                f"{direction}  #{trade_id}\n"
                f"   Entry <code>{entry}</code>{sl_str}{tp_str}\n"
                f"   {abs(int(units)):,} units  ·  P&amp;L: <b>${pnl:+,.2f}</b>"
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

        from forexmind.risk.manager import get_risk_manager
        from forexmind.data.trade_repo import close_trade_record as db_close_trade
        results = []
        for t in to_close:
            trade_id = t.get("id")
            instrument = t.get("instrument", "?")
            units = float(t.get("currentUnits", 0))
            direction = "BUY" if units > 0 else "SELL"
            pnl = float(t.get("unrealizedPL", 0))
            result = await client.close_trade(trade_id)
            if result.success:
                filled = result.filled_price or float(t.get("price", 0))
                await get_risk_manager().close_trade(str(trade_id), filled)
                await db_close_trade(trade_id, filled, pnl)
                results.append(f"✅ Closed {instrument.replace('_', '/')} {direction} @ {filled} | P&L: ${pnl:+,.2f}")
            else:
                results.append(f"❌ Failed to close #{trade_id}: {result.message}")

        await msg.edit_text("\n".join(results), parse_mode=ParseMode.HTML)
        # Notify if daily target/limit hit
        from forexmind.data.oanda_client import get_oanda_client as _oc2
        try:
            _acc2 = await _oc2().get_account()
            await _notify_daily_status(context, update.effective_chat.id, _acc2.balance)
        except Exception:
            pass

    except Exception as e:
        await msg.edit_text(f"❌ Error: {str(e)[:200]}")


async def _notify_daily_status(context_or_bot, chat_id: int, balance: float) -> None:
    """Send a notification if daily profit target or loss limit was just hit."""
    try:
        from forexmind.risk.manager import get_risk_manager
        daily = get_risk_manager().daily_status(balance)
        if daily["trading_locked"]:
            bot = context_or_bot.bot if hasattr(context_or_bot, "bot") else context_or_bot
            await bot.send_message(
                chat_id=chat_id,
                text=daily["status"] + f"\n\nToday's P&L: ${daily['daily_pnl_usd']:+,.2f}",
            )
    except Exception:
        pass


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show win rate, P&L and trade stats from OANDA."""
    msg = await update.message.reply_text("📊 Fetching stats...")
    try:
        from forexmind.data.trade_repo import get_stats as get_oanda_stats
        from forexmind.risk.manager import get_risk_manager

        oanda_stats = await get_oanda_stats()
        oanda_acc = oanda_stats.get("_account") or {}
        balance = float(oanda_acc.get("balance", 100_000))
        rm = get_risk_manager()
        daily = rm.daily_status(balance)
        account = {
            "balance": balance,
            "unrealized_pnl": oanda_acc.get("unrealized_pnl", 0),
            "open_trades": oanda_acc.get("open_trade_count", 0),
            "daily_status": daily,
        }
        text = format_stats_message(account, oanda_stats)
        await msg.edit_text(text, parse_mode=ParseMode.HTML)

    except Exception as e:
        await msg.edit_text(f"❌ Error: {str(e)[:200]}")


async def cmd_signals(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Get signals for all recommended session pairs."""
    msg = await update.message.reply_text("📡 Scanning session pairs...")
    try:
        from forexmind.agents.tools import _get_signal
        force = context.args and context.args[0].lower() == "force"
        status = get_session_status()
        if status.is_weekend and not force:
            await msg.edit_text(
                "🚫 <b>Weekend</b>  ·  Markets closed\n\n"
                "Reopen Sunday 21:00 UTC.",
                parse_mode=ParseMode.HTML,
            )
            return
        if status.session_score < 0.4 and not force:
            await msg.edit_text(
                _low_liquidity_message(status.session_score),
                parse_mode=ParseMode.HTML,
            )
            return
        pairs = best_pairs_for_session()[:6]
        if not pairs:
            # Force mode: scan major pairs even with no session recommendation
            if force:
                pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CAD", "AUD/USD"]
            else:
                await msg.edit_text(
                    "⚠️ <b>No active session</b>\n\n"
                    "No recommended pairs. Try during London or New York session.",
                    parse_mode=ParseMode.HTML,
                )
                return

        # H1 trend + M15 entry in parallel
        h1_tasks = [_get_signal(p, "H1", 300) for p in pairs]
        m15_tasks = [_get_signal(p, "M15", 200) for p in pairs]
        h1_results, m15_results = await asyncio.gather(
            asyncio.gather(*h1_tasks),
            asyncio.gather(*m15_tasks),
        )

        # Header line
        if status.active_overlaps:
            session_tag = f"🔥 {status.active_overlaps[0]}"
        elif status.active_sessions:
            session_tag = f"📡 {status.active_sessions[0]}"
        else:
            session_tag = "📡 Active"

        score_bar = _bar(status.session_score * 100, 100, 6)
        lines = [
            f"📡 <b>Session Signals</b>  ·  {session_tag}\n"
            f"Liquidity  {score_bar}  {status.session_score:.0%}  "
            f"·  <i>H1 trend + M15 entry</i>\n"
        ]

        any_aligned = False
        any_caution = False

        for pair, h1_res, m15_res in zip(pairs, h1_results, m15_results):
            h1_data = json.loads(h1_res)
            h1 = h1_data.get("signal", {})
            m15 = json.loads(m15_res).get("signal", {})

            action = h1.get("action", "HOLD")
            conf = float(h1.get("confidence", 0))
            rr = h1.get("risk_reward", "—")
            m15_action = m15.get("action", "HOLD")
            aligned = (action == m15_action and action != "HOLD")

            caution = signal_caution_warning(h1_data)

            if aligned:
                any_aligned = True
            if caution:
                any_caution = True

            if action == "HOLD" or conf < 42:
                dir_emoji = "⚪"
                action_str = "HOLD"
                clabel = "—"
            else:
                dir_emoji = "🟢" if action == "BUY" else "🔴"
                action_str = action
                clabel = confidence_label(conf)

            conf_bar_str = _bar(conf, 100, 6)
            align_icon = "✅" if aligned else "  "
            caution_icon = " ⚠️" if caution else ""
            rr_str = f"  R:R {rr}" if action_str != "HOLD" else ""

            lines.append(
                f"{dir_emoji} <b>{pair.replace('_', '/')}</b>  "
                f"{conf_bar_str} {conf:.0f}%  {clabel}\n"
                f"   {align_icon} {action_str}{rr_str}{caution_icon}"
            )

        # Footer legend — only show icons that are actually present in this scan
        footer_parts = []
        if any_aligned:
            footer_parts.append("✅ H1 + M15 aligned")
        if any_caution:
            footer_parts.append("⚠️ ATR/news caution")
        if footer_parts:
            lines.append("\n<i>" + "  ·  ".join(footer_parts) + "</i>")
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
            f"📈 <b>Backtest: {pair.replace('_', '/')} H1 (2024)</b>\n"
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


async def cmd_risk(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show live risk exposure: drawdown, daily loss limit, open positions."""
    msg = await update.message.reply_text("⏳ Calculating exposure...")
    try:
        from forexmind.data.oanda_client import get_oanda_client
        from forexmind.risk.manager import get_risk_manager

        client = get_oanda_client()
        acc = await client.get_account()
        open_trades = await client.get_open_trades()
        rm = get_risk_manager()
        balance = acc.balance
        rm.update_peak(balance)

        daily = rm.daily_status(balance)
        daily_pnl = float(daily.get("daily_pnl_usd", 0))
        daily_loss_lim = float(daily.get("loss_limit_usd", 100))
        locked = daily.get("trading_locked", False)

        # Drawdown from peak
        drawdown_pct = 0.0
        peak = getattr(rm, "_peak_balance", None)
        if peak and balance < peak:
            drawdown_pct = (peak - balance) / peak * 100

        # Open exposure
        total_open_pnl = sum(float(t.get("unrealizedPL", 0)) for t in open_trades)
        dd_bar = _bar(min(drawdown_pct, 20), 20, 10)
        daily_loss_used = max(0, -daily_pnl)
        dl_bar = _bar(min(daily_loss_used, daily_loss_lim), max(daily_loss_lim, 1), 10)

        position_lines = []
        for t in open_trades:
            inst = t.get("instrument", "?").replace("_", "/")
            units = float(t.get("currentUnits", 0))
            direction = "BUY" if units > 0 else "SELL"
            pnl = float(t.get("unrealizedPL", 0))
            entry = float(t.get("price", 0))
            sl = float((t.get("stopLossOrder") or {}).get("price", 0))
            sl_str = f"  SL {sl}" if sl else "  no SL ⚠️"
            position_lines.append(
                f"  {_pnl_icon(pnl)} {inst} {direction}  <b>${pnl:+,.2f}</b>"
                f"  (entry {entry}{sl_str})"
            )

        positions_text = "\n".join(position_lines) if position_lines else "  No open positions"
        lock_line = "🔴 <b>TRADING LOCKED</b>" if locked else "🟢 Within limits"

        await msg.edit_text(
            f"🎚 <b>Risk Dashboard</b>  ·  {len(open_trades)} open\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Balance       <b>${balance:,.2f}</b>\n"
            f"Open P&amp;L     {_pnl_icon(total_open_pnl)} ${total_open_pnl:+,.2f}\n"
            f"\n"
            f"📉 <b>Drawdown</b>  (max 20%)\n"
            f"{dd_bar}  {drawdown_pct:.1f}%\n"
            f"\n"
            f"🚫 <b>Daily Loss Limit</b>  (${daily_loss_lim:,.2f})\n"
            f"{dl_bar}  ${daily_loss_used:,.2f} used\n"
            f"{lock_line}\n"
            f"\n"
            f"<b>Positions</b>\n"
            f"{positions_text}",
            parse_mode=ParseMode.HTML,
        )
    except Exception as e:
        await msg.edit_text(f"❌ {str(e)[:200]}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle free-text messages by forwarding to Claude agent."""
    user_text = update.message.text
    msg = await update.message.reply_text("🤔 Thinking...")
    try:
        from forexmind.agents.claude_agent import get_agent
        from forexmind.agents.tools import get_pending_trade
        agent = get_agent()
        response = await agent.chat(user_text)
        if len(response) > 4000:
            response = response[:3997] + "..."

        # Check if agent proposed a trade — show confirm/cancel buttons
        pending = get_pending_trade()
        if pending:
            pair = pending["instrument"].replace("_", "/")
            direction = pending["direction"]
            entry = pending["entry"]
            sl = pending["stop_loss"]
            tp = pending["take_profit"]
            units = pending["units"]
            rr = pending["rr"]
            risk_usd = pending["risk_usd"]
            trade_summary = (
                f"\n\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"📋 Proposed Trade\n"
                f"{'🟢' if direction == 'BUY' else '🔴'} {direction} {pair}\n"
                f"Entry:  {entry}  |  Units: {units:,}\n"
                f"SL: {sl}  |  TP: {tp}\n"
                f"R:R: {rr}:1  |  Risk: ${risk_usd:.2f}"
            )
            keyboard = InlineKeyboardMarkup([[
                InlineKeyboardButton(f"✅ Confirm {direction} {pair}", callback_data="chat_confirm_trade"),
                InlineKeyboardButton("❌ Cancel", callback_data="chat_cancel_trade"),
            ]])
            await msg.edit_text(response + trade_summary, reply_markup=keyboard)
        else:
            # Send as plain text — agent responses contain raw < > characters
            await msg.edit_text(response)
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
            # Normalise confidence from 0–100 (signal) → 0–1 (risk manager)
            confidence = float(sig.get("confidence", 0)) / 100.0

            client = get_oanda_client()
            acc = await client.get_account()
            rm = get_risk_manager()
            rm.update_peak(acc.balance)

            # Reconcile cached open trades with live OANDA state before risk gating.
            try:
                open_trades = await client.get_open_trades()
                await rm.sync_open_trades({str(t.get('id', '')) for t in open_trades if t.get('id')})
            except Exception as sync_err:
                log.warning(f"/trade sync skipped (non-fatal): {sync_err}")

            force_trade = context.user_data.pop("trade_force", False)
            proposal = rm.calculate_risk(
                instrument=pair,
                direction=direction,
                entry=entry,
                atr=atr,
                account_balance=acc.balance,
                confidence=confidence,
                skip_correlation=force_trade,
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

            if "error" in result or result.get("success") is False:
                err_msg = result.get("error") or result.get("message") or "Unknown order failure"
                await query.edit_message_text(
                    f"❌ <b>Order failed</b>\n{err_msg[:500]}",
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

            trade_id = result.get("trade_id")
            if not trade_id:
                await query.edit_message_text(
                    "⚠️ <b>Order may not have filled</b>\nNo trade ID returned — check /trades.",
                    parse_mode=ParseMode.HTML,
                )
                return

            cfg = get_settings()
            mode = "Paper" if cfg.app.paper_trading else "⚠️ LIVE"
            filled = float(result.get("filled_price", 0)) or entry

            # Register trade with risk manager for correlation tracking
            from forexmind.risk.manager import OpenTrade as _OpenTrade
            await rm.register_trade(_OpenTrade(
                trade_id=str(trade_id),
                instrument=pair,
                direction=direction,
                entry_price=filled,
                stop_loss=stop_loss,
                take_profit=take_profit,
                units=proposal.units,
            ))

            # Register for split TP (close 50% at 2:1, move SL to BE)
            register_split_tp(
                trade_id=str(trade_id),
                instrument=pair,
                direction=direction,
                entry=filled,
                sl=stop_loss,
                units=proposal.units,
                chat_id=query.message.chat_id,
            )

            from forexmind.utils.helpers import format_price as _fp
            dir_emoji = "🟢" if direction == "BUY" else "🔴"
            await query.edit_message_text(
                f"{dir_emoji} <b>Trade Placed  [{mode}]</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Pair       {pair.replace('_', '/')}\n"
                f"Direction  {direction}\n"
                f"Units      {proposal.units:,}\n"
                f"Filled @   <code>{_fp(filled, pair)}</code>\n"
                f"Stop Loss  <code>{_fp(stop_loss, pair)}</code>\n"
                f"TP         <code>{_fp(take_profit, pair)}</code>\n"
                f"Risk       {proposal.risk_pct:.1f}%  (${proposal.risk_usd:.2f})\n"
                f"R:R        {proposal.risk_reward_ratio:.1f}:1\n"
                f"Trade ID   <code>{trade_id}</code>\n\n"
                f"🎯 Split-TP active  ·  50% at 2R  ·  SL → breakeven",
                parse_mode=ParseMode.HTML,
            )
            context.user_data.pop(PENDING_TRADE_KEY, None)

        except Exception as e:
            await query.edit_message_text(f"❌ Trade error: {str(e)[:300]}")

    # ── Cancel trade ──────────────────────────────────────────────────────────
    elif data == "cancel_trade":
        context.user_data.pop(PENDING_TRADE_KEY, None)
        await query.edit_message_text("❌ Trade cancelled.")

    # ── Chat-proposed trade confirm/cancel ────────────────────────────────────
    elif data == "chat_confirm_trade":
        from forexmind.agents.tools import get_pending_trade, clear_pending_trade, _place_trade
        from forexmind.risk.manager import get_risk_manager as _get_rm, OpenTrade as _OpenTrade
        pending = get_pending_trade()
        if not pending:
            await query.edit_message_text("⚠️ Trade proposal expired. Please request it again.")
            return
        await query.edit_message_text(f"⏳ Placing {pending['direction']} {pending['instrument'].replace('_', '/')}...")
        try:
            result_str = await _place_trade(
                instrument=pending["instrument"],
                direction=pending["direction"],
                units=pending["units"],
                stop_loss=pending["stop_loss"],
                take_profit=pending["take_profit"],
            )
            result = json.loads(result_str)
            clear_pending_trade()
            if "error" in result or result.get("success") is False:
                err_msg = result.get("error") or result.get("message") or "Unknown order failure"
                await query.edit_message_text(f"❌ Order failed: {err_msg[:500]}")
                return

            trade_id = result.get("trade_id")
            if not trade_id:
                await query.edit_message_text(
                    "⚠️ <b>Order may not have filled</b>\nNo trade ID returned — check /trades.",
                    parse_mode=ParseMode.HTML,
                )
                return

            cfg = get_settings()
            mode = "Paper" if cfg.app.paper_trading else "⚠️ LIVE"
            filled = float(result.get("filled_price", 0)) or pending["entry"]

            # Register trade with risk manager for correlation tracking
            await _get_rm().register_trade(_OpenTrade(
                trade_id=str(trade_id),
                instrument=pending["instrument"],
                direction=pending["direction"],
                entry_price=filled,
                stop_loss=float(pending["stop_loss"]),
                take_profit=float(pending["take_profit"]),
                units=int(pending["units"]),
            ))

            # Register for split TP
            register_split_tp(
                trade_id=str(trade_id),
                instrument=pending["instrument"],
                direction=pending["direction"],
                entry=filled,
                sl=float(pending["stop_loss"]),
                units=int(pending["units"]),
                chat_id=query.message.chat_id,
            )

            _d = pending["direction"]
            _de = "🟢" if _d == "BUY" else "🔴"
            _inst = pending["instrument"]
            from forexmind.utils.helpers import format_price as _fp2
            await query.edit_message_text(
                f"{_de} <b>Trade Placed  [{mode}]</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Pair       {_inst.replace('_', '/')}\n"
                f"Direction  {_d}\n"
                f"Units      {pending['units']:,}\n"
                f"Filled @   <code>{_fp2(filled, _inst)}</code>\n"
                f"Stop Loss  <code>{_fp2(float(pending['stop_loss']), _inst)}</code>\n"
                f"TP         <code>{_fp2(float(pending['take_profit']), _inst)}</code>\n"
                f"R:R        {pending['rr']}:1\n"
                f"Risk       ${pending['risk_usd']:.2f}\n"
                f"Trade ID   <code>{trade_id}</code>\n\n"
                f"🎯 Split-TP active  ·  50% at 2R  ·  SL → breakeven",
                parse_mode=ParseMode.HTML,
            )
        except Exception as e:
            clear_pending_trade()
            await query.edit_message_text(f"❌ Trade error: {str(e)[:300]}")

    elif data == "chat_cancel_trade":
        from forexmind.agents.tools import clear_pending_trade
        clear_pending_trade()
        await query.edit_message_text("❌ Trade cancelled.")

    # ── Close individual trade ─────────────────────────────────────────────────
    elif data.startswith("close_trade_"):
        parts = data[len("close_trade_"):].split("_", 1)
        trade_id = parts[0]
        instrument = parts[1] if len(parts) > 1 else "?"
        await query.edit_message_text(f"⏳ Closing {instrument.replace('_', '/')}...")
        try:
            from forexmind.data.oanda_client import get_oanda_client
            from forexmind.risk.manager import get_risk_manager
            from forexmind.data.trade_repo import close_trade_record as db_close_trade
            client = get_oanda_client()
            # Get P&L before closing (unrealizedPL from OANDA)
            open_trades = await client.get_open_trades()
            trade_info = next((t for t in open_trades if str(t.get("id")) == str(trade_id)), {})
            pnl = float(trade_info.get("unrealizedPL", 0))
            result = await client.close_trade(trade_id)
            if result.success:
                filled = result.filled_price
                await get_risk_manager().close_trade(str(trade_id), filled or 0.0)
                await db_close_trade(trade_id, filled or 0.0, pnl)
                _split_tp_trades.pop(str(trade_id), None)  # clean up split TP state
                _save_monitors()
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
            from forexmind.risk.manager import get_risk_manager
            from forexmind.data.trade_repo import close_trade_record as db_close_trade
            lines = ["<b>Closing all trades:</b>"]
            for t in trades:
                trade_id = t.get("id")
                instrument = t.get("instrument", "?")
                pnl = float(t.get("unrealizedPL", 0))
                result = await client.close_trade(trade_id)
                if result.success:
                    filled = result.filled_price or 0.0
                    await get_risk_manager().close_trade(str(trade_id), filled)
                    await db_close_trade(trade_id, filled, pnl)
                    _split_tp_trades.pop(str(trade_id), None)
                    _save_monitors()
                    lines.append(f"✅ {instrument.replace('_', '/')} @ {filled} | P&L: ${pnl:+,.2f}")
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


# ── Trade Monitor ─────────────────────────────────────────────────────────────

# In-memory store: chat_id → {instrument → min_profit_usd}
_monitor_targets: dict[int, dict[str, float]] = {}

MONITOR_INTERVAL = 30  # 30 seconds — tight enough to catch 2R spikes before they retrace

# Persist monitors to disk so they survive service restarts
import json as _json
from pathlib import Path as _Path

_MONITOR_PERSIST_FILE = _Path("/var/lib/forexmind/data/monitors.json") if _Path("/var/lib/forexmind").exists() \
    else _Path(__file__).resolve().parent.parent / "data" / "monitors.json"


def _save_monitors() -> None:
    """Persist active monitors and split-TP state to disk."""
    try:
        _MONITOR_PERSIST_FILE.parent.mkdir(parents=True, exist_ok=True)
        _MONITOR_PERSIST_FILE.write_text(_json.dumps({
            "monitors": {str(k): v for k, v in _monitor_targets.items()},
            "split_tp": _split_tp_trades,
        }))
    except Exception as e:
        log.warning(f"Could not save monitors: {e}")


def _load_monitors() -> None:
    """Restore monitors and split-TP state from disk on startup."""
    try:
        if _MONITOR_PERSIST_FILE.exists():
            data = _json.loads(_MONITOR_PERSIST_FILE.read_text())
            if "monitors" in data:
                # New format: {"monitors": {...}, "split_tp": {...}}
                for chat_id_str, targets in data["monitors"].items():
                    _monitor_targets[int(chat_id_str)] = targets
                for trade_id, split_data in data.get("split_tp", {}).items():
                    _split_tp_trades[trade_id] = split_data
            else:
                # Old format: top-level keys are chat_ids (monitor targets only)
                for chat_id_str, targets in data.items():
                    _monitor_targets[int(chat_id_str)] = targets
            if _monitor_targets:
                log.info(f"Restored {len(_monitor_targets)} monitor(s) from disk")
            if _split_tp_trades:
                log.info(f"Restored {len(_split_tp_trades)} split-TP entr{'y' if len(_split_tp_trades) == 1 else 'ies'} from disk")
    except Exception as e:
        log.warning(f"Could not load monitors: {e}")


# Split TP tracking: oanda_trade_id → metadata for 50%-at-2R strategy
# Populated when a trade is placed; cleared when half-closed.
# Must be declared before _load_monitors() which populates it on startup.
_split_tp_trades: dict[str, dict] = {}

_load_monitors()


def register_split_tp(
    trade_id: str,
    instrument: str,
    direction: str,
    entry: float,
    sl: float,
    units: int,
    chat_id: int,
) -> None:
    """
    Register a trade for the split-TP strategy.
    Called immediately after a market order fills.

    Strategy:
      • At TP1 (2:1 R:R): close 50%, move SL to breakeven
      • Remaining half rides to TP2 (set on order) or trailing stop
    """
    if not trade_id or trade_id in ("pending", "?", ""):
        return
    _split_tp_trades[str(trade_id)] = {
        "instrument": instrument,
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "initial_units": units,
        "half_closed": False,
        "chat_id": chat_id,
    }
    _save_monitors()
    log.info(f"Split-TP registered for trade {trade_id} ({instrument} {direction})")


def seed_split_tp_from_oanda(oanda_trades: list, chat_id: int) -> None:
    """
    Seed _split_tp_trades from live OANDA positions on startup.
    Called by the scheduler after it fetches open trades, so split-TP
    protection survives service restarts.

    Skips trades already registered (from the persisted file).
    Detects half-closed trades by comparing currentUnits vs initialUnits.
    """
    seeded = 0
    for t in oanda_trades:
        trade_id = str(t.get("id", ""))
        if not trade_id or trade_id in _split_tp_trades:
            continue
        inst = t.get("instrument", "")
        current_units = int(float(t.get("currentUnits", 0)))
        initial_units = int(float(t.get("initialUnits", current_units)))
        direction = "BUY" if current_units > 0 else "SELL"
        entry = float(t.get("price", 0))
        sl = float((t.get("stopLossOrder") or {}).get("price", 0))
        if not inst or not entry or not sl:
            continue
        # If currentUnits < 75% of initialUnits, the first half was already closed
        half_closed = abs(current_units) < abs(initial_units) * 0.75
        _split_tp_trades[trade_id] = {
            "instrument": inst,
            "direction": direction,
            "entry": entry,
            "sl": sl,
            "initial_units": abs(initial_units),
            "half_closed": half_closed,
            "chat_id": chat_id,
        }
        seeded += 1
    if seeded:
        _save_monitors()
        log.info(f"Seeded {seeded} split-TP entr{'y' if seeded == 1 else 'ies'} from OANDA startup data")


async def _monitor_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    JobQueue callback — runs every 15 minutes per chat.
    Checks each monitored trade; closes it if P&L >= target.
    """
    chat_id: int = context.job.data["chat_id"]
    targets: dict[str, float] = _monitor_targets.get(chat_id, {})
    if not targets:
        return

    try:
        from forexmind.data.oanda_client import get_oanda_client
        from forexmind.risk.manager import get_risk_manager
        from forexmind.data.trade_repo import close_trade_record as db_close_trade

        client = get_oanda_client()
        open_trades = await client.get_open_trades()

        # Build set of instruments that still have an open trade
        open_instruments: set[str] = {t.get("instrument", "") for t in open_trades}

        # ── Purge stale targets for already-closed trades ─────────────────────
        # Fetch OANDA's recently closed trades once so we can record the P&L.
        try:
            recently_closed = await client.get_recently_closed_trades(count=50)
            closed_by_id = {str(t.get("id", "")): t for t in recently_closed}
        except Exception:
            recently_closed = []
            closed_by_id = {}

        stale = [
            inst for inst in list(targets)
            if inst not in open_instruments and inst.replace("/", "_") not in open_instruments
        ]
        for inst in stale:
            targets.pop(inst, None)
            pair = inst.replace("_", "/")
            # Find the closed OANDA trade for this instrument and record it in the DB
            closed_trade = next(
                (t for t in recently_closed
                 if t.get("instrument", "") in (inst, inst.replace("/", "_"))),
                None,
            )
            close_msg = f"📭 Monitor: {pair} trade already closed (TP/SL hit or manual). Removed from watch list."
            if closed_trade:
                trade_id = str(closed_trade.get("id", ""))
                exit_price = float(closed_trade.get("averageClosePrice", 0) or 0)
                realised_pnl = float(closed_trade.get("realizedPL", 0) or 0)
                if realised_pnl != 0 or exit_price != 0:
                    try:
                        from forexmind.data.trade_repo import close_trade_record as db_close_trade
                        from forexmind.risk.manager import get_risk_manager
                        await db_close_trade(trade_id, exit_price, realised_pnl)
                        await get_risk_manager().close_trade(trade_id, exit_price)
                        result_icon = "✅ WIN" if realised_pnl > 0 else "❌ LOSS"
                        close_msg = (
                            f"{'✅' if realised_pnl > 0 else '❌'} Monitor: {pair} closed  ·  "
                            f"{result_icon}  ${realised_pnl:+,.2f}\n"
                            f"Exit @ <code>{exit_price}</code>  —  recorded in stats."
                        )
                    except Exception as e:
                        log.error(f"Failed to record closed trade {trade_id}: {e}")
            await context.bot.send_message(
                chat_id=chat_id, text=close_msg, parse_mode=ParseMode.HTML,
            )

        for trade in open_trades:
            instrument = trade.get("instrument", "")
            trade_id = str(trade.get("id", ""))
            pnl = float(trade.get("unrealizedPL", 0))
            units = float(trade.get("currentUnits", 0))
            direction = "BUY" if units > 0 else "SELL"

            # ── Split TP: close 50% at 2:1 and move SL to breakeven ──────────
            split = _split_tp_trades.get(trade_id)
            if split and not split.get("half_closed"):
                entry = split["entry"]
                sl_price = split["sl"]
                split_dir = split["direction"]
                initial_units = split["initial_units"]
                risk = abs(entry - sl_price)

                if risk > 0:
                    tp1 = (entry + 2 * risk) if split_dir == "BUY" else (entry - 2 * risk)
                    try:
                        live_price = await client.get_price(instrument)
                        current = live_price.mid
                        tp1_hit = (
                            (split_dir == "BUY" and current >= tp1)
                            or (split_dir == "SELL" and current <= tp1)
                        )
                    except Exception:
                        tp1_hit = False

                    if tp1_hit:
                        close_units = max(1, initial_units // 2)
                        partial_result = await client.partial_close_trade(trade_id, close_units)
                        if partial_result.success:
                            # Move SL to breakeven
                            await client.modify_trade_sl(trade_id, entry)
                            split["half_closed"] = True
                            _save_monitors()
                            filled_p = partial_result.filled_price
                            partial_pnl = pnl / 2  # rough half-position estimate
                            pair = instrument.replace("_", "/")
                            await context.bot.send_message(
                                chat_id=split["chat_id"],
                                text=(
                                    f"🎯 <b>Split-TP  ·  50% closed  ·  {pair}</b>\n"
                                    f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                                    f"Closed     {close_units:,} units @ <code>{filled_p}</code>\n"
                                    f"Est. P&amp;L  <b>${partial_pnl:+,.2f}</b>\n"
                                    f"SL → BE    <code>{entry}</code>  (breakeven)\n\n"
                                    f"<i>{initial_units - close_units:,} units remain — "
                                    f"riding to TP2 risk-free.</i>"
                                ),
                                parse_mode=ParseMode.HTML,
                            )
                            log.info(f"Split-TP partial close: trade {trade_id} {pair} {close_units} units @ {filled_p}")

            # ── Monitor profit target: auto-close full position ───────────────
            # If this trade has an active split-TP that hasn't fired yet,
            # check if the monitor target is lower than the split-TP target.
            # If monitor target is explicit and reachable first, honour it.
            split_entry = _split_tp_trades.get(trade_id)
            if split_entry and not split_entry.get("half_closed"):
                # Calculate the split-TP dollar value at 2R for comparison
                _s_entry = split_entry.get("entry", 0)
                _s_sl = split_entry.get("sl", 0)
                _s_risk_dist = abs(_s_entry - _s_sl)
                _s_units = split_entry.get("initial_units", 0)
                # Rough 2R USD value: risk_distance * units (simplified for majors)
                _split_tp_usd = _s_risk_dist * _s_units * 2 if _s_risk_dist and _s_units else float("inf")

                # Check if monitor has a lower explicit target for this instrument
                _mon_target = targets.get(instrument) or targets.get(instrument.replace("_", "/"))
                if _mon_target is None or _mon_target >= _split_tp_usd:
                    # No explicit monitor target, or it's higher than 2R — let split-TP run
                    continue
                # Otherwise: monitor target is lower than 2R, honour the monitor

            # Check if this instrument is being monitored for this chat
            min_profit = targets.get(instrument)
            if min_profit is None:
                # Also check slash-format e.g. EUR/USD
                min_profit = targets.get(instrument.replace("_", "/"))
            if min_profit is None:
                continue

            if pnl >= min_profit:
                result = await client.close_trade(trade_id)
                if result.success:
                    filled = result.filled_price or 0.0
                    await get_risk_manager().close_trade(trade_id, filled)
                    await db_close_trade(trade_id, filled, pnl)
                    _split_tp_trades.pop(trade_id, None)  # clean up split TP entry
                    _save_monitors()
                    pair = instrument.replace("_", "/")
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=(
                            f"✅ <b>Monitor: Auto-closed  ·  {pair}</b>\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"Direction   {direction}\n"
                            f"Exit Price  <code>{filled}</code>\n"
                            f"P&amp;L        <b>${pnl:+,.2f}</b>  "
                            f"(target: ${min_profit:+,.2f})\n\n"
                            f"<i>Auto-closed by /monitor.</i>"
                        ),
                        parse_mode=ParseMode.HTML,
                    )
                    # Remove from targets once closed
                    targets.pop(instrument, None)
                    targets.pop(instrument.replace("_", "/"), None)

        # If all targets closed, cancel the job
        if not targets:
            _monitor_targets.pop(chat_id, None)
            _save_monitors()
            context.job.schedule_removal()
            await context.bot.send_message(
                chat_id=chat_id,
                text="📭 All monitored trades closed. Monitor stopped.",
            )
        else:
            # Save state but do NOT spam a message every 90 seconds
            _save_monitors()

    except Exception as e:
        log.error(f"Monitor job error: {e}")


async def cmd_monitor(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /monitor USD/CHF         — Monitor with default target (+$5 or 50% of risk)
    /monitor USD/CHF 20      — Auto-close when P&L >= $20
    /monitor stop            — Stop all monitoring
    /monitor status          — Show what's being monitored
    """
    chat_id = update.effective_chat.id
    args = context.args or []

    if not args:
        await update.message.reply_html(
            "<b>/monitor usage:</b>\n"
            "/monitor USD/CHF       — watch, close at first profit\n"
            "/monitor USD/CHF 20    — close when P&amp;L ≥ $20\n"
            "/monitor all 10        — watch ALL open trades, close each at $10\n"
            "/monitor stop          — cancel all monitoring\n"
            "/monitor status        — show active monitors"
        )
        return

    if args[0].lower() == "stop":
        _monitor_targets.pop(chat_id, None)
        _save_monitors()
        # Remove any existing monitor jobs for this chat
        current_jobs = context.job_queue.get_jobs_by_name(f"monitor_{chat_id}")
        for job in current_jobs:
            job.schedule_removal()
        await update.message.reply_text("🛑 All trade monitoring stopped.")
        return

    if args[0].lower() == "status":
        targets = _monitor_targets.get(chat_id, {})
        if not targets:
            await update.message.reply_text("📭 No active monitors.")
            return

        # Live check: purge monitors whose trades are no longer open on OANDA
        try:
            from forexmind.data.oanda_client import get_oanda_client
            open_trades = await get_oanda_client().get_open_trades()
            open_instruments: set[str] = {t.get("instrument", "") for t in open_trades}
            stale = [
                inst for inst in list(targets)
                if inst not in open_instruments and inst.replace("/", "_") not in open_instruments
            ]
            for inst in stale:
                targets.pop(inst, None)
            if stale:
                _save_monitors()
        except Exception:
            open_instruments = set()
            stale = []

        if not targets:
            await update.message.reply_text("📭 All monitored trades have already closed.")
            return

        lines = ["👁 <b>Active Monitors</b>\n━━━━━━━━━━━━━━━━━━━━━━━━━"]
        for pair, target in targets.items():
            inst_key = pair.replace("/", "_")
            # Find current P&L for this instrument
            pnl_str = ""
            for t in open_trades:
                if t.get("instrument") == inst_key:
                    pnl = float(t.get("unrealizedPL", 0))
                    pnl_str = f"  ·  now {_pnl_icon(pnl)} ${pnl:+,.2f}"
                    break
            lines.append(f"  {pair.replace('_', '/')}  →  +${target:.2f} target{pnl_str}")
        if stale:
            lines.append(f"\n<i>Removed {len(stale)} already-closed trade(s).</i>")
        await update.message.reply_html("\n".join(lines))
        return

    # Parse pair and optional profit target
    pair_raw = args[0].upper().replace("/", "_")
    try:
        min_profit = float(args[1]) if len(args) > 1 else 5.0
    except ValueError:
        await update.message.reply_text("❌ Invalid profit target. Use a number e.g. /monitor EUR/USD 20")
        return

    msg = await update.message.reply_text(f"⏳ Setting up monitor for {pair_raw.replace('_', '/')}...")

    try:
        from forexmind.data.oanda_client import get_oanda_client
        client = get_oanda_client()
        open_trades = await client.get_open_trades()

        if pair_raw.lower() == "all":
            if not open_trades:
                await msg.edit_text("📭 No open trades to monitor.")
                return
            pairs_to_watch = {t.get("instrument"): min_profit for t in open_trades}
        else:
            # Verify the pair is actually open
            matching = [t for t in open_trades if t.get("instrument") == pair_raw]
            if not matching:
                await msg.edit_text(
                    f"📭 No open trade found for {pair_raw.replace('_', '/')}.\n"
                    f"Use /trades to see your open positions."
                )
                return
            pairs_to_watch = {pair_raw: min_profit}

        # Store targets and persist to disk
        if chat_id not in _monitor_targets:
            _monitor_targets[chat_id] = {}
        _monitor_targets[chat_id].update(pairs_to_watch)
        _save_monitors()

        # Cancel existing job for this chat to avoid duplicates
        for job in context.job_queue.get_jobs_by_name(f"monitor_{chat_id}"):
            job.schedule_removal()

        # Schedule repeating job
        context.job_queue.run_repeating(
            _monitor_job,
            interval=MONITOR_INTERVAL,
            first=30,  # First check after 30 seconds
            name=f"monitor_{chat_id}",
            data={"chat_id": chat_id},
        )

        watch_lines = "\n".join(
            f"  ·  {k.replace('_', '/')}  →  close at +${v:.2f}"
            for k, v in pairs_to_watch.items()
        )
        await msg.edit_text(
            f"👁 <b>Monitor Active</b>  ·  {len(pairs_to_watch)} pair{'s' if len(pairs_to_watch) > 1 else ''}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{watch_lines}\n\n"
            f"Checking every 90 seconds\n"
            f"<i>Silent until target hit.</i>\n\n"
            f"<code>/monitor stop</code>  to cancel  ·  "
            f"<code>/monitor status</code>  to review",
            parse_mode=ParseMode.HTML,
        )

    except Exception as e:
        await msg.edit_text(f"❌ Error: {str(e)[:200]}")


# ── App builder ────────────────────────────────────────────────────────────────

def build_telegram_app() -> "Application":  # type: ignore[type-arg]
    if not TELEGRAM_AVAILABLE:
        raise ImportError("python-telegram-bot not installed")

    cfg = get_settings()
    if not cfg.telegram.is_configured:
        raise ValueError("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set in .env")

    app = Application.builder().token(cfg.telegram.bot_token).job_queue(JobQueue()).build()

    # filters.COMMAND already lowercases, but passing a list covers any mixed-case
    # the user types (e.g. /TRADES, /Signals). PTB matches case-insensitively when
    # you register the command as all-lowercase — Telegram itself normalises to lowercase
    # for menu taps, but manual typed commands pass through as-is.
    _ci = {"filters": filters.COMMAND}
    app.add_handler(CommandHandler(["start"], cmd_start))
    app.add_handler(CommandHandler(["help"], cmd_help))
    app.add_handler(CommandHandler(["signal"], cmd_signal))
    app.add_handler(CommandHandler(["signals"], cmd_signals))
    app.add_handler(CommandHandler(["trade"], cmd_trade))
    app.add_handler(CommandHandler(["autotrade"], cmd_autotrade))
    app.add_handler(CommandHandler(["trades"], cmd_trades))
    app.add_handler(CommandHandler(["close"], cmd_close))
    app.add_handler(CommandHandler(["stats"], cmd_stats))
    app.add_handler(CommandHandler(["account"], cmd_account))
    app.add_handler(CommandHandler(["backtest"], cmd_backtest))
    app.add_handler(CommandHandler(["sessions"], cmd_sessions))
    app.add_handler(CommandHandler(["risk"], cmd_risk))
    app.add_handler(CommandHandler(["monitor"], cmd_monitor))
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
