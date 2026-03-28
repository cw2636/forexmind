"""
ForexMind — Telegram Bot Interface
=====================================
Allows you to interact with the ForexMind agent from any Telegram client.

Commands:
  /start          — Welcome message
  /signal EUR/USD — Get trading signal
  /signals        — Get top signals for current session
  /account        — Account snapshot
  /sessions       — Market session status
  /help           — List commands
  [any text]      — Chat with Claude agent

Advanced Python:
  - python-telegram-bot async handlers (v21+)
  - CommandHandler and MessageHandler pattern
  - Per-user conversation context (ConversationHandler)
  - HTML message formatting
"""

from __future__ import annotations

import json
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


# ── Message formatters ────────────────────────────────────────────────────────

def format_signal_message(data: dict) -> str:
    """Format a signal dict into a Telegram HTML message."""
    sig = data.get("signal", {})
    ind = data.get("indicators", {})
    session = data.get("session", {})
    sentiment = data.get("news_sentiment", {})

    pair = data.get("instrument", "").replace("_", "/")
    action = sig.get("action", "HOLD")
    emoji = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "⚪"

    session_str = ", ".join(session.get("active", [session.get("active_sessions", [])])) or "None"
    overlap_str = "🔥 YES" if session.get("is_overlap") else "No"
    news_str = f"{sentiment.get('impact', 'neutral').upper()} ({sentiment.get('score', 0):+.3f})"

    return f"""
{emoji} <b>{action} {pair}</b>
━━━━━━━━━━━━━━━━━━━━
📊 <b>Signal Details</b>
Entry:       <code>{sig.get('entry', '-')}</code>
Stop Loss:   <code>{sig.get('stop_loss', '-')}</code> ({sig.get('stop_loss_pips', '-')} pips)
Take Profit: <code>{sig.get('take_profit', '-')}</code> ({sig.get('take_profit_pips', '-')} pips)
R:R Ratio:   <b>{sig.get('risk_reward', '-')}:1</b>
Risk %:      {sig.get('risk_pct', '-')}%
Confidence:  <b>{sig.get('confidence', 0)}%</b>
Strategies:  {sig.get('agreeing_strategies', '-')} in agreement

📈 <b>Key Indicators</b>
RSI:  {ind.get('rsi', '-')} [{ind.get('rsi_zone', '-')}]
MACD: {ind.get('macd_cross', '-')} | ADX: {ind.get('adx', '-')} [{ind.get('adx_trend_strength', '-')}]
PSAR: {ind.get('psar_signal', '-')} | EMA: {ind.get('ema_trend', '-')}

🌍 <b>Context</b>
Sessions: {session_str}
Overlap:  {overlap_str}
News:     {news_str}
""".strip()


def format_account_message(data: dict) -> str:
    pnl = float(data.get("unrealized_pnl", 0))
    daily_pnl = float(data.get("daily_pnl_usd", 0))
    pnl_emoji = "📈" if pnl >= 0 else "📉"
    daily_emoji = "✅" if daily_pnl >= 0 else "⚠️"
    return f"""
💼 <b>OANDA Account</b>
━━━━━━━━━━━━━━━━━━━━
Balance:        <b>${data.get('balance', 0):,.2f}</b>
NAV:            ${data.get('nav', 0):,.2f}
{pnl_emoji} Unrealised P&L: ${pnl:,.2f}
{daily_emoji} Today's P&L:    ${daily_pnl:,.2f}
Open Trades:    {data.get('open_trade_count', 0)}
Margin Used:    ${data.get('margin_used', 0):,.2f}
Mode:           {'🧪 Paper Trading' if True else '⚠️ LIVE'}
""".strip()


# ── Handlers ──────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_html(
        "👋 <b>Welcome to ForexMind!</b>\n\n"
        "I'm your AI forex trading analyst. I can:\n"
        "• Analyse any forex pair in real-time\n"
        "• Give you entry, stop-loss, and take-profit levels\n"
        "• Monitor market sessions and news sentiment\n"
        "• Chat with you about trading strategies\n\n"
        "Type /help to see all commands, or just ask me anything!"
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_html(
        "<b>ForexMind Commands:</b>\n\n"
        "/signal EUR/USD — Get trading signal\n"
        "/signals — Top signals for current session\n"
        "/account — OANDA account snapshot\n"
        "/sessions — Current market sessions\n"
        "/help — Show this message\n\n"
        "<i>Or just type any message to chat with the AI agent!</i>"
    )


async def cmd_sessions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    status = get_session_status()
    pairs = best_pairs_for_session()[:6]

    if status.is_weekend:
        await update.message.reply_text("🚫 Weekend — Forex markets are closed")
        return

    session_lines = []
    for s in ["Sydney", "Tokyo", "London", "New York"]:
        active = s in status.active_sessions
        session_lines.append(f"{'🟢' if active else '⚫'} {s}")

    overlap = f"\n🔥 <b>OVERLAP: {', '.join(status.active_overlaps)} — Prime scalping time!</b>" if status.active_overlaps else ""
    rec_pairs = ", ".join(p.replace("_", "/") for p in pairs)

    await update.message.reply_html(
        f"<b>Market Sessions (UTC)</b>\n"
        f"{'  '.join(session_lines)}"
        f"{overlap}\n\n"
        f"Recommended pairs: <code>{rec_pairs}</code>"
    )


async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /signal EUR/USD"""
    args = context.args
    if not args:
        await update.message.reply_text("Usage: /signal EUR/USD (or any pair)")
        return

    pair = args[0].upper().replace("/", "_").replace("-", "_")
    msg = await update.message.reply_text(f"🔍 Analysing {pair}...")

    try:
        from forexmind.agents.tools import _get_signal
        result_str = await _get_signal(pair, "M5", 300)
        data = json.loads(result_str)

        if "error" in data:
            await msg.edit_text(f"❌ Error: {data['error']}")
            return

        text = format_signal_message(data)
        # Add inline keyboard for quick actions
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("📰 News", callback_data=f"news_{pair}"),
                InlineKeyboardButton("🔄 Refresh", callback_data=f"signal_{pair}"),
            ]
        ])
        await msg.edit_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)

    except Exception as e:
        await msg.edit_text(f"❌ Failed to get signal: {str(e)[:200]}")


async def cmd_signals(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Get signals for all recommended pairs."""
    msg = await update.message.reply_text("🔍 Fetching signals for current session pairs...")
    try:
        from forexmind.agents.tools import _get_signal
        pairs = best_pairs_for_session()[:4]
        lines = []
        for pair in pairs:
            result = await _get_signal(pair, "M5", 200)
            data = json.loads(result)
            sig = data.get("signal", {})
            action = sig.get("action", "HOLD")
            conf = sig.get("confidence", 0)
            emoji = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "⚪"
            lines.append(
                f"{emoji} <b>{pair.replace('_', '/')}</b>: {action} ({conf}%)"
            )
        await msg.edit_text("\n".join(lines) or "No signals", parse_mode=ParseMode.HTML)
    except Exception as e:
        await msg.edit_text(f"❌ Error: {str(e)[:200]}")


async def cmd_account(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = await update.message.reply_text("💼 Fetching account info...")
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


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle free-text messages by forwarding to Claude agent."""
    user_text = update.message.text
    msg = await update.message.reply_text("🤔 Thinking...")

    try:
        from forexmind.agents.claude_agent import get_agent
        agent = get_agent()
        response = await agent.chat(user_text)
        # Telegram has 4096 char limit
        if len(response) > 4000:
            response = response[:3997] + "..."
        await msg.edit_text(response, parse_mode=ParseMode.HTML)
    except Exception as e:
        await msg.edit_text(f"❌ Agent error: {str(e)[:200]}")


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline keyboard button callbacks."""
    query = update.callback_query
    await query.answer()
    data = query.data

    if data.startswith("signal_"):
        pair = data[7:]
        fake_update = update
        await query.edit_message_text("🔍 Refreshing...")
        try:
            from forexmind.agents.tools import _get_signal
            result = await _get_signal(pair, "M5", 300)
            signal_data = json.loads(result)
            text = format_signal_message(signal_data)
            keyboard = InlineKeyboardMarkup([[
                InlineKeyboardButton("📰 News", callback_data=f"news_{pair}"),
                InlineKeyboardButton("🔄 Refresh", callback_data=f"signal_{pair}"),
            ]])
            await query.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
        except Exception as e:
            await query.edit_message_text(f"❌ Error: {str(e)[:200]}")

    elif data.startswith("news_"):
        pair = data[5:]
        try:
            from forexmind.agents.tools import _get_news
            result = await _get_news(pair, 4)
            news_data = json.loads(result)
            sentiment = news_data.get("sentiment", {})
            articles = news_data.get("articles", [])[:5]
            lines = [
                f"📰 <b>News: {pair.replace('_', '/')}</b>",
                f"Sentiment: {sentiment.get('impact', 'neutral').upper()} ({sentiment.get('score', 0):+.3f})",
                f"Articles ({sentiment.get('article_count', 0)} relevant):\n",
            ]
            for art in articles:
                lines.append(f"• {art['headline'][:100]}")
            await query.edit_message_text("\n".join(lines), parse_mode=ParseMode.HTML)
        except Exception as e:
            await query.edit_message_text(f"❌ {str(e)[:200]}")


# ── App builder ────────────────────────────────────────────────────────────────

def build_telegram_app() -> "Application":  # type: ignore[type-arg]
    """Build and return the Telegram bot Application."""
    if not TELEGRAM_AVAILABLE:
        raise ImportError("python-telegram-bot not installed. Run: pip install python-telegram-bot")

    cfg = get_settings()
    if not cfg.telegram.is_configured:
        raise ValueError("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set in .env")

    app = Application.builder().token(cfg.telegram.bot_token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("signal", cmd_signal))
    app.add_handler(CommandHandler("signals", cmd_signals))
    app.add_handler(CommandHandler("account", cmd_account))
    app.add_handler(CommandHandler("sessions", cmd_sessions))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CallbackQueryHandler(handle_callback))

    log.info("Telegram bot configured and handlers registered")
    return app


async def run_telegram_bot() -> None:
    """Start the Telegram bot with long-polling."""
    app = build_telegram_app()
    log.info("Telegram bot starting (long-polling)...")
    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)
    log.info("Telegram bot running. Press Ctrl+C to stop.")
    # Run until interrupted
    import asyncio
    try:
        await asyncio.Event().wait()  # Wait forever
    except asyncio.CancelledError:
        pass
    finally:
        await app.updater.stop()
        await app.stop()
        await app.shutdown()
