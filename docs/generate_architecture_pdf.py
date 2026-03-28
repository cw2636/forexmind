"""
Generate ForexMind Architecture PDF
====================================
Run:  python docs/generate_architecture_pdf.py
Output: docs/ForexMind_Architecture.pdf
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether,
)
from reportlab.platypus import ListFlowable, ListItem
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

OUTPUT = os.path.join(os.path.dirname(__file__), "ForexMind_Architecture.pdf")

# ── Colour palette ────────────────────────────────────────────────────────────
NAVY   = colors.HexColor("#0D1B2A")
BLUE   = colors.HexColor("#1A73E8")
TEAL   = colors.HexColor("#00BFA5")
GOLD   = colors.HexColor("#F9A825")
LIGHT  = colors.HexColor("#E8F0FE")
GREY   = colors.HexColor("#F5F5F5")
MID    = colors.HexColor("#90A4AE")
RED    = colors.HexColor("#E53935")
GREEN  = colors.HexColor("#43A047")
WHITE  = colors.white


def build_styles():
    base = getSampleStyleSheet()
    s = {}

    s["cover_title"] = ParagraphStyle(
        "cover_title",
        fontSize=32, leading=40, textColor=WHITE,
        fontName="Helvetica-Bold", alignment=TA_CENTER,
        spaceAfter=10,
    )
    s["cover_sub"] = ParagraphStyle(
        "cover_sub",
        fontSize=16, leading=22, textColor=LIGHT,
        fontName="Helvetica", alignment=TA_CENTER,
        spaceAfter=20,
    )
    s["h1"] = ParagraphStyle(
        "h1",
        fontSize=20, leading=26, textColor=NAVY,
        fontName="Helvetica-Bold", spaceBefore=18, spaceAfter=8,
        borderPad=0,
    )
    s["h2"] = ParagraphStyle(
        "h2",
        fontSize=14, leading=18, textColor=BLUE,
        fontName="Helvetica-Bold", spaceBefore=12, spaceAfter=6,
    )
    s["h3"] = ParagraphStyle(
        "h3",
        fontSize=11, leading=15, textColor=NAVY,
        fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=4,
    )
    s["body"] = ParagraphStyle(
        "body",
        fontSize=10, leading=15, textColor=colors.HexColor("#212121"),
        fontName="Helvetica", spaceAfter=6, alignment=TA_JUSTIFY,
    )
    s["bullet"] = ParagraphStyle(
        "bullet",
        fontSize=10, leading=14, textColor=colors.HexColor("#212121"),
        fontName="Helvetica", spaceAfter=3,
        leftIndent=14, firstLineIndent=-14,
    )
    s["code"] = ParagraphStyle(
        "code",
        fontSize=8.5, leading=12, textColor=colors.HexColor("#1a1a1a"),
        fontName="Courier", backColor=GREY, borderPad=6,
        leftIndent=12, rightIndent=12, spaceAfter=8,
    )
    s["caption"] = ParagraphStyle(
        "caption",
        fontSize=9, leading=12, textColor=MID,
        fontName="Helvetica-Oblique", alignment=TA_CENTER, spaceAfter=8,
    )
    s["table_header"] = ParagraphStyle(
        "table_header",
        fontSize=9, leading=12, textColor=WHITE,
        fontName="Helvetica-Bold", alignment=TA_CENTER,
    )
    s["table_body"] = ParagraphStyle(
        "table_body",
        fontSize=9, leading=12, textColor=NAVY,
        fontName="Helvetica", alignment=TA_LEFT,
    )
    return s


# ── Helper for coloured section banners ──────────────────────────────────────

def section_banner(text, style, bg=NAVY, fg=WHITE):
    data = [[Paragraph(f"<font color='#{bg.hexval()[2:]}'>&#9632;</font>  {text}", style["h1"])]]
    tbl = Table(data, colWidths=[17 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("TEXTCOLOR",  (0, 0), (-1, -1), fg),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING",   (0, 0), (-1, -1), 14),
        ("ROUNDEDCORNERS", [4, 4, 4, 4]),
    ]))
    return tbl


def h1_banner(text, s):
    p = ParagraphStyle(
        "banner_text",
        fontSize=14, leading=18, textColor=WHITE,
        fontName="Helvetica-Bold",
    )
    data = [[Paragraph(text, p)]]
    tbl = Table(data, colWidths=[17 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), NAVY),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 14),
    ]))
    return tbl


def kv_table(rows, s, col_widths=None):
    """Two- or three-column key/value info table."""
    col_widths = col_widths or [5 * cm, 12 * cm]
    styled = []
    for row in rows:
        styled.append([Paragraph(str(c), s["table_body"]) for c in row])
    tbl = Table(styled, colWidths=col_widths, hAlign="LEFT")
    style = [
        ("BACKGROUND",    (0, 0), (-1, 0), BLUE),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, GREY]),
        ("GRID",          (0, 0), (-1, -1), 0.4, MID),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]
    tbl.setStyle(TableStyle(style))
    return tbl


def bullet(text, s):
    return Paragraph(f"<bullet>&bull;</bullet> {text}", s["bullet"])


# ── Cover page ────────────────────────────────────────────────────────────────

def cover_page(elements, s):
    # Dark header block
    cover_data = [[
        Paragraph("ForexMind", ParagraphStyle("ct", fontSize=38, leading=46,
            textColor=WHITE, fontName="Helvetica-Bold", alignment=TA_CENTER)),
    ], [
        Paragraph("AI-Powered Forex Trading Agent", ParagraphStyle("cs", fontSize=18,
            leading=24, textColor=LIGHT, fontName="Helvetica", alignment=TA_CENTER)),
    ], [
        Paragraph("System Architecture &amp; Technical Design Document",
            ParagraphStyle("cs2", fontSize=13, leading=18, textColor=GOLD,
            fontName="Helvetica-Bold", alignment=TA_CENTER)),
    ], [
        Paragraph("Version 1.0 &nbsp;|&nbsp; March 2026",
            ParagraphStyle("cv", fontSize=10, leading=14, textColor=MID,
            fontName="Helvetica", alignment=TA_CENTER)),
    ]]
    cover_tbl = Table(cover_data, colWidths=[17 * cm])
    cover_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), NAVY),
        ("TOPPADDING",    (0, 0), (-1, -1), 22),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 22),
        ("LEFTPADDING",   (0, 0), (-1, -1), 18),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 18),
    ]))
    elements.append(cover_tbl)
    elements.append(Spacer(1, 1 * cm))

    # Mission statement box
    mission_data = [[
        Paragraph(
            "<b>Mission:</b> Build an ensemble AI trading agent that achieves ≥70% directional "
            "accuracy on 1–5 minute forex charts across all major, minor, and exotic pairs — "
            "combining rule-based analytics, gradient-boosted ML, LSTM deep learning, "
            "and PPO reinforcement learning under a conversational Claude AI interface.",
            s["body"]
        )
    ]]
    mission_tbl = Table(mission_data, colWidths=[17 * cm])
    mission_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), LIGHT),
        ("TOPPADDING",    (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
        ("LEFTPADDING",   (0, 0), (-1, -1), 16),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 16),
        ("BOX",           (0, 0), (-1, -1), 1.5, BLUE),
    ]))
    elements.append(mission_tbl)
    elements.append(Spacer(1, 0.8 * cm))

    # Quick stats
    stats = [
        ["Metric", "Value"],
        ["Target Accuracy",    "≥70% directional accuracy on 1m–5m scalping"],
        ["Supported Pairs",    "18 pairs — Majors, Minors, Exotics"],
        ["Ensemble Strategies","4 — Rule-Based, LightGBM, Bi-LSTM, PPO-RL"],
        ["Interfaces",         "CLI (Rich), FastAPI Web Dashboard, Telegram Bot"],
        ["AI Brain",           "Anthropic Claude 3.5 Sonnet via LangChain 1.x"],
        ["Data Source",        "OANDA REST API (free practice account)"],
        ["Python Version",     "3.12  |  PyTorch 2.5.1 + CUDA 12.1"],
        ["Lines of Code",      "~4,500 across 31 source files"],
    ]
    elements.append(kv_table(stats, s, col_widths=[5.5 * cm, 11.5 * cm]))
    elements.append(PageBreak())


# ── Table of Contents ─────────────────────────────────────────────────────────

def toc_page(elements, s):
    elements.append(h1_banner("Table of Contents", s))
    elements.append(Spacer(1, 0.4 * cm))
    toc_items = [
        ("1", "System Overview & Design Philosophy", "3"),
        ("2", "High-Level Architecture Diagram", "4"),
        ("3", "Layer-by-Layer Module Breakdown", "5"),
        ("4", "Data Layer — OANDA Client & News Aggregator", "6"),
        ("5", "Technical Indicators Engine (25+ indicators)", "7"),
        ("6", "Strategy Layer — Four-Engine Ensemble", "8"),
        ("7", "Risk Management Engine", "10"),
        ("8", "Backtesting & Walk-Forward Validation", "11"),
        ("9", "Claude AI Agent — LangChain 1.x Architecture", "12"),
        ("10","User Interfaces overview", "14"),
        ("11","Database Schema & Persistence", "15"),
        ("12","Configuration System", "16"),
        ("13","Security Model & API Keys", "17"),
        ("14","Performance Targets & Benchmarks", "18"),
    ]
    toc_data = [["#", "Section", "Page"]] + toc_items
    tbl = Table(toc_data, colWidths=[1 * cm, 13.5 * cm, 2.5 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("TEXTCOLOR",  (0, 0), (-1, 0), WHITE),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, GREY]),
        ("GRID",       (0, 0), (-1, -1), 0.3, MID),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("ALIGN",      (2, 0), (2, -1), "CENTER"),
    ]))
    elements.append(tbl)
    elements.append(PageBreak())


# ── Section 1: Overview ───────────────────────────────────────────────────────

def section_overview(elements, s):
    elements.append(h1_banner("1. System Overview & Design Philosophy", s))
    elements.append(Spacer(1, 0.3 * cm))

    elements.append(Paragraph("What ForexMind Does", s["h2"]))
    elements.append(Paragraph(
        "ForexMind is a production-grade, end-to-end AI forex trading system. It continuously "
        "monitors live market data, applies 25+ technical indicators, runs four independent "
        "predictive models, aggregates their opinions via weighted voting, manages risk "
        "dynamically, and presents signals through a conversational Claude AI agent. "
        "Every component is async-first for minimal latency.", s["body"]
    ))

    elements.append(Paragraph("Core Design Principles", s["h2"]))
    principles = [
        ("<b>Ensemble over single-model</b>: No single model is reliably right in all market "
         "conditions. Four orthogonal strategies vote — rule-based, ML, deep learning, RL."),
        ("<b>Graceful degradation</b>: Every module has try/except import guards. If LightGBM "
         "model isn't trained yet, it returns HOLD. If news API is unconfigured, signals still work."),
        ("<b>Risk-first execution</b>: Signals are never directly traded. They pass through "
         "the RiskManager (kill-switch, ATR stops, Kelly sizing, R:R validation) first."),
        ("<b>Async-native I/O</b>: All network calls use asyncio.to_thread() or native async, "
         "so the agent can query multiple data sources concurrently without blocking."),
        ("<b>No hardcoded values</b>: Every tunable parameter lives in config.yaml. "
         "Risk limits, indicator periods, ensemble weights, session times — all configurable."),
        ("<b>Backtest before live</b>: Walk-forward validation and Monte Carlo simulation "
         "must run before any live trading to establish statistical edge."),
    ]
    for p in principles:
        elements.append(bullet(p, s))
    elements.append(Spacer(1, 0.3 * cm))

    elements.append(Paragraph("Technology Stack", s["h2"]))
    stack = [
        ["Layer",           "Technology",                   "Purpose"],
        ["AI Brain",        "Anthropic Claude 3.5 Sonnet",  "Conversational analysis & signal synthesis"],
        ["Agent Framework", "LangChain 1.x (bind_tools)",   "Tool orchestration & memory management"],
        ["ML Ensemble",     "LightGBM + PyTorch BiLSTM",    "Directional prediction from features"],
        ["RL Agent",        "stable-baselines3 PPO",        "Adaptive position management"],
        ["Indicators",      "pandas-ta (pure Python)",      "25+ technical indicators"],
        ["Broker/Data",     "OANDA REST API",               "Live prices, candles, order execution"],
        ["News",            "Alpha Vantage + Finnhub",      "Forex news with sentiment scoring"],
        ["Web UI",          "FastAPI + WebSocket",          "Real-time dashboard"],
        ["CLI",             "Rich (terminal)",              "Streaming chat interface"],
        ["Bot",             "python-telegram-bot v21+",     "Mobile alerts & commands"],
        ["Database",        "SQLAlchemy 2 + aiosqlite",     "SQLite async persistence"],
        ["Risk Mgmt",       "Custom (Kelly + ATR)",         "Position sizing & kill-switch"],
    ]
    elements.append(kv_table(stack, s, col_widths=[4 * cm, 6 * cm, 7 * cm]))
    elements.append(PageBreak())


# ── Section 2: High-Level Architecture Diagram (text diagram) ────────────────

def section_arch_diagram(elements, s):
    elements.append(h1_banner("2. High-Level Architecture Diagram", s))
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(Paragraph(
        "The diagram below shows information flow from external data sources through "
        "all processing layers to the user interfaces.", s["body"]
    ))
    elements.append(Spacer(1, 0.2 * cm))

    # Architecture table as a visual diagram replacement
    arch_rows = [
        ["EXTERNAL SOURCES (Live)"],
        ["OANDA REST API (prices/candles/orders)  •  Alpha Vantage News  •  Finnhub News"],
        ["↓"],
        ["DATA LAYER"],
        ["OandaClient (async)  •  NewsAggregator (dedup+sentiment)  •  SQLite (async)"],
        ["↓"],
        ["INDICATORS ENGINE  (pandas-ta)"],
        ["EMA 9/21/50/200  •  MACD  •  RSI  •  Stochastic  •  Bollinger Bands  •  ADX"],
        ["ATR  •  PSAR  •  CCI  •  Williams %R  •  MFI  •  OBV  •  Pivot Points  (+more)"],
        ["↓"],
        ["SIGNAL SCORER   Weighted composite score  [-100 → +100]"],
        ["Trend 30%  |  Momentum 25%  |  Structure 20%  |  Volume 15%  |  Volatility 10%"],
        ["↓"],
        ["STRATEGY ENSEMBLE  (weighted voting)"],
        ["Rule-Based 30%  |  LightGBM 25%  |  Bi-LSTM 25%  |  PPO-RL 20%"],
        ["↓"],
        ["RISK MANAGER"],
        ["Kill-switch  •  ATR stops  •  Kelly sizing  •  R:R validation  •  Trailing stops"],
        ["↓"],
        ["CLAUDE AGENT  (LangChain 1.x  bind_tools)"],
        ["6 tools  •  Sliding-window memory (40 msgs)  •  JSON signal synthesis"],
        ["↓"],
        ["USER INTERFACES"],
        ["Rich CLI  •  FastAPI Dashboard + WebSocket  •  Telegram Bot"],
    ]

    colours = [NAVY, NAVY, NAVY, BLUE, LIGHT, NAVY, TEAL, GREY, GREY,
               NAVY, GREEN, GREY, NAVY, GREY, GREY, NAVY, GREY,
               NAVY, GREY, GREY, NAVY, GREY, GREY, GREY]
    text_colours = [WHITE, WHITE, WHITE, WHITE, NAVY, WHITE, WHITE, NAVY, NAVY,
                    WHITE, WHITE, NAVY, WHITE, NAVY, NAVY, WHITE, NAVY,
                    WHITE, NAVY, NAVY, WHITE, NAVY, NAVY, NAVY]
    bold_rows = {0, 3, 6, 9, 10, 12, 15, 17, 20}

    for i, row in enumerate(arch_rows):
        fn = "Helvetica-Bold" if i in bold_rows else "Helvetica"
        align = TA_CENTER if row[0] == "↓" else TA_CENTER
        p = ParagraphStyle(f"ar{i}", fontSize=9 if i not in bold_rows else 10,
                           leading=14, textColor=text_colours[i],
                           fontName=fn, alignment=TA_CENTER)
        data = [[Paragraph(row[0], p)]]
        tbl = Table(data, colWidths=[17 * cm])
        bg = colours[i]
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), bg),
            ("TOPPADDING",    (0, 0), (-1, -1), 5 if i not in bold_rows else 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5 if i not in bold_rows else 8),
        ]))
        elements.append(tbl)
        if row[0] != "↓":
            elements.append(Spacer(1, 0.05 * cm))

    elements.append(PageBreak())


# ── Section 3: Module Breakdown ───────────────────────────────────────────────

def section_modules(elements, s):
    elements.append(h1_banner("3. Layer-by-Layer Module Breakdown", s))
    elements.append(Spacer(1, 0.3 * cm))

    modules = [
        ["Module / File",                      "Class / Function",        "Responsibility"],
        ["config/settings.py",                  "Settings, get_settings()", "Singleton config, reads .env + config.yaml"],
        ["config/config.yaml",                  "—",                        "All runtime tunables (pairs, periods, weights)"],
        ["utils/helpers.py",                    "pip_size, units_from_risk, kelly_fraction", "Pure trading maths, no I/O"],
        ["utils/session_times.py",              "get_session_status()",    "Sydney/Tokyo/London/NY session detection"],
        ["utils/logger.py",                     "get_logger()",            "Rich-powered structured logger"],
        ["data/oanda_client.py",                "OandaClient",             "OANDA REST — prices, candles, orders"],
        ["data/news_aggregator.py",             "NewsAggregator",          "News fetch, dedup, TextBlob sentiment"],
        ["data/database.py",                    "init_db, get_session()",  "Async SQLAlchemy engine"],
        ["data/models.py",                      "Candle, Signal, Trade",   "SQLAlchemy ORM models"],
        ["indicators/engine.py",                "IndicatorEngine",         "25+ indicators via pandas-ta"],
        ["indicators/scorer.py",                "score_snapshot()",        "Composite signal score [-100,+100]"],
        ["strategy/feature_engineering.py",     "build_feature_matrix()",  "ML feature pipeline (lags, rolling stats)"],
        ["strategy/rule_based.py",              "RuleBasedStrategy",       "6-condition scalping checklist"],
        ["strategy/ml_strategy.py",             "LightGBMStrategy, LSTMStrategy", "Gradient-boosted + Bi-LSTM models"],
        ["strategy/rl_strategy.py",             "RLStrategy, ForexTradingEnv", "PPO reinforcement learning"],
        ["strategy/ensemble.py",                "EnsembleStrategy",        "Weighted voting across all 4 strategies"],
        ["strategy/base.py",                    "BaseStrategy, StrategySignal", "Abstract base + signal dataclass"],
        ["risk/manager.py",                     "RiskManager, RiskProposal", "Kill-switch, sizing, trailing stops"],
        ["backtest/engine.py",                  "Backtester",              "Event-driven backtest, walk-forward, Monte Carlo"],
        ["agents/prompts.py",                   "SYSTEM_PROMPT",           "Claude expert trader persona definition"],
        ["agents/tools.py",                     "build_tools() → 6 tools", "LangChain StructuredTool wrappers"],
        ["agents/claude_agent.py",              "ForexMindAgent",          "bind_tools agentic loop, streaming, memory"],
        ["interfaces/cli.py",                   "run_cli()",               "Rich terminal chat with commands"],
        ["interfaces/web/app.py",               "FastAPI app",             "REST + WebSocket server"],
        ["interfaces/web/static/index.html",    "—",                       "Dark-theme live trading dashboard"],
        ["interfaces/telegram_bot.py",          "run_telegram_bot()",      "Async Telegram bot with inline keyboards"],
        ["main.py",                             "argparse entry point",    "cli/web/telegram/all/signal/backtest/train"],
    ]
    elements.append(kv_table(modules, s, col_widths=[5.5 * cm, 5.5 * cm, 6 * cm]))
    elements.append(PageBreak())


# ── Section 4: Data Layer ─────────────────────────────────────────────────────

def section_data(elements, s):
    elements.append(h1_banner("4. Data Layer — OANDA Client & News Aggregator", s))
    elements.append(Spacer(1, 0.3 * cm))

    elements.append(Paragraph("4.1  OANDA REST Client (oanda_client.py)", s["h2"]))
    elements.append(Paragraph(
        "Wraps the synchronous <i>oandapyV20</i> library inside <b>asyncio.to_thread()</b> so "
        "all calls are non-blocking. A <b>@retry</b> decorator with exponential back-off "
        "(3 attempts, 2× delay) handles transient 429/5xx errors automatically.", s["body"]
    ))
    oanda_rows = [
        ["Method",             "Returns",          "Description"],
        ["get_price(inst)",    "LivePrice",        "Bid/ask/mid for one instrument"],
        ["get_candles(inst, tf, n)", "DataFrame",  "OHLCV bars with UTC DatetimeIndex"],
        ["get_multi_candles(pairs)", "dict[str,DF]","Concurrent fetch for multiple pairs"],
        ["market_order(inst,units,sl,tp)", "OrderResult", "Place market order with bracket"],
        ["close_trade(id)",    "OrderResult",      "Close specific trade by OANDA ID"],
        ["get_open_positions()","list[dict]",      "All currently open positions"],
        ["get_account()",      "AccountSummary",   "Balance, NAV, margin, open trade count"],
    ]
    elements.append(kv_table(oanda_rows, s, col_widths=[5 * cm, 4 * cm, 8 * cm]))
    elements.append(Spacer(1, 0.3 * cm))

    elements.append(Paragraph("4.2  News Aggregator (news_aggregator.py)", s["h2"]))
    elements.append(Paragraph(
        "Implements the <b>NewsSource</b> ABC with two concrete sources. "
        "Results are deduplicated by headline hash and cached for 15 minutes. "
        "TextBlob computes a sentiment score from −1.0 (very bearish) to +1.0 (very bullish).",
        s["body"]
    ))
    news_rows = [
        ["Source",            "Rate Limit",   "Content"],
        ["Alpha Vantage",     "25 req/day",   "Forex market news with tickers"],
        ["Finnhub",           "60 req/min",   "General financial news"],
        ["get_instrument_sentiment()", "—",   "Returns: score, article_count, impact level, recent_headlines"],
    ]
    elements.append(kv_table(news_rows, s, col_widths=[5 * cm, 3.5 * cm, 8.5 * cm]))
    elements.append(PageBreak())


# ── Section 5: Indicators ─────────────────────────────────────────────────────

def section_indicators(elements, s):
    elements.append(h1_banner("5. Technical Indicators Engine (25+ indicators)", s))
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(Paragraph(
        "All indicators are computed using <b>pandas-ta</b> (pure Python — no C compilation). "
        "The engine's <b>snapshot()</b> method returns an <i>IndicatorSnapshot</i> TypedDict "
        "with derived signals (e.g. ema_trend, macd_cross, rsi_zone) ready for Claude to interpret.",
        s["body"]
    ))

    ind_rows = [
        ["Category",    "Indicators",                                     "Signal Derived"],
        ["Trend",       "EMA 9, 21, 50, 200  •  PSAR",                   "ema_trend (UP/DOWN/FLAT), psar_signal"],
        ["Momentum",    "MACD (12/26/9)  •  RSI(14)  •  Stochastic(14,3)","macd_cross, rsi_zone, stoch_cross"],
        ["Oscillators", "CCI(20)  •  Williams %R(14)  •  MFI(14)",        "Position relative to extremes"],
        ["Trend Str.",  "ADX(14) + DI+/DI−",                             "adx_trend_strength (STRONG/WEAK)"],
        ["Volatility",  "ATR(14)  •  Bollinger Bands(20,2)",              "bb_position, atr (for stop sizing)"],
        ["Volume",      "OBV  •  ROC(5/10)",                              "Volume trend confirmation"],
        ["Structure",   "Pivot Points (R1/R2/S1/S2)  •  Swing highs/lows","support, resistance levels"],
    ]
    elements.append(kv_table(ind_rows, s, col_widths=[3.5 * cm, 7 * cm, 6.5 * cm]))
    elements.append(Spacer(1, 0.3 * cm))

    elements.append(Paragraph("Signal Scorer Weights", s["h2"]))
    elements.append(Paragraph(
        "The <b>SignalScorer</b> aggregates indicator votes into a single composite score "
        "from −100 (strong SELL) to +100 (strong BUY). Direction is assigned when "
        "score ≥ +15 (BUY) or ≤ −15 (SELL).", s["body"]
    ))
    weight_rows = [
        ["Group",        "Weight", "Key Indicators"],
        ["Trend",        "30%",    "EMA stack alignment, PSAR side"],
        ["Momentum",     "25%",    "RSI, Stochastic, MACD histogram"],
        ["Structure",    "20%",    "Price vs support/resistance, BB position"],
        ["Volume",       "15%",    "OBV trend, MFI, ROC"],
        ["Volatility",   "10%",    "ATR expansion, BB squeeze"],
    ]
    elements.append(kv_table(weight_rows, s, col_widths=[4 * cm, 3 * cm, 10 * cm]))
    elements.append(PageBreak())


# ── Section 6: Strategy Ensemble ─────────────────────────────────────────────

def section_ensemble(elements, s):
    elements.append(h1_banner("6. Strategy Layer — Four-Engine Ensemble", s))
    elements.append(Spacer(1, 0.3 * cm))

    elements.append(Paragraph("6.1  Ensemble Voting Algorithm", s["h2"]))
    elements.append(Paragraph(
        "Each strategy independently produces a direction (BUY/SELL/HOLD) and a confidence "
        "score [0.0–1.0]. Weighted votes are accumulated:", s["body"]
    ))
    elements.append(Paragraph(
        "<b>buy_score</b> = Σ (weight × confidence) for each BUY strategy<br/>"
        "<b>sell_score</b> = Σ (weight × confidence) for each SELL strategy<br/>"
        "Direction wins if score ≥ 0.50 AND ≥ 2 strategies agree. Otherwise: <b>HOLD</b>.",
        s["code"]
    ))

    ens_rows = [
        ["Strategy",        "Weight", "Model Type",        "Confidence Source"],
        ["Rule-Based",       "30%",   "Expert heuristics", "conditions_met / 6"],
        ["LightGBM",         "25%",   "Gradient-boosted tree", "predict_proba() max class prob"],
        ["Bi-LSTM",          "25%",   "Deep learning",     "softmax output, 3-class"],
        ["PPO-RL",           "20%",   "Reinforcement learning", "action probability from policy"],
    ]
    elements.append(kv_table(ens_rows, s, col_widths=[4 * cm, 2.5 * cm, 5 * cm, 5.5 * cm]))
    elements.append(Spacer(1, 0.3 * cm))

    elements.append(Paragraph("6.2  Rule-Based Strategy", s["h2"]))
    conditions = [
        ("<b>HTF trend alignment</b>: H1 EMA(50) slope must agree with trade direction"),
        ("<b>EMA stack</b>: EMA9 &gt; EMA21 &gt; EMA50 (BUY) or reverse (SELL)"),
        ("<b>MACD</b>: Histogram positive (BUY) or negative (SELL)"),
        ("<b>RSI</b>: In range 40–70 for longs, 30–60 for shorts (avoids extremes)"),
        ("<b>Stochastic</b>: %K crosses above %D (BUY) or below %D (SELL)"),
        ("<b>PSAR</b>: Parabolic SAR on correct side of price"),
    ]
    elements.append(Paragraph("6 conditions checked — trade requires ≥4 met:", s["body"]))
    for c in conditions:
        elements.append(bullet(c, s))
    elements.append(Spacer(1, 0.3 * cm))

    elements.append(Paragraph("6.3  LightGBM Strategy", s["h2"]))
    lgbm_steps = [
        "<b>Feature engineering</b>: 5-bar lags, rolling stats (5/10/20 windows), candle patterns (engulfing, doji), session flags, normalized indicators",
        "<b>Training</b>: TimeSeriesSplit (5-fold, no look-ahead) with class-balanced weights",
        "<b>Pipeline</b>: sklearn.Pipeline([StandardScaler, LGBMClassifier])",
        "<b>Persistence</b>: joblib.dump() to models/lightgbm_{pair}.pkl",
        "<b>Target label</b>: next-bar direction with ≥3 pip threshold filter",
    ]
    for step in lgbm_steps:
        elements.append(bullet(step, s))
    elements.append(Spacer(1, 0.3 * cm))

    elements.append(Paragraph("6.4  Bidirectional LSTM", s["h2"]))
    lstm_arch = [
        ["Layer",              "Details"],
        ["Input",              "Sequence of 60 bars × N features (normalized)"],
        ["BiLSTM Layer 1",     "hidden=128, bidirectional, dropout=0.3"],
        ["BiLSTM Layer 2",     "hidden=128, bidirectional, dropout=0.3"],
        ["BatchNorm",          "Applied after LSTM output"],
        ["FC → Softmax",       "3-class output: BUY / HOLD / SELL"],
        ["Optimizer",          "AdamW (lr=1e-3, weight_decay=1e-4)"],
        ["Scheduler",          "CosineAnnealingLR (T_max=50)"],
        ["Gradient clipping",  "max_norm=1.0 (stability)"],
    ]
    elements.append(kv_table(lstm_arch, s, col_widths=[5 * cm, 12 * cm]))
    elements.append(Spacer(1, 0.3 * cm))

    elements.append(Paragraph("6.5  PPO Reinforcement Learning Agent", s["h2"]))
    rl_items = [
        "<b>Environment</b>: Custom gymnasium.Env — ForexTradingEnv",
        "<b>Observation space</b>: Box(window × features flattened) — last 60 bars",
        "<b>Action space</b>: Discrete(3) — HOLD / BUY / SELL",
        "<b>Reward function</b>: P&amp;L in pips − (overtrading penalty × |position changes|)",
        "<b>Policy</b>: MlpPolicy (PPO from stable-baselines3), wrapped in DummyVecEnv",
        "<b>Inference</b>: predict(obs, deterministic=True) for consistent live signals",
    ]
    for item in rl_items:
        elements.append(bullet(item, s))
    elements.append(PageBreak())


# ── Section 7: Risk Management ────────────────────────────────────────────────

def section_risk(elements, s):
    elements.append(h1_banner("7. Risk Management Engine", s))
    elements.append(Spacer(1, 0.3 * cm))

    elements.append(Paragraph(
        "The <b>RiskManager</b> is the final gatekeeper before any signal becomes an order. "
        "No trade can bypass it — not even Claude.", s["body"]
    ))

    elements.append(Paragraph("7.1  calculate_risk() Pipeline", s["h2"]))
    pipeline = [
        "1. <b>Kill-switch check</b>: Daily P&amp;L &lt; −(balance × max_daily_loss_pct) → reject ALL trades",
        "2. <b>Concurrent trade check</b>: open_trades ≥ max_concurrent_trades (default 5) → reject",
        "3. <b>ATR-based stop-loss</b>: SL = entry ± (ATR × 1.5)",
        "4. <b>ATR-based take-profit</b>: TP = entry ± (stop_pips × rr_ratio × pip_size), rr_ratio ≥ 2.0",
        "5. <b>R:R validation</b>: actual_rr = tp_pips / sl_pips — must be ≥ 1.0 or trade is rejected",
        "6. <b>Position sizing</b>: risk_pct from Kelly Criterion (capped at max_risk_per_trade_pct = 2%)",
        "7. <b>Returns RiskProposal</b>: approved=True/False with full explanation",
    ]
    for p in pipeline:
        elements.append(bullet(p, s))
    elements.append(Spacer(1, 0.3 * cm))

    elements.append(Paragraph("7.2  Kelly Criterion Implementation", s["h2"]))
    elements.append(Paragraph(
        "Half-Kelly formula is used to reduce bet size and avoid over-betting:", s["body"]
    ))
    elements.append(Paragraph(
        "kelly = (win_rate × rr_ratio − (1 − win_rate)) / rr_ratio\n"
        "risk_pct = min(max(kelly × 50,  min_risk_pct=0.5%),  max_risk_pct=2.0%)",
        s["code"]
    ))

    elements.append(Paragraph("7.3  Configuration Parameters", s["h2"]))
    risk_params = [
        ["Parameter",               "Default", "Description"],
        ["max_risk_per_trade_pct",   "2.0%",   "Maximum account risk per trade"],
        ["min_risk_per_trade_pct",   "0.5%",   "Minimum risk even for low-confidence trades"],
        ["default_rr_ratio",         "2.0",    "Minimum reward:risk ratio"],
        ["atr_stop_multiplier",      "1.5",    "Stop-loss = entry ± (ATR × 1.5)"],
        ["trailing_stop_multiplier", "1.0",    "Trailing stop distance in ATR units"],
        ["breakeven_trigger_rr",     "1.0",    "Move SL to break-even after 1R profit"],
        ["max_concurrent_trades",    "5",      "Maximum simultaneous open trades"],
        ["max_daily_loss_pct",        "5.0%",  "Kill-switch triggers at −5% daily drawdown"],
        ["spread_filter_pips",        "3.0",   "Skip trade if bid-ask spread &gt; 3 pips"],
    ]
    elements.append(kv_table(risk_params, s, col_widths=[5.5 * cm, 2.5 * cm, 9 * cm]))
    elements.append(PageBreak())


# ── Section 8: Backtesting ────────────────────────────────────────────────────

def section_backtest(elements, s):
    elements.append(h1_banner("8. Backtesting & Walk-Forward Validation", s))
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(Paragraph(
        "ForexMind requires statistical validation before any live deployment. "
        "Three validation layers ensure results are robust:", s["body"]
    ))

    bt_rows = [
        ["Validation Method",    "Purpose",                    "Implementation"],
        ["Event-Driven Backtest","Realistic SL/TP simulation", "Bar-by-bar loop, slippage + commission"],
        ["Walk-Forward CV",      "No look-ahead bias",         "TimeSeriesSplit(n_splits=5) across 2yr history"],
        ["Monte Carlo (1000 runs)","Stress test drawdown",     "Shuffle trade order → 95th-pct VaR"],
    ]
    elements.append(kv_table(bt_rows, s, col_widths=[4.5 * cm, 5.5 * cm, 7 * cm]))
    elements.append(Spacer(1, 0.3 * cm))

    elements.append(Paragraph("Backtest Metrics Reported", s["h2"]))
    metrics = [
        "<b>Win Rate</b>: % of trades that hit take-profit",
        "<b>Profit Factor</b>: gross_profit / gross_loss (target ≥ 1.5)",
        "<b>Sharpe Ratio</b>: annualized excess return / volatility (target ≥ 1.0)",
        "<b>Sortino Ratio</b>: downside-only risk adjustment",
        "<b>Maximum Drawdown %</b>: largest peak-to-trough equity decline",
        "<b>Net Return %</b>: total P&amp;L as % of starting capital",
        "<b>Average Trade Duration</b>: minutes (scalping target: 5–30 min)",
    ]
    for m in metrics:
        elements.append(bullet(m, s))
    elements.append(PageBreak())


# ── Section 9: Claude Agent ───────────────────────────────────────────────────

def section_agent(elements, s):
    elements.append(h1_banner("9. Claude AI Agent — LangChain 1.x Architecture", s))
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(Paragraph(
        "The ForexMindAgent uses <b>LangChain 1.x's bind_tools pattern</b> — "
        "NOT the deprecated AgentExecutor. Claude is bound to 6 tools at construction; "
        "when Claude decides a tool is needed, it emits a ToolCall in its response. "
        "We execute that tool, append a ToolMessage result, and loop until Claude "
        "gives a final text answer (max 6 iterations).", s["body"]
    ))

    elements.append(Paragraph("9.1  Agentic Loop (run_agent_loop)", s["h2"]))
    elements.append(Paragraph(
        "while iterations &lt; 6:\n"
        "    response = await llm_with_tools.ainvoke(messages)\n"
        "    if response.tool_calls is empty: return text_content(response)\n"
        "    for tc in response.tool_calls:\n"
        "        result = await execute_tool(tc.name, tc.args)\n"
        "        messages.append(ToolMessage(result, tool_call_id=tc.id))\n"
        "return force_final_answer(messages)",
        s["code"]
    ))

    elements.append(Paragraph("9.2  Six LangChain Tools", s["h2"]))
    tool_rows = [
        ["Tool Name",    "Async", "Description"],
        ["get_signal",   "Yes",   "Fetches live OANDA data, computes 25+ indicators, runs full ensemble, returns JSON signal package"],
        ["get_news",     "Yes",   "Fetches Alpha Vantage + Finnhub news, deduplicates, returns sentiment summary"],
        ["get_account",  "Yes",   "Returns OANDA balance, NAV, margin, open trades, daily P&L, kill-switch status"],
        ["run_backtest", "Yes",   "Fetches 1yr OANDA history, runs event-driven backtest, returns performance metrics"],
        ["place_trade",  "Yes",   "Executes market order on OANDA paper account with SL/TP bracket"],
        ["get_sessions", "No",    "Returns active forex sessions, overlaps, session score, recommended pairs"],
    ]
    elements.append(kv_table(tool_rows, s, col_widths=[3.5 * cm, 1.5 * cm, 12 * cm]))
    elements.append(Spacer(1, 0.3 * cm))

    elements.append(Paragraph("9.3  Expert Trader System Prompt", s["h2"]))
    elements.append(Paragraph(
        "Claude is given a detailed <b>SYSTEM_PROMPT</b> defining it as a 20-year professional "
        "FX trader. The prompt specifies:", s["body"]
    ))
    prompt_items = [
        "Structured JSON signal format: {action, entry, stop_loss, take_profit, confidence, reasoning}",
        "WHEN TO HOLD rules: low ADX, high spread, weekend, conflicting signals",
        "Risk enforcement: always validate R:R ≥ 2:1 before recommending trades",
        "Educational mode: explain reasoning clearly for user to learn advanced trading",
        "Session awareness: always factor in current liquidity window",
    ]
    for item in prompt_items:
        elements.append(bullet(item, s))
    elements.append(Spacer(1, 0.3 * cm))

    elements.append(Paragraph("9.4  Streaming & Memory", s["h2"]))
    mem_items = [
        "<b>Memory</b>: Sliding deque (maxlen=40 messages = 20 conversation turns) — no external library",
        "<b>stream_chat()</b>: Resolves all tool calls silently, then streams final answer token-by-token via llm.astream()",
        "<b>extract_signal()</b>: Regex parser finds JSON blocks in Claude responses — returns parsed dict or None",
    ]
    for item in mem_items:
        elements.append(bullet(item, s))
    elements.append(PageBreak())


# ── Section 10: Interfaces ────────────────────────────────────────────────────

def section_interfaces(elements, s):
    elements.append(h1_banner("10. User Interfaces", s))
    elements.append(Spacer(1, 0.3 * cm))

    iface_rows = [
        ["Interface",        "Technology",           "Key Features"],
        ["CLI (Terminal)",   "Rich + asyncio",        "Streaming responses, /signal, /account, /pairs, /sessions, /backtest commands; colour-coded panels"],
        ["Web Dashboard",    "FastAPI + WebSocket",   "Dark-theme UI, live signal cards grid, 60s auto-refresh via WebSocket, Claude chat panel, confidence bars"],
        ["Telegram Bot",     "python-telegram-bot v21","Commands: /signal PAIR, /signals, /account, /sessions; inline keyboards; free-text → Claude"],
        ["REST API",         "FastAPI (JSON)",        "GET /api/signal/{inst}, /api/account, /api/news/{inst}, /api/sessions; POST /api/chat"],
    ]
    elements.append(kv_table(iface_rows, s, col_widths=[3.5 * cm, 4.5 * cm, 9 * cm]))
    elements.append(PageBreak())


# ── Section 11: Database ──────────────────────────────────────────────────────

def section_database(elements, s):
    elements.append(h1_banner("11. Database Schema & Persistence", s))
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(Paragraph(
        "SQLite (via SQLAlchemy 2.0 async + aiosqlite). No external database server required. "
        "All writes are async to avoid blocking the event loop.", s["body"]
    ))

    db_rows = [
        ["Table",        "Key Columns",                      "Purpose"],
        ["candles",      "instrument, timeframe, time (unique)", "Historical OHLCV cache"],
        ["news_articles","headline_hash (dedup), sentiment_score", "Fetched news with sentiment"],
        ["signals",      "instrument, direction, confidence, source", "Generated trading signals"],
        ["trades",       "oanda_trade_id, status, entry/sl/tp/exit, pnl", "Full trade lifecycle"],
        ["economic_events","name, impact, time, actual/forecast", "Calendar events (future)"],
    ]
    elements.append(kv_table(db_rows, s, col_widths=[3.5 * cm, 6.5 * cm, 7 * cm]))
    elements.append(PageBreak())


# ── Section 12: Config ────────────────────────────────────────────────────────

def section_config(elements, s):
    elements.append(h1_banner("12. Configuration System", s))
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(Paragraph(
        "Two-layer configuration: <b>.env</b> for secrets (API keys), "
        "<b>config.yaml</b> for all runtime tunables. The <b>Settings</b> dataclass "
        "is a singleton (lru_cache) that merges both on first access.", s["body"]
    ))

    config_rows = [
        ["Source",       "Contains",                           "Example"],
        [".env",         "API keys, secrets, feature flags",   "OANDA_API_KEY=xxx, ANTHROPIC_API_KEY=yyy"],
        ["config.yaml",  "Pairs, timeframes, indicator periods","pairs: [EUR_USD, GBP_USD...], ema_periods: [9,21,50,200]"],
        ["config.yaml",  "Risk defaults",                      "max_risk_per_trade_pct: 2.0, default_rr_ratio: 2.0"],
        ["config.yaml",  "Ensemble weights",                   "rule_based: 0.30, lightgbm: 0.25, lstm: 0.25, rl: 0.20"],
        ["config.yaml",  "Session times (UTC)",                "london: {open: 07:00, close: 16:00}"],
        ["config.yaml",  "ML settings",                        "lookback_bars: 60, cv_folds: 5, min_confidence: 0.55"],
        ["config.yaml",  "News settings",                      "fetch_interval: 15min, sentiment_window: 4hr"],
    ]
    elements.append(kv_table(config_rows, s, col_widths=[3 * cm, 5.5 * cm, 8.5 * cm]))
    elements.append(PageBreak())


# ── Section 13: Security ──────────────────────────────────────────────────────

def section_security(elements, s):
    elements.append(h1_banner("13. Security Model & API Keys", s))
    elements.append(Spacer(1, 0.3 * cm))
    security_items = [
        "<b>No secrets in code</b>: All API keys read from .env via python-dotenv. .env is in .gitignore.",
        "<b>Paper trading default</b>: PAPER_TRADING=true in .env — live trading requires explicit opt-in (PAPER_TRADING=false)",
        "<b>Kill-switch</b>: Hard daily loss limit (−5% by default) halts ALL trading until next day",
        "<b>No SQL injection</b>: All DB writes use SQLAlchemy parameterized queries",
        "<b>WebSocket authentication</b>: (future) — currently internal-network only",
        "<b>Telegram</b>: Bot responds only to the configured TELEGRAM_CHAT_ID",
        "<b>Rate limiting</b>: @retry decorator prevents hammering APIs; news cache TTL avoids quota exhaustion",
        "<b>OANDA practice account</b>: All default API calls go to api-fxpractice.oanda.com — no real money",
    ]
    for item in security_items:
        elements.append(bullet(item, s))
    elements.append(PageBreak())


# ── Section 14: Performance Targets ──────────────────────────────────────────

def section_performance(elements, s):
    elements.append(h1_banner("14. Performance Targets & Benchmarks", s))
    elements.append(Spacer(1, 0.3 * cm))

    perf_rows = [
        ["Metric",                  "Target",   "Current Status"],
        ["Directional Accuracy",    "≥70%",     "Pending ML training on live data"],
        ["Profit Factor",           "≥1.5",     "Pending backtest on trained models"],
        ["Sharpe Ratio",            "≥1.0",     "Pending backtest"],
        ["Max Monthly Drawdown",    "≤8%",      "Enforced by kill-switch at 5%/day"],
        ["Signal Latency",          "&lt;2s",   "OANDA + indicators async pipeline"],
        ["Trade Duration Target",   "5–30 min", "M5 primary timeframe scalping"],
        ["Win Rate",                "≥55%",     "Required for positive Kelly fraction"],
        ["R:R per trade",           "≥2:1",     "Enforced by RiskManager (hard reject &lt;1:1)"],
    ]
    elements.append(kv_table(perf_rows, s, col_widths=[5.5 * cm, 3 * cm, 8.5 * cm]))
    elements.append(Spacer(1, 0.4 * cm))

    elements.append(Paragraph("Training Requirements", s["h2"]))
    train_items = [
        "Minimum 2 years of M5 data per pair (~140,000 candles per pair)",
        "LightGBM: ~5–10 min training per pair on CPU",
        "LSTM: ~30 epochs × 10 min on GPU (CUDA 12.1 available in this environment)",
        "RL (PPO): ~50,000 environment steps (~30 min on CPU)",
        "Retrain interval: 24 hours (configurable via ml.retrain_interval_hours)",
    ]
    for item in train_items:
        elements.append(bullet(item, s))

    elements.append(Spacer(1, 0.4 * cm))
    elements.append(HRFlowable(width="100%", color=MID))
    elements.append(Spacer(1, 0.2 * cm))
    elements.append(Paragraph(
        "ForexMind — Architecture Document  |  Version 1.0  |  March 2026",
        s["caption"]
    ))


# ── Main builder ──────────────────────────────────────────────────────────────

def build_pdf():
    doc = SimpleDocTemplate(
        OUTPUT,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="ForexMind Architecture",
        author="ForexMind AI",
        subject="System Architecture and Technical Design",
    )

    s = build_styles()
    elements = []

    cover_page(elements, s)
    toc_page(elements, s)
    section_overview(elements, s)
    section_arch_diagram(elements, s)
    section_modules(elements, s)
    section_data(elements, s)
    section_indicators(elements, s)
    section_ensemble(elements, s)
    section_risk(elements, s)
    section_backtest(elements, s)
    section_agent(elements, s)
    section_interfaces(elements, s)
    section_database(elements, s)
    section_config(elements, s)
    section_security(elements, s)
    section_performance(elements, s)

    doc.build(elements)
    print(f"✓ PDF written → {OUTPUT}")


if __name__ == "__main__":
    build_pdf()
