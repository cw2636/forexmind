"""
Generate ForexMind Next Steps PDF
===================================
Run:  python docs/generate_next_steps.py
Output: docs/ForexMind_Next_Steps.pdf
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
import os

OUTPUT = os.path.join(os.path.dirname(__file__), "ForexMind_Next_Steps.pdf")

NAVY  = colors.HexColor("#0D1B2A")
BLUE  = colors.HexColor("#1A73E8")
TEAL  = colors.HexColor("#00BFA5")
GOLD  = colors.HexColor("#F9A825")
LIGHT = colors.HexColor("#E8F0FE")
GREY  = colors.HexColor("#F5F5F5")
MID   = colors.HexColor("#90A4AE")
RED   = colors.HexColor("#E53935")
GREEN = colors.HexColor("#43A047")
WHITE = colors.white
DARK  = colors.HexColor("#212121")


def styles():
    s = {}
    s["h1"] = ParagraphStyle("h1", fontSize=16, leading=22, textColor=WHITE,
        fontName="Helvetica-Bold", spaceBefore=0, spaceAfter=0)
    s["h2"] = ParagraphStyle("h2", fontSize=13, leading=18, textColor=BLUE,
        fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6)
    s["h3"] = ParagraphStyle("h3", fontSize=11, leading=15, textColor=NAVY,
        fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=4)
    s["body"] = ParagraphStyle("body", fontSize=10, leading=15, textColor=DARK,
        fontName="Helvetica", spaceAfter=5, alignment=TA_JUSTIFY)
    s["bullet"] = ParagraphStyle("bullet", fontSize=10, leading=14, textColor=DARK,
        fontName="Helvetica", spaceAfter=3, leftIndent=14, firstLineIndent=-14)
    s["code"] = ParagraphStyle("code", fontSize=9, leading=13, textColor=DARK,
        fontName="Courier", backColor=GREY, borderPad=6,
        leftIndent=12, rightIndent=12, spaceAfter=8)
    s["caption"] = ParagraphStyle("caption", fontSize=9, leading=12, textColor=MID,
        fontName="Helvetica-Oblique", alignment=TA_CENTER)
    s["table_body"] = ParagraphStyle("tb", fontSize=9, leading=13, textColor=DARK,
        fontName="Helvetica")
    s["status_done"] = ParagraphStyle("done", fontSize=10, leading=14, textColor=GREEN,
        fontName="Helvetica-Bold")
    s["status_next"] = ParagraphStyle("next", fontSize=10, leading=14, textColor=BLUE,
        fontName="Helvetica-Bold")
    s["status_future"] = ParagraphStyle("future", fontSize=10, leading=14, textColor=MID,
        fontName="Helvetica-Bold")
    return s


def banner(text, s, bg=NAVY):
    p = ParagraphStyle("banner", fontSize=14, leading=18, textColor=WHITE,
                        fontName="Helvetica-Bold")
    data = [[Paragraph(text, p)]]
    tbl = Table(data, colWidths=[17 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), bg),
        ("TOPPADDING",    (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ("LEFTPADDING",   (0, 0), (-1, -1), 14),
    ]))
    return tbl


def bullet(text, s, color=DARK):
    sty = ParagraphStyle("bl", fontSize=10, leading=14, textColor=color,
        fontName="Helvetica", spaceAfter=3, leftIndent=14, firstLineIndent=-14)
    return Paragraph(f"<bullet>&bull;</bullet> {text}", sty)


def task_table(rows, s):
    """Priority task table: Phase | Task | Why | How | Effort"""
    tbl_rows = [[Paragraph(str(c), s["table_body"]) for c in row] for row in rows]
    tbl = Table(tbl_rows, colWidths=[2 * cm, 4.5 * cm, 4 * cm, 4 * cm, 2.5 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), BLUE),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, GREY]),
        ("GRID",          (0, 0), (-1, -1), 0.4, MID),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 7),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    return tbl


def phase_box(elements, num, title, color, s, items):
    """Renders a coloured phase header + bullet list."""
    hdr_data = [[
        Paragraph(f"Phase {num}", ParagraphStyle("pn", fontSize=10, textColor=WHITE,
                  fontName="Helvetica-Bold")),
        Paragraph(title, ParagraphStyle("pt", fontSize=12, textColor=WHITE,
                  fontName="Helvetica-Bold")),
    ]]
    hdr = Table(hdr_data, colWidths=[2 * cm, 15 * cm])
    hdr.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), color),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
    ]))
    elements.append(hdr)
    for item in items:
        status_icon, text = item
        elements.append(Paragraph(
            f"&nbsp;&nbsp;&nbsp;{status_icon} &nbsp; {text}",
            ParagraphStyle("pi", fontSize=10, leading=15, textColor=DARK,
                           fontName="Helvetica", spaceAfter=2, leftIndent=10)
        ))
    elements.append(Spacer(1, 0.2 * cm))


def build_pdf():
    doc = SimpleDocTemplate(
        OUTPUT, pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="ForexMind Next Steps",
        author="ForexMind AI",
    )

    s = styles()
    elements = []

    # ── Cover ─────────────────────────────────────────────────────────────────
    cover_data = [
        [Paragraph("ForexMind", ParagraphStyle("ct", fontSize=32, leading=40,
            textColor=WHITE, fontName="Helvetica-Bold", alignment=TA_CENTER))],
        [Paragraph("Next Steps & Roadmap", ParagraphStyle("cs", fontSize=20, leading=26,
            textColor=GOLD, fontName="Helvetica-Bold", alignment=TA_CENTER))],
        [Paragraph("Action plan to take ForexMind from working prototype to production trading system",
            ParagraphStyle("cd", fontSize=12, leading=17, textColor=LIGHT,
            fontName="Helvetica", alignment=TA_CENTER))],
        [Paragraph("March 2026", ParagraphStyle("cv", fontSize=10, leading=14,
            textColor=MID, fontName="Helvetica", alignment=TA_CENTER))],
    ]
    cover_tbl = Table(cover_data, colWidths=[17 * cm])
    cover_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), NAVY),
        ("TOPPADDING",    (0, 0), (-1, -1), 20),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 20),
        ("LEFTPADDING",   (0, 0), (-1, -1), 20),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 20),
    ]))
    elements.append(cover_tbl)
    elements.append(Spacer(1, 0.6 * cm))

    # Current status
    status_rows = [
        ["Item",                         "Status"],
        ["Project location",             "/home/wilson/Forex/forexmind/"],
        ["Python environment",            "mlenv (Python 3.12 + CUDA 12.1)"],
        ["All modules import cleanly",   "✅ 20/20 modules verified"],
        ["Test suite",                   "✅ 35/35 tests passing"],
        ["Dependencies installed",       "✅ All packages in requirements.txt"],
        ["YAML config",                  "✅ Fixed (new_york spacing issue resolved)"],
        ["LangChain 1.x migration",      "✅ Rewritten for bind_tools pattern"],
        ["CLI / Web / Telegram",         "✅ All interfaces importable and wired up"],
        ["API keys configured",          "⏳ Placeholder values — needs real keys in .env"],
        ["ML models trained",            "⏳ OANDA data not yet fetched; models not trained"],
        ["Live trading tested",          "⏳ Pending API keys + model training"],
    ]
    elements.append(banner("Current Status (as of March 28, 2026)", s))
    elements.append(Spacer(1, 0.2 * cm))
    styled = [[Paragraph(str(c), s["table_body"]) for c in row] for row in status_rows]
    tbl = Table(styled, colWidths=[7 * cm, 10 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), BLUE),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, GREY]),
        ("GRID",          (0, 0), (-1, -1), 0.4, MID),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ]))
    elements.append(tbl)
    elements.append(PageBreak())

    # ── Phase Roadmap ─────────────────────────────────────────────────────────
    elements.append(banner("Phased Implementation Roadmap", s))
    elements.append(Spacer(1, 0.3 * cm))

    phase_box(elements, 1, "Immediate — Configure API Keys (1–2 hours)", BLUE, s, [
        ("🔑", "Sign up at oanda.com → My Account → Manage API Access → set OANDA_API_KEY + OANDA_ACCOUNT_ID"),
        ("🔑", "Get Anthropic key: console.anthropic.com → API Keys → set ANTHROPIC_API_KEY"),
        ("🔑", "Get Alpha Vantage: alphavantage.co/support/#api-key (free) → set ALPHA_VANTAGE_API_KEY"),
        ("🔑", "Get Finnhub: finnhub.io/register (free) → set FINNHUB_API_KEY"),
        ("🔑", "Optional: Create Telegram bot via @BotFather → set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID"),
        ("📝", "Edit /home/wilson/Forex/.env — replace all 'your_..._here' placeholder values"),
        ("✅", "Verify: python -m forexmind.main signal EUR/USD   (should show a live signal)"),
    ])

    phase_box(elements, 2, "Short Term — Train ML Models (2–4 hours first time)", TEAL, s, [
        ("🤖", "Train LightGBM + LSTM for EUR_USD:   python -m forexmind.main train EUR_USD"),
        ("🤖", "Train remaining major pairs: USD_JPY, GBP_USD, USD_CHF, USD_CAD, AUD_USD, NZD_USD"),
        ("🤖", "Train minor pairs: EUR_GBP, EUR_JPY, GBP_JPY, AUD_JPY, EUR_AUD"),
        ("⚙️", "Models saved to /home/wilson/Forex/forexmind/models/{pair}_lgbm.pkl and lstm.pt"),
        ("📊", "Review LightGBM feature importance: inspect models for which indicators matter most"),
        ("🔄", "Schedule daily retraining (configured via ml.retrain_interval_hours=24 in config.yaml)"),
    ])

    phase_box(elements, 3, "Short Term — Run Backtests & Validate (1 day)", GOLD, s, [
        ("📈", "Backtest EUR_USD:   python -m forexmind.main backtest EUR_USD"),
        ("📈", "Review: win_rate ≥55%, profit_factor ≥1.5, sharpe_ratio ≥1.0, max_drawdown ≤15%"),
        ("📊", "Run walk-forward validation (built into Backtester.walk_forward())"),
        ("🎲", "Run Monte Carlo stress test (built into Backtester.monte_carlo())"),
        ("⚠️", "If metrics are poor: adjust ensemble weights in config.yaml or increase ADX filter threshold"),
        ("🔁", "Iterate: retrain with more data, wider stop multipliers, or different indicator combos"),
    ])

    phase_box(elements, 4, "Short Term — Paper Trading (1–2 weeks)", GREEN, s, [
        ("🚀", "Start CLI agent:   python -m forexmind.main cli"),
        ("💬", "Ask: 'Analyse EUR_USD for a scalping opportunity'"),
        ("💬", "Ask: 'Show me your signal for GBP_USD on M5'"),
        ("📱", "Start Telegram bot and interact from your phone"),
        ("🌐", "Start web dashboard: python -m forexmind.main web  →  http://localhost:8000"),
        ("📓", "Keep a trading journal — note when Claude is right/wrong and why"),
        ("⏱️", "Paper trade for at least 2 weeks. Target: ≥70% directional accuracy"),
    ])
    elements.append(PageBreak())

    phase_box(elements, 5, "Medium Term — RL Model Training (3–5 days)", NAVY, s, [
        ("🎮", "PPO requires more steps — run on full 2yr dataset per pair"),
        ("⚡", "Use CUDA (already available in mlenv): set device='cuda' in rl_strategy.py"),
        ("📊", "ForexTradingEnv reward tuning: adjust overtrading penalty coefficient"),
        ("🔬", "Hyperparameter search: try different PPO learning rates, n_steps, clip_range"),
        ("💾", "RL model saved to models/{pair}_ppo.zip by stable-baselines3"),
        ("🧪", "Compare RL signal quality vs rule-based + LightGBM in backtests"),
    ])

    phase_box(elements, 6, "Medium Term — Production Hardening", BLUE, s, [
        ("🔒", "Add WebSocket authentication to FastAPI (JWT token or API key header)"),
        ("📊", "Add a live P&L dashboard to the web interface (plotly charts)"),
        ("📅", "Integrate Forex Factory economic calendar API for event-based signal suppression"),
        ("🔔", "Telegram: push automatic alerts when high-confidence signal appears"),
        ("📧", "Add email/Pushover notifications as backup alert channel"),
        ("🐳", "Dockerize the application for deployment on any server"),
        ("🔄", "APScheduler: auto-retrain models daily at 00:00 UTC (off-market)"),
    ])

    phase_box(elements, 7, "Medium Term — ML Improvements", TEAL, s, [
        ("🧠", "Experiment with Transformer-based models (replace LSTM with attention)"),
        ("📊", "Add volume profile features (not available in all OANDA instruments)"),
        ("🌡️", "Add implied volatility proxy (ATR percentile over 252 days)"),
        ("🔀", "Investigate XGBoost as alternative to LightGBM for comparison"),
        ("📉", "Add regime detection (HMM or clustering) — different models for trending vs ranging"),
        ("🤖", "Online learning: incrementally update LightGBM with new daily data"),
    ])

    phase_box(elements, 8, "Long Term — Live Trading Readiness", RED, s, [
        ("⚠️", "ONLY proceed after ≥3 months of paper trading with consistent ≥70% accuracy"),
        ("⚠️", "Reduce max_risk_per_trade_pct to 0.5% for initial live deployment"),
        ("⚠️", "Set PAPER_TRADING=false in .env — this enables REAL money execution"),
        ("🏦", "Fund OANDA account with amount you can afford to lose completely"),
        ("📊", "Monitor daily: check kill-switch status, daily P&L, open positions"),
        ("🔒", "Set hard monthly loss budget: if −10%, stop all trading and review"),
        ("📈", "Scale position size ONLY after 6 months of consistent profitability"),
    ])
    elements.append(PageBreak())

    # ── Detailed Next Steps: Individual Actions ───────────────────────────────
    elements.append(banner("Detailed Action Items — Priority Order", s))
    elements.append(Spacer(1, 0.3 * cm))

    elements.append(Paragraph("Step 1: Add Your API Keys", s["h2"]))
    elements.append(Paragraph(
        "Open <b>/home/wilson/Forex/.env</b> and replace all placeholder values:", s["body"]
    ))
    elements.append(Paragraph(
        "# OANDA (free practice account)\n"
        "OANDA_API_KEY=your_real_key_here\n"
        "OANDA_ACCOUNT_ID=001-001-12345678-001\n\n"
        "# Anthropic (claude-3-5-sonnet)\n"
        "ANTHROPIC_API_KEY=sk-ant-api03-...\n\n"
        "# News (both free)\n"
        "ALPHA_VANTAGE_API_KEY=ABCDEFGH\n"
        "FINNHUB_API_KEY=xxxxx\n\n"
        "# Leave as true until you're confident\n"
        "PAPER_TRADING=true",
        s["code"]
    ))

    elements.append(Paragraph("Step 2: Verify Connection", s["h2"]))
    elements.append(Paragraph(
        "Run the following to confirm everything can reach the APIs:", s["body"]
    ))
    elements.append(Paragraph(
        "source /home/wilson/ml-workspace/mlenv/bin/activate\n"
        "cd /home/wilson/Forex\n"
        "python -m forexmind.main signal EUR/USD\n"
        "# Expected output: live bid/ask + indicator readings + ensemble HOLD signal\n"
        "# (HOLD because ML models aren't trained yet)",
        s["code"]
    ))

    elements.append(Paragraph("Step 3: Train ML Models", s["h2"]))
    elements.append(Paragraph(
        "The train command fetches 2 years of M5 OANDA history, builds the feature matrix, "
        "trains LightGBM (cross-validated) and BiLSTM (30 epochs):", s["body"]
    ))
    elements.append(Paragraph(
        "# Train EUR_USD first (most liquid, best data quality)\n"
        "python -m forexmind.main train EUR_USD\n\n"
        "# Then train other priority pairs\n"
        "for pair in USD_JPY GBP_USD USD_CHF USD_CAD AUD_USD; do\n"
        "    python -m forexmind.main train $pair\n"
        "done\n\n"
        "# Models saved to: forexmind/models/",
        s["code"]
    ))
    elements.append(Paragraph(
        "<b>Tip:</b> Training uses the CUDA 12.1 GPU already available in your mlenv — "
        "LSTM training will be ~10× faster than CPU. Expect ~10–20 minutes per pair total.",
        s["body"]
    ))

    elements.append(Paragraph("Step 4: Launch the Agent", s["h2"]))
    elements.append(Paragraph(
        "Three ways to interact with ForexMind:", s["body"]
    ))
    elements.append(Paragraph(
        "# Terminal (best for learning — see all Claude's reasoning)\n"
        "python -m forexmind.main cli\n\n"
        "# Web dashboard (best for monitoring multiple pairs)\n"
        "python -m forexmind.main web\n"
        "# Then open: http://localhost:8000\n\n"
        "# All three simultaneously\n"
        "python -m forexmind.main all",
        s["code"]
    ))
    elements.append(PageBreak())

    # ── Questions to explore with Claude ─────────────────────────────────────
    elements.append(banner("Suggested Questions to Ask ForexMind", s))
    elements.append(Spacer(1, 0.3 * cm))

    elements.append(Paragraph("For Signal Analysis", s["h2"]))
    questions_signal = [
        '"Analyse EUR/USD on M5 — should I enter a trade right now?"',
        '"What is the current trend on GBP/USD? Show me the key support and resistance levels."',
        '"Is this a good time to trade? Which session are we in and which pairs are most liquid?"',
        '"The rule-based strategy says BUY but LightGBM says HOLD. Which should I trust?"',
        '"EUR/USD just broke above the 50 EMA — is this a valid breakout or a fakeout?"',
    ]
    for q in questions_signal:
        elements.append(bullet(q, s, color=BLUE))
    elements.append(Spacer(1, 0.2 * cm))

    elements.append(Paragraph("For Learning Advanced Trading", s["h2"]))
    questions_learn = [
        '"Explain how the ensemble voting works. Why did 2 strategies agree on BUY but you said HOLD?"',
        '"What does an ADX above 25 tell me about EUR/USD right now?"',
        '"Walk me through the full risk calculation for a BUY trade on USD/JPY."',
        '"What is the Kelly Criterion and how does it determine my position size?"',
        '"My backtest shows a 62% win rate but the profit factor is only 1.1. What\'s wrong?"',
        '"How does the LSTM model differ from the rule-based strategy in what it looks for?"',
    ]
    for q in questions_learn:
        elements.append(bullet(q, s, color=TEAL))
    elements.append(Spacer(1, 0.2 * cm))

    elements.append(Paragraph("For Risk & Account Management", s["h2"]))
    questions_risk = [
        '"What is my current account status and daily P&L?"',
        '"Run a backtest on EUR/USD for 2024 and tell me if the strategy has a statistical edge."',
        '"My last three trades were losses. Should I reduce position size?"',
        '"I want to risk 1% per trade instead of the default. Is that reasonable at my win rate?"',
    ]
    for q in questions_risk:
        elements.append(bullet(q, s, color=GOLD))
    elements.append(Spacer(1, 0.4 * cm))

    # ── Improvement Backlog ───────────────────────────────────────────────────
    elements.append(banner("Technical Improvement Backlog", s, bg=TEAL))
    elements.append(Spacer(1, 0.3 * cm))

    backlog = [
        ["Priority", "Feature", "Effort", "Impact"],
        ["HIGH", "Economic calendar integration (Forex Factory)",   "2 days",  "Avoid trading during high-impact news"],
        ["HIGH", "Live P&L plotly chart in web dashboard",          "1 day",   "Real-time performance visibility"],
        ["HIGH", "Auto-push Telegram alerts for strong signals",    "4 hours", "Mobile action on opportunities"],
        ["HIGH", "Retrain scheduler (APScheduler)",                 "2 hours", "Models stay current with market"],
        ["MED",  "WebSocket authentication (JWT)",                  "1 day",   "Security before any server deployment"],
        ["MED",  "Transformer/Attention model (replace LSTM)",      "3 days",  "Potentially better accuracy"],
        ["MED",  "Regime detection (trending vs ranging filter)",   "2 days",  "Avoid whipsaws in low-ADX markets"],
        ["MED",  "XGBoost comparison to LightGBM",                  "1 day",   "Validate best gradient-boost model"],
        ["MED",  "Docker containerization",                         "4 hours", "Easy deployment on VPS/cloud"],
        ["LOW",  "Pushover/email backup notifications",             "2 hours", "Redundant alert channel"],
        ["LOW",  "Multi-timeframe confluence checker",              "2 days",  "Reduce false signals"],
        ["LOW",  "Trade screenshot to Telegram with chart annotation", "2 days","Visual signal sharing"],
    ]
    styled = [[Paragraph(str(c), s["table_body"]) for c in row] for row in backlog]
    tbl = Table(styled, colWidths=[2 * cm, 7 * cm, 2.5 * cm, 5.5 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), BLUE),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, GREY]),
        ("GRID",          (0, 0), (-1, -1), 0.4, MID),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 7),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(tbl)
    elements.append(PageBreak())

    # ── Useful Commands Reference ─────────────────────────────────────────────
    elements.append(banner("Quick Reference — Commands", s))
    elements.append(Spacer(1, 0.3 * cm))

    elements.append(Paragraph("Environment Setup", s["h2"]))
    elements.append(Paragraph(
        "source /home/wilson/ml-workspace/mlenv/bin/activate\n"
        "cd /home/wilson/Forex",
        s["code"]
    ))

    elements.append(Paragraph("Running ForexMind", s["h2"]))
    elements.append(Paragraph(
        "python -m forexmind.main cli                # Terminal chat\n"
        "python -m forexmind.main web                # Web dashboard: localhost:8000\n"
        "python -m forexmind.main telegram           # Telegram bot\n"
        "python -m forexmind.main all                # Web + Telegram\n"
        "python -m forexmind.main signal EUR/USD     # One-shot signal\n"
        "python -m forexmind.main backtest GBP/USD   # 1-year backtest\n"
        "python -m forexmind.main train EUR_USD      # Train LightGBM + LSTM",
        s["code"]
    ))

    elements.append(Paragraph("Tests", s["h2"]))
    elements.append(Paragraph(
        "python -m pytest forexmind/tests/ -v        # Run all 35 tests\n"
        "python -m pytest forexmind/tests/ -v -k indicators  # Run specific tests",
        s["code"]
    ))

    elements.append(Paragraph("Check Import Health", s["h2"]))
    elements.append(Paragraph(
        "python -c \"from forexmind.config.settings import get_settings; print(get_settings().pairs)\"",
        s["code"]
    ))

    elements.append(Paragraph("Model File Locations", s["h2"]))
    elements.append(Paragraph(
        "forexmind/models/EUR_USD_lgbm.pkl    # LightGBM model\n"
        "forexmind/models/EUR_USD_lstm.pt     # LSTM checkpoint\n"
        "forexmind/models/EUR_USD_ppo.zip     # RL model (after RL training)\n"
        "forexmind/data/forexmind.db          # SQLite database",
        s["code"]
    ))

    elements.append(Spacer(1, 0.4 * cm))
    elements.append(HRFlowable(width="100%", color=MID))
    elements.append(Spacer(1, 0.2 * cm))
    elements.append(Paragraph(
        "ForexMind — Next Steps  |  Version 1.0  |  March 2026  |  "
        "Remember: start on paper trading. Prove the edge before risking real capital.",
        s["caption"]
    ))

    doc.build(elements)
    print(f"✓ Next Steps PDF written → {OUTPUT}")


if __name__ == "__main__":
    build_pdf()
