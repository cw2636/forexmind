# ForexMind — AI-Powered Forex Trading Agent

A production-grade, end-to-end AI forex trading system that scans the market during London and US sessions, fires Telegram alerts, and automatically executes high-confidence trades on your OANDA practice (or live) account.

**Target:** ≥70% directional accuracy on M5 charts across all major pairs.

---

## Architecture

```
External Sources (OANDA REST + Alpha Vantage + Finnhub)
        ↓
Data Layer  (OandaClient · NewsAggregator · SQLite)
        ↓
Indicators Engine (25+ indicators via pandas-ta)
        ↓
Signal Scorer  [−100 → +100]
        ↓
Ensemble Strategy (Rule-Based 30% · LightGBM 25% · BiLSTM 25% · PPO-RL 20%)
        ↓
Risk Manager  (Kill-switch · ATR stops · Kelly sizing)
        ↓
Claude AI Agent  (LangChain 1.x bind_tools · 6 tools)
        ↓
Interfaces  (Rich CLI · FastAPI Web · Telegram Bot · Scheduler)
```

---

## Features

- **Automated Scheduler** — wakes up at London open (07:00 UTC), scans every 15 min through US close (21:00 UTC)
- **Smart alerting** — 65–79% confidence → Telegram alert (manual); ≥80% confidence → auto-executed trade with SL/TP
- **Ensemble ML signals** — LightGBM + BiLSTM + rule-based + PPO-RL strategies voting together
- **Risk manager** — Kelly criterion position sizing, ATR-based stops, daily loss kill-switch
- **Full DB history** — every signal and trade saved to SQLite for performance analysis
- **News sentiment** — real-time Finnhub + Alpha Vantage news scored and factored into signals
- **Session awareness** — only trades during active sessions; flags London/NY overlap (peak liquidity)
- **Paper trading safe** — OANDA practice account with $100k virtual money

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/cw2636/forexmind.git
cd forexmind

# 2. Create and activate virtual environment
python3 -m venv mlenv
source mlenv/bin/activate

# 3. Install dependencies
pip install -r forexmind/requirements.txt

# 4. Configure API keys
cp .env.example .env
# Edit .env — fill in OANDA, Anthropic, Finnhub, Telegram keys

# 5. Create database directory
mkdir -p data

# 6. Train ML models (one-time, ~10 min)
python forexmind/main.py train EUR_USD
python forexmind/main.py train GBP_USD

# 7. Launch everything
python forexmind/main.py web          # FastAPI dashboard → http://localhost:8000
python forexmind/main.py telegram     # Telegram bot (in a separate terminal)
python forexmind/main.py scheduler    # Auto-scanner + Telegram alerts (separate terminal)
python forexmind/main.py cli          # Interactive Claude chat (optional)
```

---

## All Modes

| Command | Description |
|---|---|
| `python forexmind/main.py cli` | Interactive Claude AI terminal chat |
| `python forexmind/main.py web` | FastAPI web dashboard + WebSocket signals |
| `python forexmind/main.py telegram` | Telegram bot (commands + free chat) |
| `python forexmind/main.py scheduler` | Auto-scanner — alerts + auto-trades |
| `python forexmind/main.py signal EUR/USD` | One-shot signal, no agent |
| `python forexmind/main.py backtest GBP/USD` | 1-year backtest |
| `python forexmind/main.py train EUR_USD` | Train/retrain LightGBM + LSTM |
| `python forexmind/main.py all` | Web + Telegram simultaneously |

---

## Telegram Bot Commands

| Command | What it does |
|---|---|
| `/signal EUR/USD` | Full signal with entry, SL, TP, confidence |
| `/signals` | Top signals for all recommended pairs right now |
| `/account` | OANDA account snapshot (balance, P&L, open trades) |
| `/sessions` | Current market sessions + recommended pairs |
| `/help` | List all commands |
| Any free text | Chat with Claude AI agent |

---

## Scheduler Behaviour

The scheduler runs 07:00–21:00 UTC (London open → US close) and:

| Confidence | Action |
|---|---|
| < 65% | Silent |
| 65–79% | 📢 Telegram alert — "Review and place manually" |
| ≥ 80% | ⚡ Auto-executes trade with SL/TP + Telegram confirmation |

**Tunable parameters** in `forexmind/scheduler.py`:
```python
SCAN_INTERVAL_MINUTES = 15    # How often to scan
MIN_CONFIDENCE = 65.0         # Minimum % for alert
AUTO_TRADE_CONFIDENCE = 80.0  # Minimum % for auto-trade
MIN_RR = 1.5                  # Minimum risk:reward ratio
COOLDOWN_MINUTES = 45         # Re-alert cooldown per pair
```

---

## API Keys Required

| Service | URL | Cost |
|---|---|---|
| OANDA | https://www.oanda.com → My Account → Manage API Access | Free practice |
| Anthropic | https://console.anthropic.com | Pay-per-use (~$8/mo moderate use) |
| Alpha Vantage | https://www.alphavantage.co/support/#api-key | Free (25 req/day) |
| Finnhub | https://finnhub.io/register | Free tier |
| Telegram Bot | @BotFather on Telegram | Free |

**Recommended Claude model:** `claude-sonnet-4-5` (best cost/quality balance)

---

## Project Structure

```
forexmind/
├── agents/          # Claude AI agent (LangChain 1.x bind_tools, 6 tools)
├── backtest/        # Event-driven backtesting engine
├── config/          # Settings singleton (pydantic) + config.yaml
├── data/            # OANDA client, news aggregator, SQLite ORM models
├── indicators/      # 25+ technical indicators + signal scorer [-100, +100]
├── interfaces/
│   ├── cli.py       # Rich terminal UI
│   ├── telegram_bot.py  # Telegram bot (python-telegram-bot v21+)
│   └── web/         # FastAPI + WebSocket dashboard
├── models/          # Trained ML model files (gitignored, train locally)
├── risk/            # Risk manager (Kelly sizing, kill-switch, ATR stops)
├── scheduler.py     # Automated signal scanner + Telegram push alerts
├── strategy/        # Rule-based, LightGBM, BiLSTM, PPO-RL, ensemble
├── tests/           # pytest test suite
└── utils/           # Logger, session times, pip calculations
data/
└── forexmind.db     # SQLite database (gitignored)
docs/
├── generate_architecture_pdf.py
├── generate_architecture_pptx.py
└── generate_next_steps.py
```

---

## Database Schema

Every signal and trade is persisted automatically by the scheduler:

| Table | Contents |
|---|---|
| `candles` | OHLCV price data (instrument, timeframe, OHLCV) |
| `signals` | Every generated signal (direction, confidence, entry, SL, TP) |
| `trades` | Every executed trade (units, filled price, status, P&L) |
| `news_articles` | Aggregated news with sentiment scores |
| `economic_events` | Economic calendar entries |

---

## ML Models

Models are trained locally and saved to `forexmind/models/` (gitignored):

| Model | File | Notes |
|---|---|---|
| LightGBM | `lgbm_forex.pkl` | Fast, interpretable, 77%+ accuracy on M5 |
| BiLSTM | `lstm_forex.pt` | Sequence model, improves with more data |
| PPO-RL | `ppo_forex.zip` | Reinforcement learning (optional, `train` mode) |

Retrain monthly:
```bash
python forexmind/main.py train EUR_USD
python forexmind/main.py train GBP_USD
```

---

## Technology Stack

| Category | Libraries |
|---|---|
| AI / LLM | Anthropic Claude via LangChain 1.x |
| ML | LightGBM · PyTorch BiLSTM · stable-baselines3 PPO |
| Data | OANDA REST (oandapyV20) · pandas-ta · aiohttp |
| Web | FastAPI · uvicorn · WebSocket |
| Bot | python-telegram-bot v21+ |
| DB | SQLAlchemy 2.0 + aiosqlite (async SQLite) |
| CLI | Rich · asyncio |

---

## Run Tests

```bash
python -m pytest forexmind/tests/ -v
```

---

## ⚠️ Disclaimer

This software is for **educational purposes and paper trading only**. Set `PAPER_TRADING=True` in `.env` until you are consistently profitable. Always test thoroughly on a practice account before risking real capital. Past backtested performance does not guarantee future results. The authors are not responsible for any financial losses.

