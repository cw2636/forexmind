# ForexMind — AI-Powered Forex Trading Agent

A production-grade, end-to-end AI forex trading system targeting **≥70% directional accuracy** on 1–5 minute charts across all major, minor, and exotic pairs.

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
Interfaces  (Rich CLI · FastAPI Web · Telegram Bot)
```

## Quick Start

```bash
# 1. Activate environment
source /path/to/mlenv/bin/activate

# 2. Install dependencies
pip install -r forexmind/requirements.txt

# 3. Configure API keys
cp .env.example .env
# Edit .env with your OANDA, Anthropic, Alpha Vantage, Finnhub keys

# 4. Train ML models (requires OANDA API key)
python -m forexmind.main train EUR_USD

# 5. Launch
python -m forexmind.main cli          # Terminal chat
python -m forexmind.main web          # Web dashboard → http://localhost:8000
python -m forexmind.main telegram     # Telegram bot
python -m forexmind.main signal EUR/USD  # Quick one-shot signal
```

## Run Tests

```bash
python -m pytest forexmind/tests/ -v   # 35 tests, all passing
```

## API Keys Required

| Service | URL | Cost |
|---------|-----|------|
| OANDA | https://www.oanda.com → Manage API Access | Free practice |
| Anthropic | https://console.anthropic.com | Pay-per-use |
| Alpha Vantage | https://www.alphavantage.co/support/#api-key | Free (25/day) |
| Finnhub | https://finnhub.io/register | Free tier |
| Telegram Bot | @BotFather on Telegram | Free |

## Project Structure

```
forexmind/
├── agents/          # Claude AI agent (LangChain 1.x bind_tools)
├── backtest/        # Event-driven backtesting + walk-forward + Monte Carlo
├── config/          # Settings singleton + config.yaml
├── data/            # OANDA client, news aggregator, SQLite ORM
├── indicators/      # 25+ technical indicators + signal scorer
├── interfaces/      # CLI (Rich), FastAPI web, Telegram bot
├── risk/            # Risk manager (Kelly sizing, kill-switch, trailing stops)
├── strategy/        # Rule-based, LightGBM, BiLSTM, PPO-RL, ensemble
├── tests/           # 35 pytest tests
└── utils/           # Helpers, logger, session times
docs/
├── ForexMind_Architecture.pdf
├── ForexMind_Architecture.pptx
└── ForexMind_Next_Steps.pdf
```

## Technology Stack

- **AI**: Anthropic Claude 3.5 Sonnet via LangChain 1.x
- **ML**: LightGBM · PyTorch BiLSTM · stable-baselines3 PPO
- **Data**: OANDA REST API · pandas-ta (pure Python indicators)
- **Web**: FastAPI + WebSocket
- **Bot**: python-telegram-bot v21+
- **DB**: SQLAlchemy 2 + aiosqlite (SQLite)

## ⚠️ Disclaimer

This software is for **educational purposes and paper trading only**. Always test thoroughly on a paper account before risking real capital. Past backtested performance does not guarantee future results.
