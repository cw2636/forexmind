"""
ForexMind — LangChain Tools for the Claude Agent
==================================================
Each tool wraps a core ForexMind module so Claude can call it
as part of its reasoning chain.

Tools:
  1. get_signal       — Run ensemble analysis on a forex pair
  2. get_news         — Fetch + summarise recent news for a pair
  3. get_account      — OANDA account snapshot
  4. run_backtest     — Run historical backtest
  5. place_trade      — Execute a paper trade (requires user confirmation)
  6. get_sessions     — Current market session status

Advanced Python:
  - Pydantic BaseModel schemas for LangChain tool input validation
  - async def tools for non-blocking I/O
  - try/except with structured error returns (never crash the agent)
"""

from __future__ import annotations

import json
from typing import Optional

try:
    from langchain_core.tools import StructuredTool
except ImportError:
    from langchain.tools import StructuredTool
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from forexmind.utils.logger import get_logger
from forexmind.utils.session_times import get_session_status, best_pairs_for_session
from forexmind.agents.prompts import (
    SIGNAL_TOOL_DESCRIPTION,
    NEWS_TOOL_DESCRIPTION,
    ACCOUNT_TOOL_DESCRIPTION,
    BACKTEST_TOOL_DESCRIPTION,
    PLACE_TRADE_TOOL_DESCRIPTION,
)

log = get_logger(__name__)


# ── Input schemas ─────────────────────────────────────────────────────────────

class SignalInput(BaseModel):
    instrument: str = Field(..., description="Forex pair in OANDA format, e.g. EUR_USD")
    timeframe: str = Field("M5", description="Timeframe: M1, M5, M15, H1")
    candles: int = Field(500, description="Number of historical candles to analyse")


class NewsInput(BaseModel):
    instrument: str = Field(..., description="Forex pair to get news for, e.g. EUR_USD")
    lookback_hours: int = Field(4, description="Hours of news history to include")


class AccountInput(BaseModel):
    pass   # No arguments needed


class BacktestInput(BaseModel):
    instrument: str = Field(..., description="Forex pair to backtest, e.g. EUR_USD")
    timeframe: str = Field("M5", description="Candle timeframe")
    start_date: str = Field("2024-01-01", description="Start date YYYY-MM-DD")
    end_date: str = Field("2024-12-31", description="End date YYYY-MM-DD")


class PlaceTradeInput(BaseModel):
    instrument: str = Field(..., description="Forex pair, e.g. EUR_USD")
    direction: str = Field(..., description="BUY or SELL")
    units: int = Field(..., description="Number of units (1 lot = 100,000 units)")
    stop_loss: float = Field(..., description="Stop-loss price level")
    take_profit: float = Field(..., description="Take-profit price level")


# ── Tool functions ─────────────────────────────────────────────────────────────

async def _get_signal(instrument: str, timeframe: str = "M5", candles: int = 500) -> str:
    """
    Core signal generation tool.
    Fetches live data, computes indicators, runs ensemble, returns JSON summary.
    """
    try:
        from forexmind.data.oanda_client import get_oanda_client
        from forexmind.indicators.engine import get_indicator_engine
        from forexmind.indicators.scorer import score_snapshot
        from forexmind.strategy.ensemble import get_ensemble
        from forexmind.data.news_aggregator import get_news_aggregator
        from forexmind.utils.helpers import format_price, price_to_pips, pip_size
        from forexmind.config.settings import get_settings

        # Fetch data
        client = get_oanda_client()
        price = await client.get_price(instrument)
        df = await client.get_candles(instrument, timeframe, candles)

        if df.empty:
            return json.dumps({"error": f"No price data available for {instrument}"})

        # Compute indicators
        engine = get_indicator_engine()
        df_ind = engine.compute(df)
        snap = engine.snapshot(df_ind, instrument, timeframe)
        score = score_snapshot(snap)

        # Fetch HTF context
        htf_df = None
        if timeframe in ("M1", "M5"):
            htf_df_raw = await client.get_candles(instrument, "H1", 100)
            if not htf_df_raw.empty:
                htf_df = engine.compute(htf_df_raw)

        # Ensemble signal
        ensemble = get_ensemble()
        sig = ensemble.generate_signal(df_ind, instrument, timeframe, price.mid, htf_df)

        # News sentiment
        news = get_news_aggregator()
        sentiment = news.get_instrument_sentiment(instrument, lookback_hours=4)

        # Session status
        session = get_session_status()

        ps = pip_size(instrument)
        stop_pips = price_to_pips(abs(sig.entry_price - sig.stop_loss), instrument)
        tp_pips = price_to_pips(abs(sig.take_profit - sig.entry_price), instrument)

        result = {
            "instrument": instrument,
            "timeframe": timeframe,
            "current_price": sig.entry_price,
            "bid": price.bid,
            "ask": price.ask,
            "spread_pips": round((price.ask - price.bid) / ps * 10, 2),

            # Signal
            "signal": {
                "action": sig.direction,
                "confidence": round(sig.confidence * 100, 1),
                "entry": sig.entry_price,
                "stop_loss": sig.stop_loss,
                "take_profit": sig.take_profit,
                "stop_loss_pips": round(stop_pips, 1),
                "take_profit_pips": round(tp_pips, 1),
                "risk_reward": round(sig.risk_reward_ratio if hasattr(sig, 'risk_reward_ratio') else tp_pips / stop_pips if stop_pips > 0 else 0, 2),
                "risk_pct": sig.risk_pct,
                "agreeing_strategies": f"{sig.agreeing_count}/{sig.total_strategies}",
                "buy_score": sig.buy_score,
                "sell_score": sig.sell_score,
            },

            # Key indicators
            "indicators": {
                "ema_trend": snap["ema_trend"],
                "rsi": round(snap["rsi"], 1),
                "rsi_zone": snap["rsi_zone"],
                "macd": round(snap["macd"], 5),
                "macd_hist": round(snap["macd_hist"], 5),
                "macd_cross": snap["macd_cross"],
                "adx": round(snap["adx"], 1),
                "adx_trend_strength": snap["adx_trend_strength"],
                "psar_signal": snap["psar_signal"],
                "stoch_k": round(snap["stoch_k"], 1),
                "stoch_cross": snap["stoch_cross"],
                "bb_position": round(snap["bb_position"], 2),
                "atr": round(snap["atr"], 5),
                "support": snap["support"],
                "resistance": snap["resistance"],
            },

            # Context
            "session": {
                "active": session.active_sessions,
                "overlaps": session.active_overlaps,
                "is_overlap": session.is_overlap,
                "score": session.session_score,
            },
            "news_sentiment": sentiment,
            "reasoning": sig.reasoning,
        }
        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        log.error(f"get_signal error: {e}")
        return json.dumps({"error": str(e), "instrument": instrument})


async def _get_news(instrument: str, lookback_hours: int = 4) -> str:
    """Fetch and summarise recent forex news."""
    try:
        from forexmind.data.news_aggregator import get_news_aggregator

        agg = get_news_aggregator()
        articles = await agg.fetch_all()
        sentiment = agg.get_instrument_sentiment(instrument, lookback_hours)

        relevant = [
            {
                "headline": a.headline,
                "source": a.source,
                "published": a.published_at.isoformat(),
                "sentiment": round(a.sentiment_score, 3),
                "impact": a.impact,
            }
            for a in articles
            if instrument in a.related_instruments
        ][:10]

        return json.dumps(
            {"instrument": instrument, "sentiment": sentiment, "articles": relevant},
            indent=2, default=str
        )
    except Exception as e:
        log.error(f"get_news error: {e}")
        return json.dumps({"error": str(e)})


async def _get_account() -> str:
    """Fetch OANDA account status."""
    try:
        from forexmind.data.oanda_client import get_oanda_client
        client = get_oanda_client()
        acc = await client.get_account()
        positions = await client.get_open_positions()
        from forexmind.risk.manager import get_risk_manager
        rm = get_risk_manager()
        return json.dumps({
            "balance": acc.balance,
            "nav": acc.nav,
            "unrealized_pnl": acc.unrealized_pnl,
            "margin_used": acc.margin_used,
            "margin_available": acc.margin_available,
            "open_trade_count": acc.open_trade_count,
            "daily_pnl_usd": rm.daily_pnl_usd,
            "kill_switch_active": rm.is_kill_switch_active,
            "open_positions": len(positions),
            "currency": acc.currency,
        }, indent=2)
    except Exception as e:
        log.error(f"get_account error: {e}")
        return json.dumps({"error": str(e)})


async def _run_backtest(
    instrument: str,
    timeframe: str = "M5",
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
) -> str:
    """Run historical backtest and return performance summary."""
    try:
        from forexmind.data.oanda_client import get_oanda_client
        from forexmind.backtest.engine import Backtester
        from forexmind.strategy.rule_based import RuleBasedStrategy
        from datetime import datetime, timezone

        client = get_oanda_client()
        start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
        end_dt = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)

        df = await client.get_candles(instrument, timeframe, from_dt=start_dt, to_dt=end_dt)
        if df.empty:
            return json.dumps({"error": "No historical data returned"})

        bt = Backtester()
        strategy = RuleBasedStrategy()
        result = bt.run(df, strategy, instrument, timeframe)

        return json.dumps({
            "instrument": result.instrument,
            "timeframe": result.timeframe,
            "period": f"{result.start_date} → {result.end_date}",
            "total_trades": result.total_trades,
            "win_rate": f"{result.win_rate:.1%}",
            "profit_factor": round(result.profit_factor, 2),
            "net_pnl_usd": round(result.total_pnl_usd, 2),
            "net_return_pct": round(result.net_return_pct, 2),
            "max_drawdown_pct": round(result.max_drawdown_pct, 2),
            "sharpe_ratio": round(result.sharpe_ratio, 2),
            "sortino_ratio": round(result.sortino_ratio, 2),
            "avg_trade_duration_min": round(result.avg_trade_duration_min, 1),
        }, indent=2)
    except Exception as e:
        log.error(f"run_backtest error: {e}")
        return json.dumps({"error": str(e)})


async def _place_trade(
    instrument: str,
    direction: str,
    units: int,
    stop_loss: float,
    take_profit: float,
) -> str:
    """Execute a trade on OANDA paper account."""
    try:
        from forexmind.data.oanda_client import get_oanda_client
        from forexmind.config.settings import get_settings

        cfg = get_settings()
        if not cfg.app.paper_trading:
            return json.dumps({
                "warning": "LIVE TRADING IS ENABLED. User must confirm explicitly.",
                "status": "pending_confirmation"
            })

        client = get_oanda_client()
        signed_units = units if direction == "BUY" else -units
        result = await client.market_order(
            instrument=instrument,
            units=signed_units,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        return json.dumps({
            "success": result.success,
            "trade_id": result.trade_id,
            "filled_price": result.filled_price,
            "units": result.units,
            "direction": direction,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "mode": "paper" if cfg.app.paper_trading else "LIVE",
            "message": result.message,
        }, indent=2)
    except Exception as e:
        log.error(f"place_trade error: {e}")
        return json.dumps({"error": str(e)})


def _get_sessions(_: str = "") -> str:
    """Get current forex market session status."""
    status = get_session_status()
    pairs = best_pairs_for_session()
    return json.dumps({
        "active_sessions": status.active_sessions,
        "active_overlaps": status.active_overlaps,
        "is_overlap": status.is_overlap,
        "is_weekend": status.is_weekend,
        "session_score": status.session_score,
        "recommended_pairs": pairs[:8],
        "trading_advice": (
            "HIGH LIQUIDITY — Prime scalping window!" if status.is_overlap
            else "Normal liquidity" if status.active_sessions
            else "Low liquidity — Exercise caution"
        )
    }, indent=2)


# ── Build tool list ────────────────────────────────────────────────────────────

def build_tools() -> list[StructuredTool]:
    """
    Construct and return all LangChain tools.
    Called once at agent initialisation.
    """
    return [
        StructuredTool.from_function(
            coroutine=_get_signal,
            name="get_signal",
            description=SIGNAL_TOOL_DESCRIPTION,
            args_schema=SignalInput,
        ),
        StructuredTool.from_function(
            coroutine=_get_news,
            name="get_news",
            description=NEWS_TOOL_DESCRIPTION,
            args_schema=NewsInput,
        ),
        StructuredTool.from_function(
            coroutine=_get_account,
            name="get_account",
            description=ACCOUNT_TOOL_DESCRIPTION,
            args_schema=AccountInput,
        ),
        StructuredTool.from_function(
            coroutine=_run_backtest,
            name="run_backtest",
            description=BACKTEST_TOOL_DESCRIPTION,
            args_schema=BacktestInput,
        ),
        StructuredTool.from_function(
            coroutine=_place_trade,
            name="place_trade",
            description=PLACE_TRADE_TOOL_DESCRIPTION,
            args_schema=PlaceTradeInput,
        ),
        StructuredTool.from_function(
            func=_get_sessions,
            name="get_sessions",
            description="Get current forex market session status and recommended pairs",
        ),
    ]
