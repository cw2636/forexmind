"""
ForexMind — FastAPI Web Application
======================================
REST + WebSocket server providing:
  GET  /health                 — Health check
  GET  /api/signal/{pair}      — Get trading signal for a pair
  GET  /api/signals            — Get signals for all pairs
  GET  /api/account            — Account snapshot
  GET  /api/news/{pair}        — News for a pair
  GET  /api/sessions           — Current session status
  POST /api/chat               — Chat with the Claude agent
  WS   /ws/signals             — Live signal stream via WebSocket
  GET  /                       — HTML dashboard

Advanced Python:
  - FastAPI with async path operations
  - WebSocket streaming (real-time signals pushed to browser)
  - Pydantic response models for API contracts
  - Lifespan context manager for startup/shutdown tasks
  - Background tasks for periodic signal updates
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from forexmind.data.database import init_db, close_db
from forexmind.utils.logger import get_logger
from forexmind.utils.session_times import get_session_status

log = get_logger(__name__)


# ── Pydantic models ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    response: str
    signal: dict | None = None
    timestamp: str


class TradeRequest(BaseModel):
    instrument: str
    direction: str          # "BUY" or "SELL"
    entry: float
    stop_loss: float
    take_profit: float
    atr: float = 0.0005
    confidence: float = 0.70


# ── WebSocket connection manager ──────────────────────────────────────────────

class ConnectionManager:
    """Manages active WebSocket connections for live signal broadcasting."""

    def __init__(self) -> None:
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)
        log.debug(f"WS connected. Total: {len(self.active)}")

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)
        log.debug(f"WS disconnected. Total: {len(self.active)}")

    async def broadcast(self, message: str) -> None:
        dead: list[WebSocket] = []
        for ws in self.active:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


ws_manager = ConnectionManager()

# Simple TTL cache for /api/signals — avoids hammering OANDA+ML on every refresh
_signals_cache: dict | None = None
_signals_cache_time: float = 0
SIGNALS_CACHE_TTL = 60  # seconds


# ── Background task: periodic signal updates ─────────────────────────────────

async def _signal_broadcast_loop() -> None:
    """
    Background coroutine: generates signals every 60 seconds
    and broadcasts them to all connected WebSocket clients.
    """
    from forexmind.config.settings import get_settings
    cfg = get_settings()

    while True:
        await asyncio.sleep(60)
        if not ws_manager.active:
            continue

        session = get_session_status()
        if session.is_weekend or session.session_score < 0.4:
            continue

        try:
            from forexmind.agents.tools import _get_signal
            # Broadcast signals for top 3 recommended pairs
            from forexmind.utils.session_times import best_pairs_for_session
            pairs = best_pairs_for_session()[:3]
            for pair in pairs:
                result = await _get_signal(pair, "M15", 200)
                payload = {"type": "signal", "data": json.loads(result), "timestamp": datetime.now(timezone.utc).isoformat()}
                await ws_manager.broadcast(json.dumps(payload))
                await asyncio.sleep(2)   # Stagger broadcasts
        except Exception as e:
            log.warning(f"Signal broadcast error: {e}")


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown tasks for the FastAPI application."""
    log.info("ForexMind Web Server starting...")
    await init_db()

    # Restore win/loss counters from DB so stats survive restarts
    from forexmind.risk.manager import get_risk_manager
    await get_risk_manager().load_stats_from_db()

    # Start periodic signal broadcast in background
    broadcast_task = asyncio.create_task(_signal_broadcast_loop())
    log.info("Signal broadcast loop started")

    yield  # Application runs here

    broadcast_task.cancel()
    await close_db()
    log.info("ForexMind Web Server shut down")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ForexMind AI Trading Agent",
    description="Real-time AI forex signal generator with Claude-powered chat",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/api/sessions")
async def sessions():
    status = get_session_status()
    from forexmind.utils.session_times import best_pairs_for_session
    return {
        "active_sessions": status.active_sessions,
        "active_overlaps": status.active_overlaps,
        "is_overlap": status.is_overlap,
        "is_weekend": status.is_weekend,
        "session_score": status.session_score,
        "recommended_pairs": best_pairs_for_session()[:8],
    }


@app.get("/api/stats")
async def get_stats():
    """Get all-time trade performance stats from the database."""
    from forexmind.data.trade_repo import get_stats as db_get_stats
    return await db_get_stats()


def _low_liquidity_response(score: float) -> dict:
    return {
        "low_liquidity": True,
        "session_score": score,
        "message": (
            f"Liquidity {score:.0%} — signals unavailable below 40%. "
            "Wait for an active session (Tokyo 00:00–09:00 UTC, "
            "London 07:00–16:00 UTC, New York 12:00–21:00 UTC)."
        ),
    }


@app.get("/api/signal/{instrument}")
async def get_signal(instrument: str, timeframe: str = "M5"):
    """Get trading signal for a specific instrument."""
    session = get_session_status()
    if session.session_score < 0.4:
        return JSONResponse(status_code=200, content=_low_liquidity_response(session.session_score))
    instrument = instrument.upper().replace("/", "_").replace("-", "_")
    try:
        from forexmind.agents.tools import _get_signal
        result = await _get_signal(instrument, timeframe)
        return JSONResponse(content=json.loads(result))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/signals")
async def get_all_signals(timeframe: str = "H1", refresh: bool = False):
    """Get signals for all recommended pairs — H1 primary + M15 entry confirmation."""
    global _signals_cache, _signals_cache_time

    session = get_session_status()
    if session.session_score < 0.4:
        return JSONResponse(status_code=200, content=_low_liquidity_response(session.session_score))

    # Serve from cache if fresh and not explicitly bypassed
    if not refresh and _signals_cache and (time.monotonic() - _signals_cache_time) < SIGNALS_CACHE_TTL:
        return _signals_cache

    from forexmind.utils.session_times import best_pairs_for_session
    from forexmind.agents.tools import _get_signal

    pairs = best_pairs_for_session()[:6]
    # Fetch H1 (trend) and M15 (entry) in parallel for all pairs
    h1_tasks = [_get_signal(pair, "H1", 300) for pair in pairs]
    m15_tasks = [_get_signal(pair, "M15", 200) for pair in pairs]
    h1_results, m15_results = await asyncio.gather(
        asyncio.gather(*h1_tasks, return_exceptions=True),
        asyncio.gather(*m15_tasks, return_exceptions=True),
    )

    # Merge M15 alignment info into H1 signal
    results = []
    for h1_res, m15_res in zip(h1_results, m15_results):
        if isinstance(h1_res, Exception):
            results.append(h1_res)
            continue
        try:
            h1_data = json.loads(h1_res)
            m15_data = json.loads(m15_res) if isinstance(m15_res, str) else {}
            h1_action = h1_data.get("signal", {}).get("action", "HOLD")
            m15_action = m15_data.get("signal", {}).get("action", "HOLD")
            h1_data["m15_aligned"] = (h1_action == m15_action and h1_action != "HOLD")
            results.append(json.dumps(h1_data))
        except Exception:
            results.append(h1_res)

    signals = []
    for pair, result in zip(pairs, results):
        if isinstance(result, str):
            try:
                signals.append(json.loads(result))
            except Exception:
                signals.append({"instrument": pair, "error": "parse error"})
        else:
            signals.append({"instrument": pair, "error": str(result)})

    payload = {"signals": signals, "count": len(signals), "cached_at": datetime.now(timezone.utc).isoformat()}
    _signals_cache = payload
    _signals_cache_time = time.monotonic()
    return payload


@app.get("/api/account")
async def get_account():
    try:
        from forexmind.agents.tools import _get_account
        result = await _get_account()
        return JSONResponse(content=json.loads(result))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trades")
async def get_open_trades():
    """Get all currently open trades from OANDA."""
    try:
        from forexmind.data.oanda_client import get_oanda_client
        client = get_oanda_client()
        trades = await client.get_open_trades()
        return {"trades": trades, "count": len(trades)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trades/{trade_id}/close")
async def close_trade_api(trade_id: str):
    """Close an open trade by OANDA trade ID."""
    try:
        from forexmind.data.oanda_client import get_oanda_client
        client = get_oanda_client()
        result = await client.close_trade(trade_id)
        return {
            "success": True,
            "trade_id": trade_id,
            "realized_pnl": result.realized_pl if hasattr(result, "realized_pl") else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/news/{instrument}")
async def get_news(instrument: str, lookback_hours: int = 4):
    instrument = instrument.upper().replace("/", "_")
    try:
        from forexmind.agents.tools import _get_news
        result = await _get_news(instrument, lookback_hours)
        return JSONResponse(content=json.loads(result))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """Chat with the Claude trading agent (non-streaming fallback)."""
    try:
        from forexmind.agents.claude_agent import get_agent
        agent = get_agent()
        response = await agent.chat(request.message)
        signal = agent.extract_signal(response)
        return ChatResponse(
            response=response,
            signal=signal,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream the Claude agent response token-by-token via Server-Sent Events."""
    async def event_generator():
        try:
            from forexmind.agents.claude_agent import get_agent
            agent = get_agent()
            async for chunk in agent.stream_chat(request.message):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/trade")
async def place_trade_api(request: TradeRequest):
    """Place a trade from the web dashboard using the risk manager for position sizing."""
    session = get_session_status()
    if session.session_score < 0.4:
        return JSONResponse(status_code=400, content=_low_liquidity_response(session.session_score))

    instrument = request.instrument.upper().replace("/", "_").replace("-", "_")
    direction = request.direction.upper()
    if direction not in ("BUY", "SELL"):
        raise HTTPException(status_code=400, detail="direction must be BUY or SELL")
    try:
        from forexmind.agents.tools import _place_trade
        from forexmind.data.oanda_client import get_oanda_client
        from forexmind.risk.manager import get_risk_manager

        client = get_oanda_client()
        acc = await client.get_account()
        rm = get_risk_manager()
        rm.update_peak(acc.balance)

        proposal = rm.calculate_risk(
            instrument=instrument,
            direction=direction,
            entry=request.entry,
            atr=request.atr,
            account_balance=acc.balance,
            confidence=request.confidence,
        )
        if not proposal.approved:
            return JSONResponse(status_code=400, content={"error": proposal.rejection_reason})

        result_str = await _place_trade(
            instrument=instrument,
            direction=direction,
            units=proposal.units,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
        )
        result = json.loads(result_str)
        log.info(f"_place_trade raw result: {result}")
        if "error" in result:
            return JSONResponse(status_code=400, content=result)

        trade_id = result.get("trade_id") or "—"
        # Use filled_price from result; fall back to request entry if 0 or missing
        filled_raw = result.get("filled_price", 0)
        filled = float(filled_raw) if filled_raw else request.entry
        filled_fmt = f"{filled:.5f}" if filled else str(request.entry)

        log.info(f"Web trade placed: {direction} {instrument} id={trade_id} filled={filled_fmt} units={proposal.units}")

        # Notify Telegram so web-placed trades appear alongside auto-trades
        try:
            from forexmind.config.settings import get_settings
            from telegram import Bot
            from telegram.constants import ParseMode
            cfg = get_settings()
            if cfg.telegram.is_configured:
                emoji = "🟢" if direction == "BUY" else "🔴"
                msg = (
                    f"{emoji} <b>WEB TRADE: {direction} {instrument.replace('_','/')}</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"Trade ID:    <code>{trade_id}</code>\n"
                    f"Filled @     <code>{filled_fmt}</code>\n"
                    f"Stop Loss:   <code>{request.stop_loss:.5f}</code>\n"
                    f"Take Profit: <code>{request.take_profit:.5f}</code>\n"
                    f"Units:       {proposal.units:,}\n"
                    f"Risk:        <b>{proposal.risk_pct:.1f}%</b>\n"
                    f"R:R:         2.0:1\n"
                    f"\n<i>Placed via web dashboard</i>"
                )
                bot = Bot(token=cfg.telegram.bot_token)
                await bot.send_message(chat_id=cfg.telegram.chat_id, text=msg, parse_mode=ParseMode.HTML)
        except Exception as tg_err:
            log.warning(f"Telegram notify failed for web trade: {tg_err}")

        return {
            "success": True,
            "trade_id": trade_id,
            "filled_price": filled,
            "units": proposal.units,
            "risk_pct": proposal.risk_pct,
            "instrument": instrument,
            "direction": direction,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/signals")
async def websocket_signals(websocket: WebSocket):
    """
    WebSocket endpoint for real-time signal streaming.
    Clients connect here to receive live signal updates.
    """
    await ws_manager.connect(websocket)
    # Send initial session status immediately on connect
    session = get_session_status()
    init_payload: dict = {
        "type": "session",
        "data": {
            "active_sessions": session.active_sessions,
            "is_overlap": session.is_overlap,
            "session_score": session.session_score,
            "low_liquidity": session.session_score < 0.4,
        },
    }
    if session.session_score < 0.4:
        init_payload["data"]["message"] = (
            f"Liquidity {session.session_score:.0%} — signals paused. "
            "Returns when an active session opens."
        )
    await websocket.send_text(json.dumps(init_payload))
    try:
        while True:
            # Keep connection alive by reading (client can send pings)
            text = await websocket.receive_text()
            if text == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the trading dashboard HTML page."""
    from pathlib import Path
    static_path = Path(__file__).parent / "static" / "index.html"
    if static_path.exists():
        return HTMLResponse(content=static_path.read_text())
    return HTMLResponse(content="<h1>ForexMind</h1><p>Dashboard not found. Check static/index.html</p>")
