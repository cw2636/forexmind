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
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
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
        if session.is_weekend or session.session_score < 0.2:
            continue

        try:
            from forexmind.agents.tools import _get_signal
            # Broadcast signals for top 3 recommended pairs
            from forexmind.utils.session_times import best_pairs_for_session
            pairs = best_pairs_for_session()[:3]
            for pair in pairs:
                result = await _get_signal(pair, "M5", 200)
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


@app.get("/api/signal/{instrument}")
async def get_signal(instrument: str, timeframe: str = "M5"):
    """Get trading signal for a specific instrument."""
    instrument = instrument.upper().replace("/", "_").replace("-", "_")
    try:
        from forexmind.agents.tools import _get_signal
        result = await _get_signal(instrument, timeframe)
        return JSONResponse(content=json.loads(result))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/signals")
async def get_all_signals(timeframe: str = "M5"):
    """Get signals for all recommended pairs in the current session."""
    from forexmind.utils.session_times import best_pairs_for_session
    from forexmind.agents.tools import _get_signal

    pairs = best_pairs_for_session()[:6]
    tasks = [_get_signal(pair, timeframe) for pair in pairs]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    signals = []
    for pair, result in zip(pairs, results):
        if isinstance(result, str):
            try:
                signals.append(json.loads(result))
            except Exception:
                signals.append({"instrument": pair, "error": "parse error"})
        else:
            signals.append({"instrument": pair, "error": str(result)})

    return {"signals": signals, "count": len(signals)}


@app.get("/api/account")
async def get_account():
    try:
        from forexmind.agents.tools import _get_account
        result = await _get_account()
        return JSONResponse(content=json.loads(result))
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
    """Chat with the Claude trading agent."""
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


@app.websocket("/ws/signals")
async def websocket_signals(websocket: WebSocket):
    """
    WebSocket endpoint for real-time signal streaming.
    Clients connect here to receive live signal updates.
    """
    await ws_manager.connect(websocket)
    # Send initial session status immediately on connect
    session = get_session_status()
    await websocket.send_text(json.dumps({
        "type": "session",
        "data": {
            "active_sessions": session.active_sessions,
            "is_overlap": session.is_overlap,
        }
    }))
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
