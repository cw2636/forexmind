"""
ForexMind — Main Entry Point
================================
Unified CLI that launches any part of the ForexMind system.

Usage:
    python main.py cli          — Start terminal chat interface
    python main.py web          — Start FastAPI web dashboard (port 8000)
    python main.py telegram     — Start Telegram bot
    python main.py all          — Start web + telegram (CLI separately)
    python main.py signal EUR_USD   — Quick one-shot signal without agent chat
    python main.py backtest EUR_USD — Quick backtest
    python main.py train EUR_USD    — Train ML models on recent data

Advanced Python:
  - asyncio.gather for parallel task execution
  - argparse for CLI argument parsing
  - uvicorn programmatic startup
  - Graceful shutdown with signal handlers
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Add project to path so imports work when running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forexmind.utils.logger import get_logger
from forexmind.config.settings import get_settings

log = get_logger(__name__)


# ── Startup validation ────────────────────────────────────────────────────────

def check_config() -> None:
    """Print configuration warnings at startup."""
    from rich.console import Console
    from rich.panel import Panel
    console = Console()

    cfg = get_settings()
    warnings = cfg.validate()

    if warnings:
        console.print(Panel(
            "\n".join(f"[yellow]⚠  {w}[/yellow]" for w in warnings),
            title="[bold yellow]Configuration Warnings[/bold yellow]",
            border_style="yellow",
        ))
    else:
        console.print("[green]✓ Configuration looks complete[/green]")

    if cfg.app.paper_trading:
        console.print("[blue]ℹ  Paper trading mode is ACTIVE — no real money at risk[/blue]")
    else:
        console.print("[bold red]⚠  LIVE TRADING MODE — real money is at risk![/bold red]")


# ── CLI mode ──────────────────────────────────────────────────────────────────

async def run_cli_mode() -> None:
    from forexmind.interfaces.cli import run_cli
    from forexmind.data.database import init_db
    await init_db()
    await run_cli()


# ── Web mode ──────────────────────────────────────────────────────────────────

async def run_web_mode() -> None:
    import uvicorn
    cfg = get_settings()
    log.info(f"Starting web dashboard on http://0.0.0.0:{cfg.app.web_port}")
    config = uvicorn.Config(
        "forexmind.interfaces.web.app:app",
        host="0.0.0.0",
        port=cfg.app.web_port,
        reload=False,
        log_level=cfg.app.log_level.lower(),
    )
    server = uvicorn.Server(config)
    await server.serve()


# ── Telegram mode ─────────────────────────────────────────────────────────────

async def run_telegram_mode() -> None:
    from forexmind.interfaces.telegram_bot import run_telegram_bot
    await run_telegram_bot()


# ── All mode (web + telegram) ─────────────────────────────────────────────────

async def run_all_mode() -> None:
    """Run web server and Telegram bot concurrently."""
    tasks = [
        asyncio.create_task(run_web_mode()),
        asyncio.create_task(run_telegram_mode()),
    ]
    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        log.info("Shutting down...")
        for task in tasks:
            task.cancel()


# ── Quick signal ──────────────────────────────────────────────────────────────

async def quick_signal(pair: str) -> None:
    """One-shot signal without starting the full agent."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.json import JSON

    console = Console()
    pair = pair.upper().replace("/", "_")
    console.print(f"[blue]Analysing {pair}...[/blue]")

    try:
        from forexmind.agents.tools import _get_signal
        result = await _get_signal(pair, "M5", 300)
        console.print(Panel(JSON(result), title=f"Signal: {pair}", border_style="blue"))
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[yellow]Tip: Make sure OANDA_API_KEY and OANDA_ACCOUNT_ID are set in .env[/yellow]")


# ── Quick backtest ────────────────────────────────────────────────────────────

async def quick_backtest(pair: str) -> None:
    """Run a quick 1-year backtest and print results."""
    from rich.console import Console
    from rich.json import JSON
    from rich.panel import Panel

    console = Console()
    pair = pair.upper().replace("/", "_")
    console.print(f"[blue]Running backtest on {pair}...[/blue]")

    try:
        from forexmind.agents.tools import _run_backtest
        result = await _run_backtest(pair, "M5", "2024-01-01", "2024-12-31")
        console.print(Panel(JSON(result), title=f"Backtest: {pair}", border_style="cyan"))
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


# ── Training ──────────────────────────────────────────────────────────────────

async def train_models(pair: str) -> None:
    """
    Fetch 2 years of historical data and train LightGBM + LSTM models.
    This will take several minutes depending on data size and hardware.
    """
    from rich.console import Console
    from rich.progress import Progress

    console = Console()
    pair = pair.upper().replace("/", "_")
    console.print(f"[blue]Training ML models on {pair} history...[/blue]")

    try:
        from forexmind.data.oanda_client import get_oanda_client
        from forexmind.indicators.engine import get_indicator_engine
        from forexmind.strategy.ml_strategy import LightGBMStrategy, LSTMStrategy
        from datetime import datetime, timezone

        client = get_oanda_client()
        console.print("[dim]Fetching 2 years of M5 data (this may take a moment)...[/dim]")

        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)
        df = await client.get_candles(pair, "M5", from_dt=start, to_dt=end)

        if df.empty:
            console.print("[red]No data returned. Check OANDA API key.[/red]")
            return

        engine = get_indicator_engine()
        df_ind = engine.compute(df)
        console.print(f"[green]Got {len(df_ind)} bars. Starting training...[/green]")

        # Train LightGBM
        console.print("[blue]Training LightGBM...[/blue]")
        lgbm = LightGBMStrategy()
        metrics = lgbm.train(df_ind)
        console.print(f"[green]LightGBM done. Accuracy: {metrics.get('accuracy', 0):.4f}[/green]")

        # Train LSTM
        console.print("[blue]Training LSTM (may take a few minutes)...[/blue]")
        lstm = LSTMStrategy()
        lstm_metrics = lstm.train(df_ind, epochs=30)
        console.print(f"[green]LSTM done. Accuracy: {lstm_metrics.get('accuracy', 0):.4f}[/green]")

        console.print("[bold green]✓ Models trained and saved![/bold green]")

    except Exception as e:
        console.print(f"[red]Training error: {e}[/red]")
        import traceback
        traceback.print_exc()


# ── Argument parser ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ForexMind AI Trading Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py cli                    Start terminal chat interface
  python main.py web                    Start web dashboard (http://localhost:8000)
  python main.py telegram               Start Telegram bot
  python main.py all                    Start web + telegram simultaneously
  python main.py signal EUR/USD         Quick one-shot signal
  python main.py backtest GBP/USD       Run 1-year backtest
  python main.py train EUR_USD          Train ML models
        """
    )
    parser.add_argument(
        "mode",
        choices=["cli", "web", "telegram", "all", "signal", "backtest", "train"],
        help="Which interface to run",
    )
    parser.add_argument(
        "pair",
        nargs="?",
        default="EUR_USD",
        help="Forex pair for signal/backtest/train modes",
    )
    args = parser.parse_args()

    check_config()

    mode_map = {
        "cli": lambda: asyncio.run(run_cli_mode()),
        "web": lambda: asyncio.run(run_web_mode()),
        "telegram": lambda: asyncio.run(run_telegram_mode()),
        "all": lambda: asyncio.run(run_all_mode()),
        "signal": lambda: asyncio.run(quick_signal(args.pair)),
        "backtest": lambda: asyncio.run(quick_backtest(args.pair)),
        "train": lambda: asyncio.run(train_models(args.pair)),
    }

    try:
        mode_map[args.mode]()
    except KeyboardInterrupt:
        log.info("ForexMind stopped by user")


if __name__ == "__main__":
    main()
