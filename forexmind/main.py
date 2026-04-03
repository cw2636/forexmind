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


# ── Scheduler mode ────────────────────────────────────────────────────────────

async def run_scheduler_mode() -> None:
    from forexmind.scheduler import run_scheduler
    from forexmind.data.database import init_db
    await init_db()
    await run_scheduler()


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
        from forexmind.strategy.rl_strategy import RLStrategy
        from datetime import datetime, timezone

        import logging as _logging
        _logging.getLogger("oandapyV20").setLevel(_logging.WARNING)

        client = get_oanda_client()

        # Strategy: train LightGBM + PPO on M5 (high-frequency, large dataset),
        # train LSTM on H1 (hourly bars) — far cleaner signal, fewer bars needed.
        # H1 with forward_bars=12 = 12 hours ahead (clear directional moves).
        import pandas as pd
        from datetime import timedelta

        TRAIN_PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY"]
        start = datetime(2018, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 12, 31, tzinfo=timezone.utc)

        console.print("[dim]Fetching M5 data (LightGBM + PPO) across 3 pairs from 2018...[/dim]")
        m5_window = timedelta(weeks=2)

        # Fetch each pair and compute FULL feature matrices per-pair (with DatetimeIndex intact),
        # then concatenate. This avoids timestamp deduplication and the 'int has no to_pydatetime' crash.
        from forexmind.strategy.feature_engineering import build_feature_matrix
        engine = get_indicator_engine()
        pair_feature_dfs: list[pd.DataFrame] = []  # M5 per-pair, for LightGBM + PPO

        async def fetch_pair(pair: str, granularity: str, window: timedelta) -> pd.DataFrame:
            """Fetch and deduplicate one pair's raw OHLCV data."""
            cursor = start
            chunks: list[pd.DataFrame] = []
            err_count = 0
            while cursor < end:
                chunk_end = min(cursor + window, end)
                try:
                    chunk = await client.get_candles(pair, granularity, from_dt=cursor, to_dt=chunk_end)
                    if not chunk.empty:
                        chunks.append(chunk)
                except Exception as e:
                    err_count += 1
                    if err_count <= 3:
                        console.print(f"[yellow]  {pair}/{granularity} chunk {cursor.date()}: {e}[/yellow]")
                cursor = chunk_end
            if not chunks:
                return pd.DataFrame()
            raw = pd.concat(chunks).sort_index()
            return raw[~raw.index.duplicated(keep="last")]

        # ── M5 fetch for LightGBM / PPO ───────────────────────────────────────
        for train_pair in TRAIN_PAIRS:
            console.print(f"[dim]  M5 {train_pair}...[/dim]")
            raw = await fetch_pair(train_pair, "M5", m5_window)
            if raw.empty:
                console.print(f"[yellow]  {train_pair}: no M5 data, skipping.[/yellow]")
                continue
            pair_features = build_feature_matrix(engine.compute(raw), add_target=True)
            console.print(f"[dim]    → {len(pair_features):,} M5 feature rows for {train_pair}[/dim]")
            pair_feature_dfs.append(pair_features)

        if not pair_feature_dfs:
            console.print("[red]No M5 data returned. Check OANDA API key.[/red]")
            return

        df_ind = pd.concat(pair_feature_dfs, ignore_index=True)
        console.print(f"[green]M5 combined: {len(df_ind):,} rows across {len(pair_feature_dfs)} pairs.[/green]")

        # ── H1 fetch for LSTM ─────────────────────────────────────────────────
        # H1 bars: 1 bar = 1 hour. forward_bars=12 → predict 12h ahead.
        # ~7 years × 8760 H1 bars/year ≈ 61,000 bars per pair — fast to train.
        # Signal-to-noise is far better than M5 for directional LSTM.
        console.print("[dim]Fetching H1 data for LSTM (12h lookahead)...[/dim]")
        h1_window = timedelta(weeks=26)   # 26-week chunks → fewer API calls on H1
        h1_feature_dfs: list[pd.DataFrame] = []
        for train_pair in TRAIN_PAIRS:
            console.print(f"[dim]  H1 {train_pair}...[/dim]")
            raw_h1 = await fetch_pair(train_pair, "H1", h1_window)
            if raw_h1.empty:
                console.print(f"[yellow]  {train_pair}: no H1 data, skipping.[/yellow]")
                continue
            pair_h1 = build_feature_matrix(engine.compute(raw_h1), add_target=True)
            console.print(f"[dim]    → {len(pair_h1):,} H1 feature rows for {train_pair}[/dim]")
            h1_feature_dfs.append(pair_h1)
        console.print(f"[green]H1 combined: {sum(len(d) for d in h1_feature_dfs):,} rows across {len(h1_feature_dfs)} pairs.[/green]")

        # Train LightGBM on M5 rows (skip if model < 24h old)
        import time as _time
        lgbm_path = get_settings().app.models_dir / "lgbm_forex.pkl"
        lgbm_age_h = (_time.time() - lgbm_path.stat().st_mtime) / 3600 if lgbm_path.exists() else 999
        if lgbm_age_h < 24:
            console.print(f"[yellow]LightGBM skipped — model is only {lgbm_age_h:.1f}h old (< 24h threshold).[/yellow]")
        else:
            console.print("[blue]Training LightGBM on M5 dataset...[/blue]")
            lgbm = LightGBMStrategy()
            metrics = lgbm.train(df_ind)
            console.print(f"[green]LightGBM done. Accuracy: {metrics.get('accuracy', 0):.4f}[/green]")

        # Train LSTM per-pair on H1 data (12h lookahead, binary UP/DOWN classification)
        use_lstm_data = h1_feature_dfs if h1_feature_dfs else pair_feature_dfs
        lstm_label = "H1" if h1_feature_dfs else "M5"
        console.print(f"[blue]Training LSTM on {lstm_label} per-pair (target 70% accuracy)...[/blue]")
        lstm = LSTMStrategy()
        lstm_metrics: dict = {}
        for pair_idx, pair_feat_df in enumerate(use_lstm_data):
            pair_name = TRAIN_PAIRS[pair_idx]
            console.print(f"[dim]  LSTM training on {pair_name} ({len(pair_feat_df):,} rows)...[/dim]")
            warm = pair_idx > 0  # fine-tune from prev pair after first
            pair_metrics = lstm.train(
                pair_feat_df,
                target_accuracy=0.70,
                max_rows=500_000,   # H1 has ~61k rows/pair — no real cap needed
                warm_start=warm,
            )
            console.print(f"[dim]    {pair_name} → accuracy: {pair_metrics.get('accuracy', 0):.4f}[/dim]")
            if pair_metrics.get("accuracy", 0) > lstm_metrics.get("accuracy", 0):
                lstm_metrics = pair_metrics
        console.print(f"[green]LSTM done. Best accuracy: {lstm_metrics.get('accuracy', 0):.4f} | Params: {lstm_metrics.get('best_params', {})}[/green]")

        # Train PPO RL Agent on EUR_USD (single coherent time-series with DatetimeIndex)
        console.print("[blue]Training PPO RL Agent (~200k timesteps)...[/blue]")
        try:
            rl = RLStrategy()
            rl_metrics = rl.train(pair_feature_dfs[0], instrument=TRAIN_PAIRS[0], total_timesteps=200_000)
            console.print(f"[green]PPO RL done. Timesteps: {rl_metrics.get('total_timesteps', 0):,}[/green]")
        except ImportError as e:
            console.print(f"[yellow]PPO RL skipped: {e}[/yellow]")

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
  python main.py scheduler              Auto-scan US session, push Telegram alerts
  python main.py signal EUR/USD         Quick one-shot signal
  python main.py backtest GBP/USD       Run 1-year backtest
  python main.py train EUR_USD          Train ML models
        """
    )
    parser.add_argument(
        "mode",
        choices=["cli", "web", "telegram", "all", "scheduler", "signal", "backtest", "train"],
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
        "scheduler": lambda: asyncio.run(run_scheduler_mode()),
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
