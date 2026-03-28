"""
ForexMind — Rich CLI Interface
================================
A beautiful terminal chat interface using the `rich` library.

Features:
  - Conversational chat loop with the Claude agent
  - Live signal dashboard panel showing all monitored pairs
  - Colour-coded signal display (green=BUY, red=SELL, grey=HOLD)
  - Auto-refresh live signals every 60 seconds in background
  - /commands for shortcuts

Commands:
  /signal EUR_USD       — Get quick signal without a chat message
  /pairs                — List all monitored pairs
  /account              — Show account snapshot
  /sessions             — Current market sessions
  /backtest EUR_USD 5m  — Quick backtest
  /clear                — Clear conversation history
  /quit                 — Exit

Advanced Python:
  - asyncio.run() with background tasks
  - rich.live.Live for non-blocking dashboard updates
  - rich.layout.Layout for multi-panel display
  - signal handler for clean Ctrl+C exit
"""

from __future__ import annotations

import asyncio
import signal
import sys
from datetime import datetime, timezone

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from rich import box

from forexmind.utils.logger import get_logger
from forexmind.utils.session_times import get_session_status

log = get_logger(__name__)
console = Console()

BANNER = """
[bold cyan]╔═══════════════════════════════════════════════════╗[/]
[bold cyan]║      ForexMind AI Trading Agent  v1.0             ║[/]
[bold cyan]║   Type [white]/help[/white] for commands  ·  [white]/quit[/white] to exit      ║[/]
[bold cyan]╚═══════════════════════════════════════════════════╝[/]
"""

HELP_TEXT = """
[bold yellow]Available Commands:[/bold yellow]

  [cyan]/signal[/cyan] [white]EUR_USD[/white]         Get trading signal for a pair
  [cyan]/pairs[/cyan]                   List all monitored pairs
  [cyan]/account[/cyan]                 Show OANDA account snapshot
  [cyan]/sessions[/cyan]                Current market sessions
  [cyan]/backtest[/cyan] [white]EUR_USD[/white]        Run backtest on a pair
  [cyan]/clear[/cyan]                   Clear conversation history
  [cyan]/quit[/cyan] or [cyan]/exit[/cyan]           Exit ForexMind

  [italic]Or just type naturally — I'm a conversational AI![/italic]
  Examples:
    "Should I buy EUR/USD right now?"
    "What's the EUR/USD outlook for today?"
    "Explain the MACD signal on GBP/USD"
    "Is now a good time to trade?"
"""

QUICK_COMMANDS = {
    "/help": "show_help",
    "/pairs": "show_pairs",
    "/account": "show_account",
    "/sessions": "show_sessions",
    "/clear": "clear_memory",
    "/quit": "quit",
    "/exit": "quit",
}


async def run_cli() -> None:
    """Main entry point for the CLI interface."""
    console.print(BANNER)

    # Initialise agent
    try:
        from forexmind.agents.claude_agent import get_agent
        agent = get_agent()
        console.print(f"[green]✓[/green] Claude agent ready (tools: {', '.join(agent.tool_names)})\n")
    except Exception as e:
        console.print(f"[red]✗ Agent init failed:[/red] {e}")
        console.print("[yellow]Tip: Make sure ANTHROPIC_API_KEY is set in .env[/yellow]")
        return

    # Show initial session status
    await _show_sessions()

    while True:
        try:
            user_input = Prompt.ask(
                "\n[bold green]You[/bold green]",
                console=console,
            ).strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Goodbye! Good trading! 📈[/yellow]")
            break

        if not user_input:
            continue

        # ── Quick Commands ──────────────────────────────────────────────────
        lower = user_input.lower()

        if lower in ("/quit", "/exit"):
            console.print("[yellow]Goodbye! Good trading! 📈[/yellow]")
            break

        elif lower == "/help":
            console.print(HELP_TEXT)
            continue

        elif lower == "/clear":
            agent.clear_memory()
            console.print("[green]Conversation history cleared.[/green]")
            continue

        elif lower == "/sessions":
            await _show_sessions()
            continue

        elif lower == "/pairs":
            from forexmind.config.settings import get_settings
            pairs = get_settings().pairs
            t = Table(title="Monitored Pairs", box=box.SIMPLE)
            t.add_column("Pair", style="cyan")
            for p in pairs:
                t.add_row(p.replace("_", "/"))
            console.print(t)
            continue

        elif lower == "/account":
            console.print("[dim]Fetching account info...[/dim]")
            try:
                from forexmind.agents.tools import _get_account
                result = await _get_account()
                import json
                data = json.loads(result)
                if "error" in data:
                    console.print(f"[red]{data['error']}[/red]")
                else:
                    _print_account_panel(data)
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
            continue

        elif lower.startswith("/signal "):
            pair = lower.split(" ", 1)[1].strip().upper().replace("/", "_")
            console.print(f"[dim]Analysing {pair}...[/dim]")
            response = await agent.chat(f"Give me a full trading signal for {pair}")
            _print_agent_response(response, agent)
            continue

        elif lower.startswith("/backtest "):
            args = user_input.split()[1:]
            pair = args[0].upper().replace("/", "_") if args else "EUR_USD"
            response = await agent.chat(f"Run a backtest on {pair} for the last 12 months")
            _print_agent_response(response, agent)
            continue

        # ── Natural language chat ───────────────────────────────────────────
        console.print()
        response_panel = Panel(
            "[dim]ForexMind is thinking...[/dim]",
            title="[bold blue]ForexMind[/bold blue]",
            border_style="blue",
        )

        # Stream response token-by-token for responsive feel
        full_response = ""
        with console.status("[bold blue]ForexMind is analysing...[/bold blue]"):
            async for chunk in agent.stream_chat(user_input):
                full_response += chunk

        _print_agent_response(full_response, agent)


def _print_agent_response(response: str, agent: Any = None) -> None:  # type: ignore[type-arg]
    """Print the agent response with signal highlighting."""
    # Check if there's a JSON signal to highlight
    signal_data = None
    if agent and hasattr(agent, "extract_signal"):
        signal_data = agent.extract_signal(response)

    if signal_data:
        _print_signal_panel(signal_data)

    # Print the full response as Markdown
    console.print(
        Panel(
            Markdown(response),
            title="[bold blue]ForexMind[/bold blue]",
            border_style="blue",
        )
    )


def _print_signal_panel(signal: dict) -> None:
    """Print a colour-coded signal summary panel."""
    action = signal.get("action", "HOLD")
    colour = "green" if action == "BUY" else "red" if action == "SELL" else "yellow"

    t = Table(box=box.SIMPLE, show_header=False)
    t.add_column("Key", style="bold")
    t.add_column("Value")

    t.add_row("Action", f"[bold {colour}]{action}[/bold {colour}]")
    t.add_row("Instrument", str(signal.get("instrument", "")))
    t.add_row("Entry", str(signal.get("entry", "")))
    t.add_row("Stop Loss", f"[red]{signal.get('stop_loss', '')}[/red] ({signal.get('stop_loss_pips', '')} pips)")
    t.add_row("Take Profit", f"[green]{signal.get('take_profit', '')}[/green] ({signal.get('take_profit_pips', '')} pips)")
    t.add_row("R:R", str(signal.get("risk_reward", "")))
    t.add_row("Confidence", f"{signal.get('confidence', 0)}%")
    t.add_row("Risk %", f"{signal.get('risk_pct', '')}%")

    console.print(Panel(t, title=f"[bold {colour}]📊 SIGNAL: {action}[/bold {colour}]", border_style=colour))


async def _show_sessions() -> None:
    """Print current forex session status."""
    status = get_session_status()
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")

    if status.is_weekend:
        console.print(Panel("[yellow]🚫 Weekend — Forex markets closed[/yellow]", title=f"Sessions ({now})"))
        return

    t = Table(box=box.SIMPLE, show_header=False)
    t.add_column("Session")
    t.add_column("Status")

    session_info = [
        ("Sydney",   "Sydney" in status.active_sessions),
        ("Tokyo",    "Tokyo" in status.active_sessions),
        ("London",   "London" in status.active_sessions),
        ("New York", "New York" in status.active_sessions),
    ]

    for name, active in session_info:
        t.add_row(name, "[green]● OPEN[/green]" if active else "[dim]○ closed[/dim]")

    overlap_text = ""
    if status.active_overlaps:
        overlap_text = f"\n[bold green]🔥 OVERLAP: {', '.join(status.active_overlaps)} — Prime scalping time![/bold green]"

    console.print(Panel(
        t.__rich_console__(console, console.options),  # type: ignore[arg-type]
        title=f"[bold]Market Sessions — {now}[/bold]",
        subtitle=overlap_text,
    ))


def _print_account_panel(data: dict) -> None:
    t = Table(box=box.SIMPLE, show_header=False)
    t.add_column("Key", style="bold")
    t.add_column("Value")
    t.add_row("Balance", f"${data.get('balance', 0):,.2f}")
    t.add_row("NAV", f"${data.get('nav', 0):,.2f}")
    t.add_row("Unrealised P&L", f"${data.get('unrealized_pnl', 0):,.2f}")
    t.add_row("Open Trades", str(data.get("open_trade_count", 0)))
    t.add_row("Daily P&L", f"${data.get('daily_pnl_usd', 0):,.2f}")
    t.add_row("Margin Available", f"${data.get('margin_available', 0):,.2f}")
    colour = "green" if float(data.get("unrealized_pnl", 0)) >= 0 else "red"
    console.print(Panel(t, title="[bold]OANDA Account[/bold]", border_style=colour))
