"""
ForexMind — Configuration Manager
===================================
Loads settings from .env + config.yaml and exposes a singleton Settings object.

Advanced Python concepts demonstrated here:
  - dataclasses with field() and __post_init__
  - @property for derived attributes
  - @lru_cache for singleton pattern
  - Type hints with generics (list[str], dict[str, Any])
  - Pathlib for OS-agnostic paths
  - from __future__ import annotations (postponed evaluation)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Project root: forexmind/config/settings.py → forexmind/ → /home/wilson/Forex/
ROOT_DIR = Path(__file__).resolve().parent.parent        # .../forexmind/
PROJECT_DIR = ROOT_DIR.parent                            # .../Forex/

# System paths used when installed as a .deb package
SYSTEM_CONFIG = Path("/etc/forexmind")
SYSTEM_DATA = Path("/var/lib/forexmind")

# Prefer /etc/forexmind/.env (installed package) over project-root .env (dev)
# Check for the actual .env file, not just the directory — avoids false positive when
# /etc/forexmind/ exists but hasn't been configured yet (e.g. fresh .deb install)
_system_env = SYSTEM_CONFIG / ".env"
ENV_FILE = _system_env if _system_env.is_file() else PROJECT_DIR / ".env"

# Load .env at module import time — safe to call multiple times
load_dotenv(ENV_FILE, override=False)


# ── Sub-config dataclasses ────────────────────────────────────────────────────

@dataclass
class OandaConfig:
    """OANDA broker/data API settings."""
    api_key: str = field(default_factory=lambda: os.environ.get("OANDA_API_KEY", ""))
    account_id: str = field(default_factory=lambda: os.environ.get("OANDA_ACCOUNT_ID", ""))
    environment: str = field(default_factory=lambda: os.environ.get("OANDA_ENVIRONMENT", "practice"))

    @property
    def rest_url(self) -> str:
        return (
            "https://api-fxtrade.oanda.com"
            if self.environment == "live"
            else "https://api-fxpractice.oanda.com"
        )

    @property
    def stream_url(self) -> str:
        return (
            "https://stream-fxtrade.oanda.com"
            if self.environment == "live"
            else "https://stream-fxpractice.oanda.com"
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.account_id)


@dataclass
class ClaudeConfig:
    """Anthropic Claude LLM settings."""
    api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    model: str = field(
        default_factory=lambda: os.environ.get("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")
    )
    temperature: float = 0.1    # Low temperature = deterministic, disciplined trading decisions
    max_tokens: int = 4096

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)


@dataclass
class DataConfig:
    """External data source API settings."""
    alpha_vantage_key: str = field(
        default_factory=lambda: os.environ.get("ALPHA_VANTAGE_API_KEY", "")
    )
    finnhub_key: str = field(
        default_factory=lambda: os.environ.get("FINNHUB_API_KEY", "")
    )
    database_url: str = field(
        default_factory=lambda: os.environ.get(
            "DATABASE_URL",
            f"sqlite+aiosqlite:///{SYSTEM_DATA}/data/forexmind.db"
            if SYSTEM_DATA.exists()
            else f"sqlite+aiosqlite:///{ROOT_DIR}/data/forexmind.db",
        )
    )
    # How long (seconds) to use cached price data before re-fetching
    cache_ttl_seconds: int = 60


@dataclass
class TelegramConfig:
    """Telegram bot settings."""
    bot_token: str = field(default_factory=lambda: os.environ.get("TELEGRAM_BOT_TOKEN", ""))
    chat_id: str = field(default_factory=lambda: os.environ.get("TELEGRAM_CHAT_ID", ""))

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)


@dataclass
class RiskConfig:
    """
    Risk parameters tuned for aggressive compounding on a small account.

    Confidence-tiered sizing (set in risk/manager.py):
      < 0.55  → skip trade entirely
      0.55–0.65 → 2% risk  (marginal signal)
      0.65–0.75 → 3% risk  (moderate conviction)
      0.75–0.85 → 4% risk  (strong conviction)
      > 0.85  → 5% risk  (highest conviction)

    Circuit breakers:
      daily_loss_limit_pct   — reset each day (prevents a bad session compounding)
      max_total_drawdown_pct — measured from peak equity (prevents slow account bleed)
    """
    # Confidence-tier risk bounds (used by tiered sizing logic)
    max_risk_per_trade_pct: float = 5.0       # top tier (conf > 0.85)
    min_risk_per_trade_pct: float = 2.0       # bottom tradeable tier (conf 0.55–0.65)
    default_rr_ratio: float = 2.0
    atr_stop_multiplier: float = 1.5
    trailing_stop_multiplier: float = 1.0
    breakeven_trigger_rr: float = 1.0
    max_concurrent_trades: int = 3            # fewer, higher-quality positions only
    spread_filter_pips: float = 3.0

    # Daily loss limit — hard stop for the day
    daily_loss_limit_pct: float = 5.0         # halt if day loss exceeds 5% of balance

    # Daily profit target — meaningful stretch goal for progress tracking
    daily_profit_target_pct: float = 2.0      # ~$2,040 on $102k — locks in gains on exceptional days

    # Total drawdown halt — measured from peak account equity
    max_total_drawdown_pct: float = 20.0      # halt and alert if account drops 20% from peak

    # Minimum confidence to place any trade at all
    min_signal_confidence: float = 0.55

    # ATR-based dynamic TP (Fix 1)
    # TP = max(ATR * atr_tp_multiplier, SL_distance * min_rr_floor)
    # This decouples TP sizing from the fixed RR ratio — TP now tracks volatility.
    atr_tp_multiplier: float = 1.2   # tune between 0.8 and 2.0
    min_rr_floor: float = 1.0        # never place a trade with RR below 1:1


@dataclass
class AppConfig:
    """Application-level runtime settings."""
    log_level: str = field(default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO"))
    web_port: int = field(
        default_factory=lambda: int(os.environ.get("WEB_PORT", "8000"))
    )
    paper_trading: bool = field(
        default_factory=lambda: os.environ.get("PAPER_TRADING", "True").lower() == "true"
    )
    data_dir: Path = field(
        default_factory=lambda: SYSTEM_DATA / "data" if SYSTEM_DATA.exists() else ROOT_DIR / "data"
    )
    models_dir: Path = field(
        default_factory=lambda: SYSTEM_DATA / "models" if SYSTEM_DATA.exists() else ROOT_DIR / "models"
    )

    def __post_init__(self) -> None:
        # Ensure critical directories exist at startup
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


# ── Master Settings Object ────────────────────────────────────────────────────

@dataclass
class Settings:
    """
    Singleton settings object. Access everywhere via:

        from forexmind.config.settings import get_settings
        cfg = get_settings()
        print(cfg.claude.model)

    The @lru_cache on get_settings() ensures only one instance is created.
    Call get_settings.cache_clear() in unit tests to reset between test cases.
    """
    oanda: OandaConfig = field(default_factory=OandaConfig)
    claude: ClaudeConfig = field(default_factory=ClaudeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    app: AppConfig = field(default_factory=AppConfig)

    # Populated from config.yaml in __post_init__
    pairs: list[str] = field(default_factory=list)
    timeframes_primary: list[str] = field(default_factory=list)
    timeframes_context: list[str] = field(default_factory=list)
    yaml_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.yaml_config = _load_yaml_config()
        timeframes = self.yaml_config.get("timeframes", {})

        if not self.pairs:
            self.pairs = self.yaml_config.get("pairs", ["EUR_USD", "GBP_USD", "USD_JPY"])
        if not self.timeframes_primary:
            self.timeframes_primary = timeframes.get("primary", ["M1", "M5"])
        if not self.timeframes_context:
            self.timeframes_context = timeframes.get("context", ["M15", "H1"])

    @property
    def ensemble_weights(self) -> dict[str, float]:
        return self.yaml_config.get(
            "ensemble_weights",
            {"rule_based": 0.30, "lightgbm": 0.25, "lstm": 0.25, "rl_agent": 0.20},
        )

    @property
    def indicator_config(self) -> dict[str, Any]:
        return self.yaml_config.get("indicators", {})

    @property
    def risk_config_yaml(self) -> dict[str, Any]:
        return self.yaml_config.get("risk", {})

    @property
    def backtest_config(self) -> dict[str, Any]:
        return self.yaml_config.get("backtest", {})

    @property
    def ml_config(self) -> dict[str, Any]:
        return self.yaml_config.get("ml", {})

    @property
    def news_config(self) -> dict[str, Any]:
        return self.yaml_config.get("news", {})

    def validate(self) -> list[str]:
        """
        Check required config is present. Returns list of warning strings.
        Run this at startup so the user knows exactly what's misconfigured.
        """
        warnings: list[str] = []
        if not self.oanda.is_configured:
            warnings.append(
                "OANDA_API_KEY / OANDA_ACCOUNT_ID not set — live price data disabled"
            )
        if not self.claude.is_configured:
            warnings.append("ANTHROPIC_API_KEY not set — AI agent disabled")
        if not self.data.alpha_vantage_key:
            warnings.append("ALPHA_VANTAGE_API_KEY not set — Alpha Vantage news disabled")
        if not self.data.finnhub_key:
            warnings.append("FINNHUB_API_KEY not set — Finnhub news disabled")
        if not self.telegram.is_configured:
            warnings.append("Telegram not configured — bot alerts disabled")
        if self.app.paper_trading is False:
            warnings.append("⚠️  LIVE TRADING IS ENABLED — real money at risk!")
        return warnings


# ── Private helpers ───────────────────────────────────────────────────────────

def _load_yaml_config() -> dict[str, Any]:
    """Load config.yaml — user override (/etc/forexmind), then package default, then dev co-located."""
    candidates = [
        SYSTEM_CONFIG / "config.yaml",              # user override (installed package)
        Path("/usr/share/forexmind/config.yaml"),   # package default (installed, read-only)
        Path(__file__).resolve().parent / "config.yaml",  # development (co-located)
    ]
    for yaml_path in candidates:
        if yaml_path.exists():
            with open(yaml_path, encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
    return {}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns the application-wide singleton Settings instance.
    Thread-safe due to CPython's GIL; lru_cache makes it a singleton.

    Usage in tests:
        get_settings.cache_clear()   # Force re-creation with fresh env
    """
    return Settings()
