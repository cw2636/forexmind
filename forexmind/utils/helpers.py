"""
ForexMind — Trading Utilities
================================
Pure-function helpers used throughout the codebase.
No side effects, no I/O — easy to unit-test.

Advanced Python concepts:
  - functools.lru_cache on pure functions
  - Decimal for precise pip calculations
  - TypeVar and Generic usage
"""

from __future__ import annotations

import math
import os
from decimal import Decimal
from typing import Union

import pandas as pd


# ── Pip / Price Utilities ─────────────────────────────────────────────────────

# JPY pairs have a pip at the 2nd decimal place; all others at the 4th
_JPY_PAIRS = frozenset({"USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "CHF_JPY", "CAD_JPY", "NZD_JPY"})

# Spot metals quoted in USD/oz — 1 pip = $1 (natural unit for ATR-based sizing)
_SPOT_METALS = frozenset({"XAU_USD", "XAG_USD", "XPT_USD", "XPD_USD"})


def pip_size(instrument: str) -> float:
    """
    Return the pip size for an instrument.
    JPY pairs: 0.01 | Spot metals (XAU_USD etc): 1.0 | all others: 0.0001
    """
    if "JPY" in instrument:
        return 0.01
    if instrument in _SPOT_METALS:
        return 1.0  # Gold: $1 per pip per oz; keeps ATR-based SL in readable single-digit pips
    return 0.0001


def pips_to_price(pips: float, instrument: str) -> float:
    """Convert a pip count to a price delta."""
    return pips * pip_size(instrument)


def price_to_pips(price_delta: float, instrument: str) -> float:
    """Convert a price delta to pips."""
    return abs(price_delta) / pip_size(instrument)


def spread_pips(bid: float, ask: float, instrument: str) -> float:
    """Calculate spread in pips."""
    return price_to_pips(ask - bid, instrument)


# ── Position Sizing ───────────────────────────────────────────────────────────

def pip_value_usd(instrument: str, current_price: float = 1.0) -> float:
    """
    USD value of 1 pip movement per 1 unit for a given instrument.

    Pair types:
      USD/XXX (USD_CAD, USD_CHF, USD_JPY): pip is in quote currency → divide by current price
      XXX/USD (EUR_USD, GBP_USD, AUD_USD): pip is already in USD → use pip_size directly
      XXX/YYY (EUR_GBP): requires cross rate — approximate as pip_size (minor error)
    """
    ps = pip_size(instrument)
    parts = instrument.split("_")
    if len(parts) != 2:
        return ps
    base, quote = parts[0], parts[1]
    if base in ("XAU", "XAG", "XPT", "XPD"):
        # Spot metals: already quoted in USD; pip_value = pip_size per unit (1 oz)
        return ps
    if base == "USD":
        # pip is in quote currency (CAD, CHF, JPY) — convert to USD
        price = current_price if current_price > 0 else 1.0
        return ps / price
    else:
        # pip is in USD already (EUR_USD, GBP_USD, AUD_USD) or cross (approx)
        return ps


def units_from_risk(
    account_balance: float,
    risk_pct: float,
    stop_loss_pips: float,
    instrument: str,
    pip_value_per_unit: float | None = None,
    current_price: float = 1.0,
) -> int:
    """
    Calculate position size (units) given a fixed-risk model.

    Formula:
        risk_amount = account_balance * (risk_pct / 100)
        units = risk_amount / (stop_loss_pips * pip_value_usd_per_unit)

    Args:
        account_balance: Account equity in USD
        risk_pct: Percentage of account to risk (e.g. 1.5 for 1.5%)
        stop_loss_pips: Distance to stop loss in pips
        instrument: e.g. "EUR_USD"
        pip_value_per_unit: USD value of 1 pip per 1 unit (auto-calculated if None)
        current_price: Current market price (needed for USD/XXX pairs)

    Returns:
        Rounded position size in OANDA units (nearest 1,000 = 1 micro lot)

    Hard caps (read from PAPER_TRADING env var):
        Paper trading: 100,000 units (1 standard lot, $10/pip) — safe for $100k practice account
        Live trading:   20,000 units (2 mini lots,   $2/pip) — safe for $500 live account at 50:1 leverage

    Example (paper, $100k account, 5% risk, 20-pip stop, EUR_USD):
        risk_amount = $5,000
        units = $5,000 / (20 × 0.0001) = 250,000 → capped at 100,000 (1 standard lot)

    Example (live, $500 account, 5% risk, 20-pip stop, EUR_USD):
        risk_amount = $25
        units = $25 / (20 × 0.0001) = 12,500 → 12,000 units (1.2 mini lots)
    """
    if stop_loss_pips <= 0:
        return 0
    pip_val = pip_value_per_unit if pip_value_per_unit is not None else pip_value_usd(instrument, current_price)
    risk_amount = account_balance * (risk_pct / 100.0)
    raw_units = risk_amount / (stop_loss_pips * pip_val)

    # Spot metals: 1 unit = 1 troy oz. Round to nearest oz, min 1, cap 1,000 oz.
    if instrument in _SPOT_METALS:
        return min(1000, max(1, int(round(raw_units))))

    # Lot size caps — scale with account balance so the cap doesn't suppress edge
    # as the account grows.  Formula: balance × 40 units per dollar.
    #   $500  →  20,000 units  (2 mini lots, $2/pip EUR/USD — safe for $500 live)
    #   $1,500 → 60,000 units  (6 mini lots)
    #   $2,500 → 100,000 units (1 standard lot — matches paper cap)
    # Hard ceiling: 200,000 units regardless of balance (single-trade overexposure guard).
    # Paper trading uses a fixed 100,000 cap (paper account is $100k — no scaling needed).
    is_paper = os.environ.get("PAPER_TRADING", "true").lower() == "true"
    if is_paper:
        MAX_UNITS = 100_000
    else:
        MAX_UNITS = min(200_000, max(20_000, int(account_balance * 40)))

    # Round down to nearest 1,000 units (1 micro lot) for clean sizing
    return min(MAX_UNITS, max(1000, int(math.floor(raw_units / 1000) * 1000)))


def kelly_fraction(win_rate: float, rr_ratio: float) -> float:
    """
    Kelly Criterion: optimal fraction of bankroll to bet.

    f* = W - (1-W) / R
    where W = win rate, R = reward/risk ratio

    We apply a half-Kelly (f*/2) for safety — a common practitioner choice.
    Returns a fraction between 0.0 and 0.25 (capped for safety).
    """
    if rr_ratio <= 0 or win_rate <= 0:
        return 0.0
    full_kelly = win_rate - (1.0 - win_rate) / rr_ratio
    half_kelly = full_kelly / 2.0
    return max(0.0, min(half_kelly, 0.25))   # Cap at 25% of bankroll


# ── Confidence-Scaled Risk ───────────────────────────────────────────────────

def confidence_scaled_risk(confidence: float, min_conf: float = 0.55) -> float:
    """
    Map a signal confidence score to a risk percentage using fixed tiers.

    Tier table:
      < 0.55  → 0.0  (skip — signal too weak)
      0.55–0.65 → 2.0%
      0.65–0.75 → 3.0%
      0.75–0.85 → 4.0%
      > 0.85  → 5.0%

    Returns the risk % to use (0.0 means do not trade).
    """
    if confidence < min_conf:
        return 0.0
    if confidence >= 0.85:
        return 5.0
    if confidence >= 0.75:
        return 4.0
    if confidence >= 0.65:
        return 3.0
    return 2.0   # 0.55 – 0.65 tier


# ── Stop / Take-Profit Calculation ───────────────────────────────────────────

def atr_stop_loss(entry: float, atr: float, direction: str, multiplier: float = 1.5) -> float:
    """
    Calculate stop-loss price using ATR.

    For a BUY:  stop = entry - (atr * multiplier)
    For a SELL: stop = entry + (atr * multiplier)
    """
    delta = atr * multiplier
    return entry - delta if direction.upper() == "BUY" else entry + delta


def atr_take_profit(
    entry: float,
    stop_loss: float,
    direction: str,
    rr_ratio: float = 2.0,
) -> float:
    """
    Calculate take-profit using a fixed R:R ratio.

    risk = abs(entry - stop_loss)
    For BUY:  tp = entry + risk * rr_ratio
    For SELL: tp = entry - risk * rr_ratio
    """
    risk = abs(entry - stop_loss)
    if direction.upper() == "BUY":
        return entry + risk * rr_ratio
    return entry - risk * rr_ratio


# ── DataFrame Helpers ─────────────────────────────────────────────────────────

def ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalise a DataFrame to have lowercase OHLCV columns
    and a DatetimeIndex.  Raises ValueError if columns are missing.
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required OHLCV columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.set_index("time")
        else:
            df.index = pd.to_datetime(df.index, utc=True)
    if "volume" not in df.columns:
        df["volume"] = 0.0
    return df.sort_index()


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample a 1-minute OHLCV DataFrame to a higher timeframe.

    Args:
        df: DataFrame with DatetimeIndex and open/high/low/close/volume columns
        rule: Pandas offset alias, e.g. "5T", "15T", "1H"
    """
    return df.resample(rule).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()


# ── Formatting ────────────────────────────────────────────────────────────────

def format_price(price: float, instrument: str) -> str:
    """Format price to appropriate decimal places for display."""
    if "JPY" in instrument:
        decimals = 3
    elif instrument in _SPOT_METALS:
        decimals = 2
    else:
        decimals = 5
    return f"{price:.{decimals}f}"


def format_pips(pips: float) -> str:
    return f"{pips:.1f} pips"


def pct_change(old: float, new: float) -> float:
    """Percentage change from old to new."""
    if old == 0:
        return 0.0
    return (new - old) / abs(old) * 100.0
