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
from decimal import Decimal
from typing import Union

import pandas as pd


# ── Pip / Price Utilities ─────────────────────────────────────────────────────

# JPY pairs have a pip at the 2nd decimal place; all others at the 4th
_JPY_PAIRS = frozenset({"USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "CHF_JPY", "CAD_JPY", "NZD_JPY"})


def pip_size(instrument: str) -> float:
    """
    Return the pip size for an instrument.
    JPY pairs: 0.01,  all others: 0.0001
    """
    return 0.01 if any(jpy in instrument for jpy in ("JPY",)) else 0.0001


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

def units_from_risk(
    account_balance: float,
    risk_pct: float,
    stop_loss_pips: float,
    instrument: str,
    pip_value_per_unit: float = 1.0,
) -> int:
    """
    Calculate position size (units) given a fixed-risk model.

    Formula:
        risk_amount = account_balance * (risk_pct / 100)
        units = risk_amount / (stop_loss_pips * pip_value_per_unit)

    Args:
        account_balance: Account equity in account currency (USD assumed)
        risk_pct: Percentage of account to risk (e.g. 1.5 for 1.5%)
        stop_loss_pips: Distance to stop loss in pips
        instrument: e.g. "EUR_USD"
        pip_value_per_unit: USD value of 1 pip per 1 unit traded (~0.0001 for most pairs)

    Returns:
        Rounded position size in OANDA units (1 std lot = 100,000 units)
    """
    if stop_loss_pips <= 0:
        return 0
    risk_amount = account_balance * (risk_pct / 100.0)
    raw_units = risk_amount / (stop_loss_pips * pip_value_per_unit * pip_size(instrument))
    # Round down to nearest 1000 units (mini lot) for cleaner sizing
    return max(1000, int(math.floor(raw_units / 1000) * 1000))


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
    decimals = 3 if any(jpy in instrument for jpy in ("JPY",)) else 5
    return f"{price:.{decimals}f}"


def format_pips(pips: float) -> str:
    return f"{pips:.1f} pips"


def pct_change(old: float, new: float) -> float:
    """Percentage change from old to new."""
    if old == 0:
        return 0.0
    return (new - old) / abs(old) * 100.0
