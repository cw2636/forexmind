"""
ForexMind — Forex Market Session Detector
===========================================
Identifies which trading sessions are currently active (UTC) and whether
we are in a high-volatility overlap window.

Scalpers should primarily trade London and New York overlap (12:00–16:00 UTC)
and the Tokyo–London overlap (07:00–09:00 UTC) for maximum liquidity.

Advanced Python concepts:
  - datetime / timezone handling with pytz & zoneinfo
  - named tuples for structured return types
  - __slots__ for memory-efficient dataclasses
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import NamedTuple

import pytz

UTC = pytz.utc


@dataclass(frozen=True, slots=True)
class SessionWindow:
    name: str
    open_utc: time     # Opening time in UTC
    close_utc: time    # Closing time in UTC (may be < open if spans midnight)


# ── Session definitions (all UTC) ─────────────────────────────────────────────
SESSIONS: list[SessionWindow] = [
    SessionWindow("Sydney",   time(21, 0), time(6, 0)),
    SessionWindow("Tokyo",    time(0, 0),  time(9, 0)),
    SessionWindow("London",   time(7, 0),  time(16, 0)),
    SessionWindow("New York", time(12, 0), time(21, 0)),
]

# High-volatility overlap windows — prime scalping windows
OVERLAP_WINDOWS: list[SessionWindow] = [
    SessionWindow("London–NY Overlap",     time(12, 0), time(16, 0)),
    SessionWindow("Tokyo–London Overlap",  time(7, 0),  time(9, 0)),
]


class SessionStatus(NamedTuple):
    """Snapshot of which sessions are active at a given moment."""
    active_sessions: list[str]
    active_overlaps: list[str]
    is_overlap: bool        # True if in any high-volatility overlap
    is_weekend: bool        # Forex closes Friday 21:00 UTC – Sunday 21:00 UTC
    session_score: float    # 0.0–1.0; higher = more liquidity / better to trade


def _is_in_window(window: SessionWindow, dt: datetime) -> bool:
    """Check if a UTC datetime falls within a session window (handles midnight-spanning)."""
    t = dt.time().replace(second=0, microsecond=0)
    o, c = window.open_utc, window.close_utc
    if o < c:
        return o <= t < c
    # Window spans midnight (e.g. Sydney 21:00–06:00)
    return t >= o or t < c


def get_session_status(dt: datetime | None = None) -> SessionStatus:
    """
    Return the current session status.

    Args:
        dt: UTC datetime to evaluate. Defaults to now() if None.

    Returns:
        SessionStatus named tuple with active sessions, overlaps, and a
        session_score (0.0–1.0) that can be used as a trading filter.
    """
    if dt is None:
        dt = datetime.now(UTC)
    elif dt.tzinfo is None:
        dt = UTC.localize(dt)

    # Forex weekend: approx Friday 21:00 to Sunday 21:00 UTC
    is_weekend = dt.weekday() == 5 or (dt.weekday() == 6) or (
        dt.weekday() == 4 and dt.hour >= 21
    )
    if is_weekend:
        return SessionStatus([], [], False, True, 0.0)

    active = [s.name for s in SESSIONS if _is_in_window(s, dt)]
    overlaps = [o.name for o in OVERLAP_WINDOWS if _is_in_window(o, dt)]

    # Score: base 0.2 per session, +0.3 bonus for each overlap
    score = min(1.0, len(active) * 0.2 + len(overlaps) * 0.3)

    return SessionStatus(
        active_sessions=active,
        active_overlaps=overlaps,
        is_overlap=bool(overlaps),
        is_weekend=False,
        session_score=round(score, 2),
    )


def best_pairs_for_session(dt: datetime | None = None) -> list[str]:
    """
    Suggest the most liquid pairs for the current session.
    Used as a pre-filter by the indicator engine.
    """
    status = get_session_status(dt)
    active = set(status.active_sessions)

    recommendations: list[str] = []
    if "London" in active or "New York" in active:
        recommendations += ["EUR_USD", "GBP_USD", "USD_CHF", "EUR_GBP", "USD_CAD"]
    if "Tokyo" in active:
        recommendations += ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY"]
    if "Sydney" in active:
        recommendations += ["AUD_USD", "NZD_USD", "AUD_JPY"]

    # Dedupe while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for pair in recommendations:
        if pair not in seen:
            seen.add(pair)
            result.append(pair)
    return result
