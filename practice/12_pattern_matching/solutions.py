"""
Topic 12 — Structural Pattern Matching
========================================
SOLUTIONS

Run: python 12_pattern_matching/solutions.py
Requires Python >= 3.10
"""

from __future__ import annotations
from dataclasses import dataclass
import sys

if sys.version_info < (3, 10):
    print("Pattern matching requires Python 3.10+. You have:", sys.version)
    sys.exit(1)


# ── Shared dataclasses (same as exercises.py) ────────────────────────────────

@dataclass
class Buy:
    pair: str
    units: int
    confidence: float

@dataclass
class Sell:
    pair: str
    units: int

@dataclass
class Hold:
    pair: str


# ── Exercise 1: Literal + wildcard ───────────────────────────────────────────
# match/case on a plain string — covers BUY / SELL / HOLD and a catch-all.

def route_direction(d: str) -> str:
    match d:
        case "BUY":
            return "Going LONG"
        case "SELL":
            return "Going SHORT"
        case "HOLD":
            return "Staying flat"
        case _:
            return f"Invalid direction: {d}"


# ── Exercise 2: Dict pattern with guard ──────────────────────────────────────
# Dict patterns match *subset* keys — extra keys are fine.
# Guards (if ...) are evaluated after the pattern matches.

def evaluate_signal(sig: dict) -> str:
    match sig:
        case {"direction": "BUY", "confidence": c} if c >= 0.7:
            return "Strong BUY"
        case {"direction": "BUY", "confidence": c}:
            return f"Weak BUY ({c:.2f})"
        case {"direction": "SELL"}:
            return "SELL — execute"
        case {"direction": "HOLD"}:
            return "Hold"
        case _:
            return "Malformed signal"


# ── Exercise 3: Class pattern ────────────────────────────────────────────────
# Python uses positional OR keyword class patterns.
# Keyword form: ClassName(attr=var) binds the attribute value to `var`.

def dispatch(cmd) -> str:
    match cmd:
        case Buy(pair=p, units=u, confidence=c) if c >= 0.8:
            return f"Execute strong BUY on {p} x{u}"
        case Buy(pair=p, confidence=c):
            return f"Tentative BUY on {p}"
        case Sell(pair=p, units=u):
            return f"Execute SELL on {p} x{u}"
        case Hold(pair=p):
            return f"Hold {p}"


# ── Exercise 4: Sequence pattern ─────────────────────────────────────────────
# [first, *mid, last] — *mid captures zero or more middle elements.
# Ordered most-specific → least-specific ([] before [x] before rest).

def classify_trend(closes: list) -> str:
    match closes:
        case []:
            return "no data"
        case [x]:
            return f"single: {x}"
        case [a, b] if a < b:
            return "simple uptrend"
        case [a, b]:
            return "simple downtrend or flat"
        case [first, *mid, last] if last > first:
            return f"uptrend: {first:.4f}→{last:.4f}"
        case [first, *mid, last]:
            return f"downtrend: {first:.4f}→{last:.4f}"


# ── Exercise 5: OR pattern + guard chains ────────────────────────────────────
# p1 | p2 | p3 — tries each alternative; first match wins.
# Capture variable (p if ...) after specific literals fall through.

def validate_pair(pair: str) -> str:
    match pair:
        case "EUR_USD" | "EUR_GBP" | "EUR_JPY":
            return "Euro pair — liquid"
        case "GBP_USD" | "GBP_JPY":
            return "Sterling pair"
        case p if p.endswith("_JPY"):
            return f"JPY pair: {p}"
        case p if "_" in p:
            return f"Other: {p}"
        case _:
            return "Invalid format"


# ── Tests ─────────────────────────────────────────────────────────────────────

def tests():
    print("--- Exercise 1: Literal + wildcard ---")
    assert route_direction("BUY")  == "Going LONG"
    assert route_direction("SELL") == "Going SHORT"
    assert route_direction("HOLD") == "Staying flat"
    assert route_direction("LONG") == "Invalid direction: LONG"
    print("  PASS")

    print("--- Exercise 2: Dict pattern with guard ---")
    assert evaluate_signal({"direction": "BUY", "confidence": 0.8}) == "Strong BUY"
    assert evaluate_signal({"direction": "BUY", "confidence": 0.5}) == "Weak BUY (0.50)"
    assert evaluate_signal({"direction": "SELL"}) == "SELL — execute"
    assert evaluate_signal({"direction": "HOLD"}) == "Hold"
    assert evaluate_signal({}) == "Malformed signal"
    print("  PASS")

    print("--- Exercise 3: Class pattern ---")
    assert dispatch(Buy("EUR_USD", 1000, 0.9)) == "Execute strong BUY on EUR_USD x1000"
    assert dispatch(Buy("EUR_USD", 1000, 0.5)) == "Tentative BUY on EUR_USD"
    assert dispatch(Sell("GBP_USD", 500))      == "Execute SELL on GBP_USD x500"
    assert dispatch(Hold("USD_JPY"))            == "Hold USD_JPY"
    print("  PASS")

    print("--- Exercise 4: Sequence pattern ---")
    assert classify_trend([])                          == "no data"
    assert classify_trend([1.1050])                    == "single: 1.105"
    assert classify_trend([1.0, 2.0])                  == "simple uptrend"
    assert classify_trend([2.0, 1.0])                  == "simple downtrend or flat"
    assert "uptrend" in classify_trend([1.1050, 1.1060, 1.1055, 1.1080])
    assert "downtrend" in classify_trend([1.1080, 1.1060, 1.1050])
    print("  PASS")

    print("--- Exercise 5: OR pattern + guard ---")
    assert validate_pair("EUR_USD") == "Euro pair — liquid"
    assert validate_pair("EUR_GBP") == "Euro pair — liquid"
    assert validate_pair("GBP_USD") == "Sterling pair"
    assert validate_pair("AUD_JPY") == "JPY pair: AUD_JPY"
    assert validate_pair("AUD_USD") == "Other: AUD_USD"
    assert validate_pair("EURUSD")  == "Invalid format"
    print("  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("Topic 12 — Pattern Matching — SOLUTIONS")
    print("=" * 60)
    tests()
    print("\nAll solutions verified!")
    print("=" * 60)
