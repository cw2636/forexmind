"""
Topic 12 — Structural Pattern Matching
========================================
EXERCISES + SOLUTIONS

Run: python 12_pattern_matching/exercises.py
Requires Python >= 3.10
"""

from __future__ import annotations
from dataclasses import dataclass
import sys

if sys.version_info < (3, 10):
    print("Pattern matching requires Python 3.10+. You have:", sys.version)
    sys.exit(1)


# ── EXERCISE 1: Literal + wildcard ───────────────────────────────────────────
# Write route_direction(d: str) -> str using match/case:
#   "BUY"  → "Going LONG"
#   "SELL" → "Going SHORT"
#   "HOLD" → "Staying flat"
#   other  → "Invalid direction: <d>"
# TODO

# ── EXERCISE 2: Dict pattern with guard ──────────────────────────────────────
# Write evaluate_signal(sig: dict) -> str:
#   {"direction":"BUY","confidence":c} where c>=0.7 → "Strong BUY"
#   {"direction":"BUY","confidence":c}              → f"Weak BUY ({c:.2f})"
#   {"direction":"SELL"}                            → "SELL — execute"
#   {"direction":"HOLD"}                            → "Hold"
#   anything else                                   → "Malformed signal"
# TODO

# ── EXERCISE 3: Class pattern ────────────────────────────────────────────────
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

# Write dispatch(cmd) -> str using class patterns.
# Buy with confidence>=0.8 → "Execute strong BUY on <pair> x<units>"
# Buy with lower confidence → "Tentative BUY on <pair>"
# Sell                      → "Execute SELL on <pair> x<units>"
# Hold                      → "Hold <pair>"
# TODO

# ── EXERCISE 4: Sequence pattern ─────────────────────────────────────────────
# Write classify_trend(closes: list) -> str:
#   []          → "no data"
#   [x]         → f"single: {x}"
#   [a, b] if a<b → "simple uptrend"
#   [a, b]      → "simple downtrend or flat"
#   [first, *mid, last] if last>first → f"uptrend: {first:.4f}→{last:.4f}"
#   [first, *mid, last]               → f"downtrend: {first:.4f}→{last:.4f}"
# TODO

# ── EXERCISE 5: OR pattern + guard chains ────────────────────────────────────
# Write validate_pair(pair: str) -> str:
#   "EUR_USD" | "EUR_GBP" | "EUR_JPY" → "Euro pair — liquid"
#   "GBP_USD" | "GBP_JPY"             → "Sterling pair"
#   p if p.endswith("_JPY")           → f"JPY pair: {p}"
#   p if "_" in p                     → f"Other: {p}"
#   _                                 → "Invalid format"
# TODO


# ─────────────────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────────────────

def tests():
    print("--- Exercise 1 ---")
    assert route_direction("BUY")  == "Going LONG"
    assert route_direction("HOLD") == "Staying flat"
    assert "Invalid" in route_direction("LONG")
    print("  PASS")

    print("--- Exercise 2 ---")
    assert evaluate_signal({"direction":"BUY","confidence":0.8}) == "Strong BUY"
    assert evaluate_signal({"direction":"BUY","confidence":0.5}).startswith("Weak BUY")
    assert evaluate_signal({"direction":"SELL"}) == "SELL — execute"
    assert evaluate_signal({}) == "Malformed signal"
    print("  PASS")

    print("--- Exercise 3 ---")
    assert "strong" in dispatch(Buy("EUR_USD",1000,0.9))
    assert "Tentative" in dispatch(Buy("EUR_USD",1000,0.5))
    assert "SELL" in dispatch(Sell("GBP_USD",500))
    assert "Hold" in dispatch(Hold("USD_JPY"))
    print("  PASS")

    print("--- Exercise 4 ---")
    assert classify_trend([]) == "no data"
    assert classify_trend([1.1050]) == "single: 1.105"
    assert "uptrend" in classify_trend([1.1050,1.1060,1.1055,1.1080])
    assert "downtrend" in classify_trend([1.1080,1.1060,1.1050])
    print("  PASS")

    print("--- Exercise 5 ---")
    assert "Euro" in validate_pair("EUR_USD")
    assert "Sterling" in validate_pair("GBP_USD")
    assert "JPY" in validate_pair("AUD_JPY")
    assert "Other" in validate_pair("AUD_USD")
    assert "Invalid" in validate_pair("EURUSD")
    print("  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("Topic 12 — Pattern Matching — Exercise Runner")
    print("=" * 60)
    try:
        tests()
    except (NameError, AssertionError) as e:
        print(f"  INCOMPLETE: {e}")
    print("=" * 60)


# ── SOLUTIONS ─────────────────────────────────────────────────────────────────
"""
def route_direction(d):
    match d:
        case "BUY":  return "Going LONG"
        case "SELL": return "Going SHORT"
        case "HOLD": return "Staying flat"
        case _: return f"Invalid direction: {d}"

def evaluate_signal(sig):
    match sig:
        case {"direction":"BUY","confidence":c} if c>=0.7: return "Strong BUY"
        case {"direction":"BUY","confidence":c}: return f"Weak BUY ({c:.2f})"
        case {"direction":"SELL"}: return "SELL — execute"
        case {"direction":"HOLD"}: return "Hold"
        case _: return "Malformed signal"

def dispatch(cmd):
    match cmd:
        case Buy(pair=p,units=u,confidence=c) if c>=0.8:
            return f"Execute strong BUY on {p} x{u}"
        case Buy(pair=p,confidence=c):
            return f"Tentative BUY on {p}"
        case Sell(pair=p,units=u):
            return f"Execute SELL on {p} x{u}"
        case Hold(pair=p):
            return f"Hold {p}"

def classify_trend(closes):
    match closes:
        case []: return "no data"
        case [x]: return f"single: {x}"
        case [a,b] if a<b: return "simple uptrend"
        case [a,b]: return "simple downtrend or flat"
        case [first,*mid,last] if last>first:
            return f"uptrend: {first:.4f}→{last:.4f}"
        case [first,*mid,last]:
            return f"downtrend: {first:.4f}→{last:.4f}"

def validate_pair(pair):
    match pair:
        case "EUR_USD"|"EUR_GBP"|"EUR_JPY": return "Euro pair — liquid"
        case "GBP_USD"|"GBP_JPY": return "Sterling pair"
        case p if p.endswith("_JPY"): return f"JPY pair: {p}"
        case p if "_" in p: return f"Other: {p}"
        case _: return "Invalid format"
"""
