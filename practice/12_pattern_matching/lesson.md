# Topic 12 — Structural Pattern Matching

## The Four Interview Questions

---

### 1. WHAT is it?

`match`/`case` (Python 3.10+) is structural pattern matching — a powerful switch
statement that matches values by **structure**, not just equality.
It's far more expressive than `if/elif` chains.

---

### 2. WHY does Python have it?

Complex routing logic (like dispatching trading signals by type, or parsing
API responses) becomes verbose and fragile with `if/elif`. Pattern matching
makes the structure explicit and exhaustive.

---

### 3. SHOW ME — All Pattern Types

```python
# ── 1. Literal patterns ──────────────────────────────────────────────────────
def describe_direction(direction: str) -> str:
    match direction:
        case "BUY":
            return "Long position"
        case "SELL":
            return "Short position"
        case "HOLD":
            return "No action"
        case _:                  # wildcard — matches everything else
            return f"Unknown: {direction}"


# ── 2. Capture patterns ───────────────────────────────────────────────────────
def route_signal(signal: dict) -> str:
    match signal:
        case {"direction": "BUY", "confidence": c} if c >= 0.7:
            return f"Strong BUY (confidence {c})"
        case {"direction": "BUY", "confidence": c}:
            return f"Weak BUY (confidence {c})"
        case {"direction": "SELL"}:
            return "SELL signal"
        case _:
            return "HOLD"


# ── 3. Class patterns ─────────────────────────────────────────────────────────
from dataclasses import dataclass

@dataclass
class BuySignal:
    pair: str
    confidence: float

@dataclass
class SellSignal:
    pair: str
    confidence: float

@dataclass
class HoldSignal:
    pair: str

def process(signal):
    match signal:
        case BuySignal(pair=p, confidence=c) if c >= 0.7:
            return f"Execute BUY on {p} (strong)"
        case BuySignal(pair=p, confidence=c):
            return f"Tentative BUY on {p} (weak)"
        case SellSignal(pair=p):
            return f"Execute SELL on {p}"
        case HoldSignal(pair=p):
            return f"Hold {p}"


# ── 4. Sequence patterns ─────────────────────────────────────────────────────
def describe_candles(closes: list) -> str:
    match closes:
        case []:
            return "No data"
        case [single]:
            return f"Single candle: {single}"
        case [first, *rest] if first > rest[-1]:
            return f"Downtrend: {first} → {rest[-1]}"
        case [first, *rest]:
            return f"Uptrend: {first} → {rest[-1]}"


# ── 5. OR patterns ────────────────────────────────────────────────────────────
def is_major_session(session: str) -> bool:
    match session:
        case "london" | "new_york" | "overlap":
            return True
        case _:
            return False


# ── 6. Guard clauses ─────────────────────────────────────────────────────────
def risk_check(proposal: dict) -> str:
    match proposal:
        case {"risk_pct": r, "units": u} if r > 5.0:
            return f"REJECTED: risk {r}% too high"
        case {"risk_pct": r, "units": u} if u > 10_000:
            return f"REJECTED: units {u} too large"
        case {"risk_pct": r, "units": u}:
            return f"APPROVED: {r}% risk, {u} units"
```

---

## Quick Reference Cheatsheet

```python
match value:
    case 42:                            # literal
        ...
    case x:                             # capture into x
        ...
    case int() | str():                 # type OR
        ...
    case [first, *rest]:                # sequence
        ...
    case {"key": v}:                    # mapping (partial match)
        ...
    case MyClass(attr=val):             # class pattern
        ...
    case _ if condition:                # guard on wildcard
        ...
    case _:                             # wildcard (default)
        ...
```
