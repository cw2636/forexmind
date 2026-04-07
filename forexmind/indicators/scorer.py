"""
ForexMind — Signal Scorer
===========================
Converts raw indicator values into a clean composite signal score (-100 to +100).

Positive score → bullish bias  (closer to +100 = stronger BUY)
Negative score → bearish bias  (closer to -100 = stronger SELL)
Near-zero      → no clear signal (HOLD)

Scoring Design:
  Each indicator group votes (+1 = bullish, -1 = bearish, 0 = neutral).
  Votes are weighted by category confidence and summed.
  Final score is normalised to [-100, +100].

Advanced Python concepts:
  - Named constants for weights (avoids magic numbers)
  - Dictionary dispatch pattern (score_*() functions keyed by name)
  - @dataclass for structured composite result
"""

from __future__ import annotations

from dataclasses import dataclass

from forexmind.indicators.engine import IndicatorSnapshot
from forexmind.utils.logger import get_logger
from forexmind.config.settings import get_settings

log = get_logger(__name__)

# ── Vote weights (must sum to 1.0) ────────────────────────────────────────────
WEIGHTS = {
    "trend":      0.30,   # EMA stack, MACD, ADX direction
    "momentum":   0.25,   # RSI, Stochastic, Williams %R
    "structure":  0.20,   # Price vs PSAR, BB position
    "volatility": 0.10,   # ATR regime, BB squeeze
    "volume":     0.15,   # OBV direction, MFI
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 0.001, "Weights must sum to 1.0"


@dataclass
class SignalScore:
    """Composite signal score with per-group breakdown."""
    composite: float          # Normalised [-100, +100]
    direction: str            # "BUY" | "SELL" | "HOLD"
    confidence: float         # 0.0–1.0
    trend_vote: float         # Raw vote for trend group
    momentum_vote: float
    structure_vote: float
    volatility_vote: float
    volume_vote: float
    reasoning: str            # Human-readable explanation for Claude


def score_snapshot(snap: IndicatorSnapshot, session_score: float | None = None) -> SignalScore:
    """
    Compute composite signal score from an IndicatorSnapshot.

    Returns a SignalScore with direction and confidence.

    Hard gates (force HOLD regardless of indicator votes):
      - ADX below minimum threshold (config: indicators.adx_trend_threshold, default 25):
        market is ranging/choppy — scalpers consistently lose here.
        Gate is set at 60% of the trend threshold to allow borderline trending markets.
      - ATR near zero: no volatility, probably off-hours or data gap
    """
    cfg = get_settings()
    adx_trend_threshold = cfg.indicator_config.get("adx_trend_threshold", 25)
    # Hard gate at 60% of the trend threshold (e.g. 15 when threshold=25).
    # Allows borderline-trending markets (ADX 15-25) to pass and get voted on,
    # instead of being hard-rejected at 20 regardless of all other indicators.
    adx_gate = adx_trend_threshold * 0.60

    # ── Hard gates — check before any voting ─────────────────────────────────
    adx_val = snap["adx"]
    if adx_val < adx_gate:
        return SignalScore(
            composite=0.0, direction="HOLD", confidence=0.0,
            trend_vote=0.0, momentum_vote=0.0, structure_vote=0.0,
            volatility_vote=0.0, volume_vote=0.0,
            reasoning=f"HOLD — ADX={adx_val:.1f} < {adx_gate:.0f}: ranging market, no directional edge",
        )

    if snap["atr"] == 0.0 or snap["atr_pct"] < 0.01:
        return SignalScore(
            composite=0.0, direction="HOLD", confidence=0.0,
            trend_vote=0.0, momentum_vote=0.0, structure_vote=0.0,
            volatility_vote=0.0, volume_vote=0.0,
            reasoning="HOLD — ATR near zero: insufficient volatility or data gap",
        )

    # ── Trend Group ───────────────────────────────────────────────────────────
    trend_votes: list[float] = []

    # EMA stack alignment — full alignment scores ±1.0, partial (weak) ±0.5
    if snap["ema_trend"] == "bullish":
        trend_votes.append(1.0)
    elif snap["ema_trend"] == "weak_bullish":
        trend_votes.append(0.5)    # Short-term bullish but 50 EMA not yet crossed
    elif snap["ema_trend"] == "bearish":
        trend_votes.append(-1.0)
    elif snap["ema_trend"] == "weak_bearish":
        trend_votes.append(-0.5)   # Short-term bearish but 50 EMA not yet crossed
    else:
        trend_votes.append(0.0)

    # Price vs EMA50 (key dynamic support/resistance)
    close = snap.get("bb_mid", snap["ema_21"])  # Use BB mid as close proxy
    if snap["ema_50"] > 0:
        # Use EMA21 vs EMA50 as proxy for momentum
        if snap["ema_21"] > snap["ema_50"]:
            trend_votes.append(0.5)
        elif snap["ema_21"] < snap["ema_50"]:
            trend_votes.append(-0.5)
        else:
            trend_votes.append(0.0)

    # MACD
    if snap["macd_cross"] == "bull_cross":
        trend_votes.append(1.0)
    elif snap["macd_cross"] == "bear_cross":
        trend_votes.append(-1.0)
    elif snap["macd"] > snap["macd_signal"] and snap["macd_hist"] > 0:
        trend_votes.append(0.6)
    elif snap["macd"] < snap["macd_signal"] and snap["macd_hist"] < 0:
        trend_votes.append(-0.6)
    else:
        trend_votes.append(0.0)

    # ADX directional index: +DI > -DI = bullish trend, +DI < -DI = bearish trend.
    # This confirms WHICH direction the trend is moving, not just that it IS trending.
    dmp = snap.get("dmp", 0.0)
    dmn = snap.get("dmn", 0.0)
    if dmp > 0 and dmn > 0:
        di_spread = (dmp - dmn) / max(dmp + dmn, 1.0)  # Normalized -1 to +1
        trend_votes.append(di_spread * 0.7)             # Weighted DI confirmation

    # ADX strength multiplier: scale votes up as ADX rises above threshold
    adx_multiplier = min(snap["adx"] / adx_trend_threshold, 1.5) if snap["adx"] >= adx_gate else 0.5
    trend_vote = _avg(trend_votes) * adx_multiplier

    # ── Momentum Group ────────────────────────────────────────────────────────
    momentum_votes: list[float] = []

    # RSI — directional bias with zone strength
    rsi = snap["rsi"]
    if rsi < 30:
        momentum_votes.append(1.0 * ((30 - rsi) / 30))   # Stronger oversold = stronger buy
    elif rsi > 70:
        momentum_votes.append(-1.0 * ((rsi - 70) / 30))  # Stronger overbought = stronger sell
    elif 40 < rsi < 50:
        momentum_votes.append(-0.2)   # Slightly bearish momentum
    elif 50 < rsi < 60:
        momentum_votes.append(0.2)    # Slightly bullish momentum
    else:
        momentum_votes.append(0.0)

    # Stochastic
    if snap["stoch_cross"] == "bull_cross" and snap["stoch_k"] < 30:
        momentum_votes.append(1.0)    # Oversold + cross = strong buy
    elif snap["stoch_cross"] == "bear_cross" and snap["stoch_k"] > 70:
        momentum_votes.append(-1.0)   # Overbought + cross = strong sell
    elif snap["stoch_k"] > snap["stoch_d"]:
        momentum_votes.append(0.3)
    elif snap["stoch_k"] < snap["stoch_d"]:
        momentum_votes.append(-0.3)
    else:
        momentum_votes.append(0.0)

    # Williams %R
    wr = snap["williams_r"]
    if wr < -80:
        momentum_votes.append(0.7)    # Oversold
    elif wr > -20:
        momentum_votes.append(-0.7)   # Overbought
    else:
        # Linear interpolation: -50 = neutral
        momentum_votes.append((wr + 50) / 50 * -0.3)

    # CCI
    cci = snap["cci"]
    if cci < -100:
        momentum_votes.append(0.5)
    elif cci > 100:
        momentum_votes.append(-0.5)
    else:
        momentum_votes.append(cci / 200 * -0.3)

    momentum_vote = _avg(momentum_votes)

    # ── Structure Group ───────────────────────────────────────────────────────
    structure_votes: list[float] = []

    # Parabolic SAR
    if snap["psar_signal"] == "bullish":
        structure_votes.append(0.8)
    elif snap["psar_signal"] == "bearish":
        structure_votes.append(-0.8)

    # BB position (0=lower band, 0.5=middle, 1.0=upper band)
    bb_pos = snap["bb_position"]
    if bb_pos < 0.1:
        structure_votes.append(0.7)    # Price near lower band = buy pressure
    elif bb_pos > 0.9:
        structure_votes.append(-0.7)   # Price near upper band = sell pressure
    elif bb_pos < 0.4:
        structure_votes.append(0.2)
    elif bb_pos > 0.6:
        structure_votes.append(-0.2)

    structure_vote = _avg(structure_votes)

    # ── Volatility Group ─────────────────────────────────────────────────────
    volatility_votes: list[float] = []

    # BB squeeze: narrow bands often precede a big move (directional ambiguous)
    # We give a small bonus if direction aligns with BB position
    bb_width = snap["bb_width"]
    if bb_width < 0.005:   # Very tight squeeze
        # Squeeze amplifies the structure vote
        volatility_votes.append(structure_vote * 0.5)
    else:
        volatility_votes.append(0.0)

    # ATR: high volatility = confirm existing direction signal, but also = wider stops needed
    atr_pct = snap["atr_pct"]
    if atr_pct > 0.1:     # Volatile market — reduce confidence slightly
        volatility_votes.append(-0.1)   # Penalise extreme volatility

    volatility_vote = _avg(volatility_votes) if volatility_votes else 0.0

    # ── Volume Group ──────────────────────────────────────────────────────────
    volume_votes: list[float] = []

    # MFI (money flow)
    mfi = snap["mfi"]
    if mfi < 20:
        volume_votes.append(0.8)   # Oversold by money flow
    elif mfi > 80:
        volume_votes.append(-0.8)
    elif mfi > 55:
        volume_votes.append(0.3)   # Bullish money flow
    elif mfi < 45:
        volume_votes.append(-0.3)

    volume_vote = _avg(volume_votes) if volume_votes else 0.0

    # ── Composite Score ───────────────────────────────────────────────────────
    raw = (
        trend_vote      * WEIGHTS["trend"]
        + momentum_vote * WEIGHTS["momentum"]
        + structure_vote* WEIGHTS["structure"]
        + volatility_vote * WEIGHTS["volatility"]
        + volume_vote   * WEIGHTS["volume"]
    )
    composite = round(raw * 100, 2)                    # Scale to [-100, +100]

    # ── Session quality bonus ─────────────────────────────────────────────────
    # London-NY overlap (session_score ≈ 0.8-1.0) produces the highest liquidity and
    # largest directional moves. Amplify the composite score by up to +15% during
    # prime sessions — signals taken during high-liquidity windows are historically
    # more reliable for scalping.
    if session_score is not None and abs(composite) > 5.0:
        # Boost: +0% at session_score=0.5, +15% at session_score=1.0
        # No boost or reduction below 0.5 (keep off-hours signals unchanged)
        if session_score > 0.5:
            session_boost = 1.0 + (session_score - 0.5) * 0.30  # max +15%
            composite = composite * session_boost

    composite = max(-100.0, min(100.0, composite))

    # ── Direction and Confidence ──────────────────────────────────────────────
    abs_score = abs(composite)
    if composite > 15:
        direction = "BUY"
    elif composite < -15:
        direction = "SELL"
    else:
        direction = "HOLD"

    # Confidence: proportion of max possible score, scaled to [0,1]
    confidence = round(min(abs_score / 65.0, 1.0), 4)

    # ── Reasoning string for the AI agent ─────────────────────────────────────
    reasoning = _build_reasoning(snap, trend_vote, momentum_vote, structure_vote, composite)

    return SignalScore(
        composite=composite,
        direction=direction,
        confidence=confidence,
        trend_vote=round(trend_vote, 3),
        momentum_vote=round(momentum_vote, 3),
        structure_vote=round(structure_vote, 3),
        volatility_vote=round(volatility_vote, 3),
        volume_vote=round(volume_vote, 3),
        reasoning=reasoning,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _avg(votes: list[float]) -> float:
    return sum(votes) / len(votes) if votes else 0.0


def _build_reasoning(
    snap: IndicatorSnapshot,
    trend_vote: float,
    momentum_vote: float,
    structure_vote: float,
    composite: float,
) -> str:
    parts: list[str] = []

    parts.append(f"EMA trend: {snap['ema_trend']} (9={snap['ema_9']:.5f}, 21={snap['ema_21']:.5f}, 50={snap['ema_50']:.5f})")
    parts.append(f"MACD: {snap['macd']:.5f} vs signal {snap['macd_signal']:.5f} [{snap['macd_cross']}]")
    parts.append(f"ADX={snap['adx']:.1f} [{snap['adx_trend_strength']}], PSAR={snap['psar_signal']}")
    parts.append(f"RSI={snap['rsi']:.1f} [{snap['rsi_zone']}], Stoch K={snap['stoch_k']:.1f} D={snap['stoch_d']:.1f} [{snap['stoch_cross']}]")
    parts.append(f"BB position={snap['bb_position']:.2f} (0=lower,1=upper), ATR={snap['atr']:.5f} ({snap['atr_pct']:.2f}%)")
    parts.append(f"MFI={snap['mfi']:.1f}, Williams %R={snap['williams_r']:.1f}")
    parts.append(f"Support={snap['support']:.5f}, Resistance={snap['resistance']:.5f}")
    parts.append(
        f"Vote breakdown — Trend:{trend_vote:+.2f} Momentum:{momentum_vote:+.2f} "
        f"Structure:{structure_vote:+.2f} → Composite: {composite:+.1f}"
    )
    return " | ".join(parts)
