"""
COT (Commitment of Traders) data fetcher.

Downloads CFTC weekly disaggregated futures-and-options COT data,
maps currency futures to forex pair instruments, and computes net
speculator position as a directional bias feature.

Cache: refreshed once per week, on or after Friday 20:30 UTC
(CFTC releases at ~15:30 ET / 20:30 UTC on Fridays).
"""

from __future__ import annotations

import csv
import io
import logging
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional
from urllib.request import urlopen, Request

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CFTC disaggregated COT: Futures-Only (annual ZIP, most recent year first)
# ---------------------------------------------------------------------------
_COT_URL = "https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"
_CACHE_DIR = Path(__file__).parent.parent / "cache"
_CACHE_FILE = _CACHE_DIR / "cot_cache.json"

# How many seconds before we consider the cache stale (7 days)
_CACHE_TTL = 7 * 24 * 3600

# ---------------------------------------------------------------------------
# Currency futures CFTC market codes → forex pair instrument
#   net_position = Large Spec Longs - Large Spec Shorts
#   inverted=True  → instrument is USD/XXX so we flip the sign
# ---------------------------------------------------------------------------
_FUTURES_MAP: Dict[str, Dict] = {
    "EURO FX":                {"instrument": "EUR_USD", "inverted": False},
    "BRITISH POUND":          {"instrument": "GBP_USD", "inverted": False},
    "JAPANESE YEN":           {"instrument": "USD_JPY", "inverted": True},
    "AUSTRALIAN DOLLAR":      {"instrument": "AUD_USD", "inverted": False},
    "CANADIAN DOLLAR":        {"instrument": "USD_CAD", "inverted": True},
    "SWISS FRANC":            {"instrument": "USD_CHF", "inverted": True},
    "NEW ZEALAND DOLLAR":     {"instrument": "NZD_USD", "inverted": False},
    "MEXICAN PESO":           {"instrument": "USD_MXN", "inverted": True},
}

# Column names in the CFTC disaggregated CSV (partial list we care about)
_COL_MARKET   = "Market_and_Exchange_Names"
_COL_DATE     = "Report_Date_as_MM_DD_YYYY"
_COL_LS_LONG  = "Lev_Money_Positions_Long_All"   # Leveraged Money (specs) longs
_COL_LS_SHORT = "Lev_Money_Positions_Short_All"  # Leveraged Money (specs) shorts
_COL_NR_LONG  = "NonRept_Positions_Long_All"     # Non-reportable (retail) longs  — not used but kept
_COL_NR_SHORT = "NonRept_Positions_Short_All"

# ---------------------------------------------------------------------------
# In-memory cache  { instrument: {"net_position": int, "direction": str,
#                                  "change": int, "date": str,
#                                  "fetched_at": float} }
# ---------------------------------------------------------------------------
_cache: Dict[str, dict] = {}
_last_fetch: float = 0.0


def _need_refresh() -> bool:
    """Return True if cache is empty or older than TTL."""
    if not _cache:
        return True
    age = time.time() - _last_fetch
    return age > _CACHE_TTL


def _download_cot_csv(year: int) -> Optional[str]:
    """Download CFTC ZIP for *year* and return the CSV text, or None on error."""
    url = _COT_URL.format(year=year)
    try:
        req = Request(url, headers={"User-Agent": "ForexMind/1.0"})
        resp = urlopen(req, timeout=30)
        raw = resp.read()
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            # The ZIP contains one .txt (CSV) file
            names = [n for n in zf.namelist() if n.endswith(".txt")]
            if not names:
                _log.warning("COT ZIP for %d has no .txt file", year)
                return None
            with zf.open(names[0]) as f:
                return f.read().decode("latin-1")
    except Exception as exc:
        _log.warning("COT download failed for year %d: %s", year, exc)
        return None


def _parse_cot_csv(csv_text: str) -> Dict[str, dict]:
    """
    Parse the CFTC disaggregated CSV and return the most recent row for
    each currency futures market we care about.

    Returns { instrument: {"net_position": int, "prev_net": int, "date": str} }
    """
    reader = csv.DictReader(io.StringIO(csv_text))

    # Collect all rows for markets we care about, keyed by market name
    rows_by_market: Dict[str, list] = {m: [] for m in _FUTURES_MAP}

    for row in reader:
        market = row.get(_COL_MARKET, "").strip().upper()
        if market in _FUTURES_MAP:
            rows_by_market[market].append(row)

    result: Dict[str, dict] = {}

    for market, rows in rows_by_market.items():
        if not rows:
            continue

        mapping = _FUTURES_MAP[market]
        instrument = mapping["instrument"]
        inverted = mapping["inverted"]

        # Sort descending by date so rows[0] is most recent
        def _parse_date(r: dict) -> datetime:
            try:
                return datetime.strptime(r[_COL_DATE].strip(), "%m/%d/%Y")
            except Exception:
                return datetime.min

        rows.sort(key=_parse_date, reverse=True)

        def _net(row: dict) -> int:
            try:
                ls_long  = int(row[_COL_LS_LONG].replace(",", ""))
                ls_short = int(row[_COL_LS_SHORT].replace(",", ""))
                net = ls_long - ls_short
                return -net if inverted else net
            except Exception:
                return 0

        latest = rows[0]
        net = _net(latest)
        prev_net = _net(rows[1]) if len(rows) > 1 else net
        change = net - prev_net

        result[instrument] = {
            "net_position": net,
            "change":       change,
            "date":         latest.get(_COL_DATE, "unknown").strip(),
        }

    return result


def _classify_direction(net: int, change: int) -> str:
    """
    Convert numeric net position to directional bias.
    - Pure sign of net_position gives structural bias.
    - change adds momentum confirmation: only flip to NEUTRAL if change
      is moving against the position significantly (>10% reversal of net).
    """
    if net == 0:
        return "NEUTRAL"
    if net > 0:
        # Specs are net long → BUY bias, unless momentum reversing hard
        if change < 0 and abs(change) > abs(net) * 0.10:
            return "NEUTRAL"
        return "BUY"
    else:
        # Specs are net short → SELL bias
        if change > 0 and abs(change) > abs(net) * 0.10:
            return "NEUTRAL"
        return "SELL"


def _refresh() -> None:
    """Download and parse COT data; populate in-memory cache."""
    global _last_fetch, _cache

    now = datetime.now(timezone.utc)
    year = now.year

    csv_text = _download_cot_csv(year)
    if csv_text is None:
        # Try previous year as fallback (e.g., early January)
        csv_text = _download_cot_csv(year - 1)
    if csv_text is None:
        _log.error("COT: failed to download data for %d and %d", year, year - 1)
        return

    parsed = _parse_cot_csv(csv_text)

    new_cache: Dict[str, dict] = {}
    for instrument, data in parsed.items():
        net    = data["net_position"]
        change = data["change"]
        direction = _classify_direction(net, change)
        new_cache[instrument] = {
            "net_position": net,
            "direction":    direction,
            "change":       change,
            "date":         data["date"],
            "fetched_at":   time.time(),
        }
        _log.info(
            "COT %-10s  net=%+8d  chg=%+7d  dir=%s  (%s)",
            instrument, net, change, direction, data["date"],
        )

    _cache = new_cache
    _last_fetch = time.time()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_cot_bias(instrument: str) -> dict:
    """
    Return COT positioning bias for *instrument*.

    Returns a dict:
        {
            "net_position": int,     # Leveraged-money net longs (positive = bullish)
            "direction":    str,     # "BUY" | "SELL" | "NEUTRAL"
            "change":       int,     # week-over-week change in net position
            "date":         str,     # COT report date (MM/DD/YYYY)
        }

    Returns an empty dict if the instrument is not covered or data unavailable.
    """
    if _need_refresh():
        try:
            _refresh()
        except Exception as exc:
            _log.warning("COT refresh error: %s", exc)

    entry = _cache.get(instrument)
    if entry is None:
        return {}

    return {
        "net_position": entry["net_position"],
        "direction":    entry["direction"],
        "change":       entry["change"],
        "date":         entry["date"],
    }


def get_all_cot_biases() -> Dict[str, dict]:
    """Return COT bias for all tracked instruments."""
    if _need_refresh():
        try:
            _refresh()
        except Exception as exc:
            _log.warning("COT refresh error: %s", exc)
    return {
        instr: {
            "net_position": v["net_position"],
            "direction":    v["direction"],
            "change":       v["change"],
            "date":         v["date"],
        }
        for instr, v in _cache.items()
    }
