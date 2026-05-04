"""
ForexMind — OANDA REST API Client
=====================================
Full async wrapper around the oandapyV20 library.

Covers:
  - Fetching historical OHLCV candles (any instrument, any granularity)
  - Fetching live bid/ask prices (pricing stream)
  - Account summary (balance, margin, open positions)
  - Placing, modifying, and closing orders/trades
  - Fetching open positions

OANDA Practice Account setup (free):
  1. Register at https://www.oanda.com/register/#/
  2. My Account → Manage API Access → Generate Token
  3. Copy API key and Account ID to .env

Advanced Python concepts:
  - asyncio.to_thread() to run synchronous code in a thread pool
  - Retry logic with exponential back-off (custom decorator)
  - dataclasses for structured return types
  - Generator-based streaming
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, TypeVar

import pandas as pd

try:
    import oandapyV20
    import oandapyV20.endpoints.accounts as accounts
    import oandapyV20.endpoints.instruments as instruments
    import oandapyV20.endpoints.orders as orders
    import oandapyV20.endpoints.positions as positions
    import oandapyV20.endpoints.pricing as pricing
    import oandapyV20.endpoints.trades as trades_ep
    from oandapyV20 import API as OandaAPI
    from oandapyV20.exceptions import V20Error
    OANDA_AVAILABLE = True
except ImportError:
    OANDA_AVAILABLE = False

from forexmind.config.settings import get_settings
from forexmind.utils.logger import get_logger

log = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# OANDA granularity map: internal name → OANDA enum
GRANULARITY_MAP: dict[str, str] = {
    "M1": "M1", "M5": "M5", "M15": "M15", "M30": "M30",
    "H1": "H1", "H4": "H4", "D": "D", "W": "W",
}


# ── Return Types ──────────────────────────────────────────────────────────────

@dataclass
class AccountSummary:
    account_id: str
    balance: float
    nav: float              # Net Asset Value
    unrealized_pnl: float
    realized_pnl: float
    margin_used: float
    margin_available: float
    open_trade_count: int
    currency: str


@dataclass
class LivePrice:
    instrument: str
    bid: float
    ask: float
    mid: float
    tradeable: bool
    timestamp: datetime


@dataclass
class OrderResult:
    success: bool
    trade_id: str = ""
    order_id: str = ""
    filled_price: float = 0.0
    units: int = 0
    message: str = ""


# ── Retry Decorator ───────────────────────────────────────────────────────────

def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator for retrying a coroutine on transient errors.
    Implements exponential backoff: delay, delay*backoff, delay*backoff^2, ...
    On ConnectionError / RemoteDisconnected, recreates the OANDA client before
    retrying so stale HTTP connections don't persist.
    """
    def decorator(fn: F) -> F:
        @wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        raise
                    err_str = str(e).lower()
                    is_connection_err = any(k in err_str for k in (
                        "connection aborted", "remotedisconnected",
                        "connection reset", "broken pipe", "connection refused",
                    ))
                    # Recreate the underlying OANDA API client on connection errors
                    # so stale pooled connections are discarded before the retry.
                    if is_connection_err and args and hasattr(args[0], "_client"):
                        try:
                            cfg = args[0]._cfg
                            args[0]._client = OandaAPI(
                                access_token=cfg.api_key,
                                environment=cfg.environment,
                            )
                            log.info(f"[{fn.__name__}] Reconnected OANDA client after connection error")
                        except Exception:
                            pass
                    log.warning(
                        f"[{fn.__name__}] attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
        return wrapper  # type: ignore[return-value]
    return decorator


# ── OANDA Client ──────────────────────────────────────────────────────────────

class OandaClient:
    """
    Async-friendly OANDA REST API client.

    oandapyV20 is synchronous, so all blocking calls are wrapped in
    asyncio.to_thread() to keep the event loop unblocked.
    """

    def __init__(self) -> None:
        if not OANDA_AVAILABLE:
            raise ImportError("oandapyV20 is not installed. Run: pip install oandapyV20")

        cfg = get_settings().oanda
        if not cfg.is_configured:
            raise ValueError(
                "OANDA_API_KEY and/or OANDA_ACCOUNT_ID are not set in .env"
            )

        self._cfg = cfg
        self._client = OandaAPI(
            access_token=cfg.api_key,
            environment=cfg.environment,
        )
        log.info(f"OandaClient ready ({cfg.environment} environment)")

    # ── Account ────────────────────────────────────────────────────────────────

    @retry(max_attempts=3)
    async def get_account(self) -> AccountSummary:
        """Fetch account balance, NAV, margin, and trade count."""
        request = accounts.AccountSummary(self._cfg.account_id)
        data = await asyncio.to_thread(self._client.request, request)
        acc = data["account"]
        return AccountSummary(
            account_id=acc["id"],
            balance=float(acc["balance"]),
            nav=float(acc["NAV"]),
            unrealized_pnl=float(acc["unrealizedPL"]),
            realized_pnl=float(acc.get("pl", 0)),
            margin_used=float(acc["marginUsed"]),
            margin_available=float(acc["marginAvailable"]),
            open_trade_count=int(acc["openTradeCount"]),
            currency=acc["currency"],
        )

    # ── Pricing ────────────────────────────────────────────────────────────────

    @retry(max_attempts=3)
    async def get_price(self, instrument: str) -> LivePrice:
        """Get current bid/ask for a single instrument."""
        request = pricing.PricingInfo(
            accountID=self._cfg.account_id,
            params={"instruments": instrument},
        )
        data = await asyncio.to_thread(self._client.request, request)
        p = data["prices"][0]
        bid = float(p["bids"][0]["price"])
        ask = float(p["asks"][0]["price"])
        return LivePrice(
            instrument=instrument,
            bid=bid,
            ask=ask,
            mid=(bid + ask) / 2,
            tradeable=p.get("tradeable", True),
            timestamp=datetime.now(timezone.utc),
        )

    async def get_prices(self, instruments_list: list[str]) -> dict[str, LivePrice]:
        """Fetch current prices for multiple instruments concurrently."""
        tasks = [self.get_price(inst) for inst in instruments_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return {
            inst: res
            for inst, res in zip(instruments_list, results)
            if isinstance(res, LivePrice)
        }

    # ── Historical Candles ────────────────────────────────────────────────────

    @retry(max_attempts=3)
    async def get_candles(
        self,
        instrument: str,
        granularity: str = "M5",
        count: int = 500,
        from_dt: datetime | None = None,
        to_dt: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candle data as a pandas DataFrame.

        Args:
            instrument: e.g. "EUR_USD"
            granularity: "M1", "M5", "M15", "H1", "H4", "D"
            count: Number of candles (max 5000 per request)
            from_dt: Start datetime (UTC). If set, ignores count.
            to_dt: End datetime (UTC). Used with from_dt.

        Returns:
            DataFrame with DatetimeIndex (UTC) and columns:
            open, high, low, close, volume
        """
        gran = GRANULARITY_MAP.get(granularity, granularity)
        params: dict[str, Any] = {
            "granularity": gran,
            "price": "M",   # Mid prices
        }

        if from_dt and to_dt:
            # Paginate: OANDA max is 5000 candles per request.
            # For long date ranges (e.g. full year H1 = ~6500 candles) we split
            # into multiple requests and concatenate.
            from datetime import timedelta
            PAGE_SIZE = 4500  # stay safely under 5000 limit
            all_rows: list[dict] = []
            cursor = from_dt
            while cursor < to_dt:
                page_params = {
                    "granularity": gran,
                    "price": "M",
                    "from": cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "count": PAGE_SIZE,
                }
                req = instruments.InstrumentsCandles(instrument, params=page_params)
                page_data = await asyncio.to_thread(self._client.request, req)
                page_candles = page_data.get("candles", [])
                if not page_candles:
                    break
                for candle in page_candles:
                    ts = pd.Timestamp(candle["time"]).tz_convert("UTC")
                    if ts >= to_dt:
                        break
                    if not candle.get("complete", True):
                        continue
                    mid = candle["mid"]
                    all_rows.append({
                        "time": ts,
                        "open": float(mid["o"]),
                        "high": float(mid["h"]),
                        "low": float(mid["l"]),
                        "close": float(mid["c"]),
                        "volume": int(candle.get("volume", 0)),
                    })
                # Advance cursor to last candle time + 1 second
                last_ts = pd.Timestamp(page_candles[-1]["time"]).tz_convert("UTC")
                if last_ts <= cursor:
                    break  # No progress — stop
                cursor = last_ts.to_pydatetime() + timedelta(seconds=1)
                if len(page_candles) < PAGE_SIZE:
                    break  # Last page
                await asyncio.sleep(0.3)  # Respect rate limits between pages

            rows = all_rows
        else:
            params["count"] = min(count, 5000)
            request = instruments.InstrumentsCandles(instrument, params=params)
            data = await asyncio.to_thread(self._client.request, request)
            rows = []
            for candle in data.get("candles", []):
                if not candle.get("complete", True):
                    continue
                mid = candle["mid"]
                rows.append({
                    "time": pd.Timestamp(candle["time"]).tz_convert("UTC"),
                    "open": float(mid["o"]),
                    "high": float(mid["h"]),
                    "low": float(mid["l"]),
                    "close": float(mid["c"]),
                    "volume": int(candle.get("volume", 0)),
                })

        if not rows:
            log.warning(f"No candles returned for {instrument} {granularity}")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(rows).set_index("time").sort_index()
        # Deduplicate in case pages overlapped
        df = df[~df.index.duplicated(keep="last")]
        log.debug(f"Fetched {len(df)} {granularity} candles for {instrument}")
        return df

    async def get_multi_candles(
        self,
        instruments_list: list[str],
        granularity: str = "M5",
        count: int = 500,
    ) -> dict[str, pd.DataFrame]:
        """Fetch candles for multiple instruments concurrently."""
        tasks = {
            inst: self.get_candles(inst, granularity, count)
            for inst in instruments_list
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        return {
            inst: res
            for inst, res in zip(tasks.keys(), results)
            if isinstance(res, pd.DataFrame) and not res.empty
        }

    # ── Order Management ──────────────────────────────────────────────────────

    async def market_order(
        self,
        instrument: str,
        units: int,          # Positive = BUY, Negative = SELL
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> OrderResult:
        """
        Place a market order with optional stop-loss and take-profit.

        In paper trading mode, still uses the practice API — real fills
        at real spreads, just no real money.
        """
        order_data: dict[str, Any] = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(units),
                "timeInForce": "FOK",   # Fill Or Kill
            }
        }

        # Decimal places per instrument type: JPY=3, metals (XAU/XAG)=2, others=5
        price_decimals = 3 if "JPY" in instrument else (2 if instrument.startswith(("XAU", "XAG")) else 5)
        if stop_loss is not None:
            order_data["order"]["stopLossOnFill"] = {
                "price": f"{stop_loss:.{price_decimals}f}",
                "timeInForce": "GTC",
            }
        if take_profit is not None:
            order_data["order"]["takeProfitOnFill"] = {
                "price": f"{take_profit:.{price_decimals}f}",
                "timeInForce": "GTC",
            }

        try:
            request = orders.OrderCreate(self._cfg.account_id, data=order_data)
            data = await asyncio.to_thread(self._client.request, request)

            fill = data.get("orderFillTransaction", {})
            trade_id = fill.get("tradeOpened", {}).get("tradeID", "")
            filled_price = float(fill.get("price", 0))

            # Fallback: if OANDA omitted tradeOpened (e.g. adding to existing
            # position), scan relatedTransactionIDs for the trade ID and if
            # still missing, fetch open trades and pick the newest for this
            # instrument.
            if not trade_id:
                # relatedTransactionIDs sometimes contains the trade ID directly
                related = data.get("relatedTransactionIDs", [])
                if related:
                    trade_id = related[-1]
                    log.debug(f"trade_id from relatedTransactionIDs: {trade_id}")

            if not trade_id:
                try:
                    req2 = trades_ep.TradesList(
                        self._cfg.account_id,
                        params={"state": "OPEN", "instrument": instrument},
                    )
                    resp2 = await asyncio.to_thread(self._client.request, req2)
                    open_list = resp2.get("trades", [])
                    if open_list:
                        # Most recently opened trade is first
                        trade_id = str(open_list[0].get("id", ""))
                        if not filled_price:
                            filled_price = float(open_list[0].get("price", 0))
                        log.info(f"trade_id resolved via open trades fallback: {trade_id}")
                except Exception as fe:
                    log.warning(f"trade_id fallback fetch failed: {fe}")

            log.debug(f"market_order response keys: {list(data.keys())}; fill keys: {list(fill.keys())}")
            return OrderResult(
                success=True,
                trade_id=trade_id,
                order_id=fill.get("orderID", ""),
                filled_price=filled_price,
                units=abs(units),
            )
        except V20Error as e:
            log.error(f"OANDA order failed: {e}")
            return OrderResult(success=False, message=str(e))

    @retry(max_attempts=3)
    async def close_trade(self, trade_id: str) -> OrderResult:
        """Close an open trade by its OANDA trade ID."""
        try:
            request = trades_ep.TradeClose(self._cfg.account_id, tradeID=trade_id)
            data = await asyncio.to_thread(self._client.request, request)
            fill = data.get("orderFillTransaction", {})
            return OrderResult(
                success=True,
                trade_id=trade_id,
                filled_price=float(fill.get("price", 0)),
            )
        except V20Error as e:
            log.error(f"OANDA close_trade failed: {e}")
            return OrderResult(success=False, message=str(e))

    @retry(max_attempts=3)
    async def partial_close_trade(self, trade_id: str, units: int) -> OrderResult:
        """
        Partially close a trade by closing `units` of its position.
        units must be positive; the direction is inferred from the open trade.
        """
        try:
            request = trades_ep.TradeClose(
                self._cfg.account_id,
                tradeID=trade_id,
                data={"units": str(abs(units))},
            )
            data = await asyncio.to_thread(self._client.request, request)
            fill = data.get("orderFillTransaction", {})
            return OrderResult(
                success=True,
                trade_id=trade_id,
                filled_price=float(fill.get("price", 0)),
                units=abs(units),
            )
        except V20Error as e:
            log.error(f"partial_close_trade failed: {e}")
            return OrderResult(success=False, message=str(e))

    @retry(max_attempts=3)
    async def modify_trade_sl(self, trade_id: str, new_sl: float, instrument: str = "") -> bool:
        """Move the stop-loss of an open trade to new_sl."""
        try:
            price_decimals = 3 if "JPY" in instrument else (2 if instrument.startswith(("XAU", "XAG")) else 5)
            data = {
                "stopLoss": {
                    "price": f"{new_sl:.{price_decimals}f}",
                    "timeInForce": "GTC",
                }
            }
            request = trades_ep.TradeCRCDO(
                self._cfg.account_id,
                tradeID=trade_id,
                data=data,
            )
            await asyncio.to_thread(self._client.request, request)
            log.info(f"Trade {trade_id} SL moved to {new_sl:.{price_decimals}f}")
            return True
        except V20Error as e:
            log.error(f"modify_trade_sl failed: {e}")
            return False

    async def modify_trade_tp(self, trade_id: str, new_tp: float, instrument: str = "") -> bool:
        """Move the take-profit of an open trade to new_tp."""
        try:
            price_decimals = 3 if "JPY" in instrument else (2 if instrument.startswith(("XAU", "XAG")) else 5)
            data = {
                "takeProfit": {
                    "price": f"{new_tp:.{price_decimals}f}",
                    "timeInForce": "GTC",
                }
            }
            request = trades_ep.TradeCRCDO(
                self._cfg.account_id,
                tradeID=trade_id,
                data=data,
            )
            await asyncio.to_thread(self._client.request, request)
            log.info(f"Trade {trade_id} TP moved to {new_tp:.{price_decimals}f}")
            return True
        except V20Error as e:
            log.error(f"modify_trade_tp failed: {e}")
            return False

    @retry(max_attempts=3)
    async def get_open_positions(self) -> list[dict[str, Any]]:
        """Return a list of currently open positions."""
        request = positions.PositionList(self._cfg.account_id)
        data = await asyncio.to_thread(self._client.request, request)
        return [p for p in data.get("positions", []) if
                float(p.get("long", {}).get("units", 0)) != 0
                or float(p.get("short", {}).get("units", 0)) != 0]

    @retry(max_attempts=3)
    async def get_open_trades(self) -> list[dict[str, Any]]:
        """Return open trades with individual trade IDs (required for closing)."""
        request = trades_ep.TradesList(self._cfg.account_id, params={"state": "OPEN"})
        data = await asyncio.to_thread(self._client.request, request)
        return data.get("trades", [])

    @retry(max_attempts=3)
    async def get_recently_closed_trades(self, count: int = 50) -> list[dict[str, Any]]:
        """Return the most recent closed trades (SL/TP hits and manual closes)."""
        request = trades_ep.TradesList(
            self._cfg.account_id,
            params={"state": "CLOSED", "count": str(count)},
        )
        data = await asyncio.to_thread(self._client.request, request)
        return data.get("trades", [])

    async def get_retail_sentiment(self, instrument: str) -> dict[str, Any]:
        """
        Fetch OANDA order-book snapshot for an instrument.

        Returns a dict with:
          long_pct   — % of open client positions that are long  (0–100)
          short_pct  — % of open client positions that are short (0–100)
          bias       — "long_crowded" | "short_crowded" | "neutral"
          contrarian — suggested contrarian direction ("SELL" | "BUY" | "HOLD")

        Interpretation (contra-indicator): when ≥70% of retail is long, the
        "dumb money" is max-long and institutional flow is usually the other way.
        Use to BLOCK trades that align with extreme retail positioning.

        Returns empty dict on any error (non-fatal — caller must handle gracefully).
        """
        try:
            import oandapyV20.endpoints.instruments as _instruments
            params = {"period": str(3600)}   # 1-hour snapshot
            req = _instruments.InstrumentsOrderBook(instrument, params=params)
            data = await asyncio.to_thread(self._client.request, req)
            buckets = data.get("orderBook", {}).get("buckets", [])
            if not buckets:
                return {}

            long_units  = sum(float(b.get("longCountPercent",  0)) for b in buckets)
            short_units = sum(float(b.get("shortCountPercent", 0)) for b in buckets)
            total = long_units + short_units
            if total <= 0:
                return {}

            long_pct  = round(long_units  / total * 100, 1)
            short_pct = round(short_units / total * 100, 1)

            CROWD_THRESHOLD = 70.0
            if long_pct >= CROWD_THRESHOLD:
                bias, contrarian = "long_crowded", "SELL"
            elif short_pct >= CROWD_THRESHOLD:
                bias, contrarian = "short_crowded", "BUY"
            else:
                bias, contrarian = "neutral", "HOLD"

            return {
                "long_pct":   long_pct,
                "short_pct":  short_pct,
                "bias":       bias,
                "contrarian": contrarian,
            }
        except Exception as e:
            log.debug(f"Retail sentiment fetch failed for {instrument}: {e}")
            return {}


# ── Singleton factory ─────────────────────────────────────────────────────────

_client_instance: OandaClient | None = None


def get_oanda_client() -> OandaClient:
    """
    Return the module-level singleton OandaClient.
    Raises ValueError if OANDA is not configured.
    """
    global _client_instance
    if _client_instance is None:
        _client_instance = OandaClient()
    return _client_instance
