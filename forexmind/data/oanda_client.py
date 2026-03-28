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
            params["from"] = from_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            params["to"] = to_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            params["count"] = min(count, 5000)

        request = instruments.InstrumentsCandles(instrument, params=params)
        data = await asyncio.to_thread(self._client.request, request)

        rows = []
        for candle in data.get("candles", []):
            if not candle.get("complete", True):
                continue   # Skip the in-progress (unclosed) candle
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

        if stop_loss is not None:
            order_data["order"]["stopLossOnFill"] = {
                "price": f"{stop_loss:.5f}",
                "timeInForce": "GTC",
            }
        if take_profit is not None:
            order_data["order"]["takeProfitOnFill"] = {
                "price": f"{take_profit:.5f}",
                "timeInForce": "GTC",
            }

        try:
            request = orders.OrderCreate(self._cfg.account_id, data=order_data)
            data = await asyncio.to_thread(self._client.request, request)

            fill = data.get("orderFillTransaction", {})
            return OrderResult(
                success=True,
                trade_id=fill.get("tradeOpened", {}).get("tradeID", ""),
                order_id=fill.get("orderID", ""),
                filled_price=float(fill.get("price", 0)),
                units=abs(units),
            )
        except V20Error as e:
            log.error(f"OANDA order failed: {e}")
            return OrderResult(success=False, message=str(e))

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

    async def get_open_positions(self) -> list[dict[str, Any]]:
        """Return a list of currently open positions."""
        request = positions.PositionList(self._cfg.account_id)
        data = await asyncio.to_thread(self._client.request, request)
        return [p for p in data.get("positions", []) if
                float(p.get("long", {}).get("units", 0)) != 0
                or float(p.get("short", {}).get("units", 0)) != 0]


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
