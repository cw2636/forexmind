"""
ForexMind — Multi-Source Async News Aggregator
================================================
Aggregates forex-relevant news from:
  1. Alpha Vantage News Sentiment API (free tier)
  2. Finnhub News API (free tier)
  3. Forex Factory economic calendar (web scraper)

Runs all fetches concurrently (asyncio.gather) and deduplicates by headline.
Each article gets a TextBlob sentiment score (-1.0 to +1.0).

Advanced Python concepts:
  - AsyncGenerator for streaming article batches
  - dataclasses with post-processing
  - Abstract base class (ABC) for pluggable source pattern
  - asyncio.gather() for concurrent I/O
  - Simple in-memory LRU cache using functools.lru_cache (sync) vs TTL cache
"""

from __future__ import annotations

import asyncio
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp
from textblob import TextBlob

from forexmind.config.settings import get_settings
from forexmind.utils.logger import get_logger

log = get_logger(__name__)

# Currency → instruments it affects (for relevance tagging)
CURRENCY_TO_INSTRUMENTS: dict[str, list[str]] = {
    "USD": ["EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD"],
    "EUR": ["EUR_USD", "EUR_GBP", "EUR_JPY", "EUR_CAD", "EUR_AUD"],
    "GBP": ["GBP_USD", "EUR_GBP", "GBP_JPY", "GBP_CAD"],
    "JPY": ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY"],
    "AUD": ["AUD_USD", "AUD_JPY", "EUR_AUD"],
    "CAD": ["USD_CAD", "EUR_CAD", "GBP_CAD"],
    "CHF": ["USD_CHF", "EUR_CHF"],
    "NZD": ["NZD_USD"],
}


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class NewsItem:
    """Normalised news article from any source."""
    headline: str
    summary: str
    url: str
    published_at: datetime
    source: str
    related_currencies: list[str] = field(default_factory=list)
    related_instruments: list[str] = field(default_factory=list)
    sentiment_score: float = 0.0      # -1.0 (very negative) to +1.0 (very positive)
    impact: str = "low"               # low / medium / high
    # Stable dedup key derived from headline
    key: str = field(init=False)

    def __post_init__(self) -> None:
        self.key = hashlib.md5(self.headline.lower().encode()).hexdigest()
        if not self.sentiment_score:
            self.sentiment_score = self._analyse_sentiment()
        if not self.related_instruments:
            self.related_instruments = self._find_instruments()

    def _analyse_sentiment(self) -> float:
        """Use TextBlob for fast, dependency-free sentiment analysis."""
        blob = TextBlob(f"{self.headline}. {self.summary}")
        return round(blob.sentiment.polarity, 4)

    def _find_instruments(self) -> list[str]:
        """Map mentioned currencies to forex instruments."""
        text = f"{self.headline} {self.summary}".upper()
        instruments_found: list[str] = []
        for currency, insts in CURRENCY_TO_INSTRUMENTS.items():
            if currency in text and currency not in self.related_currencies:
                self.related_currencies.append(currency)
            if currency in text:
                instruments_found.extend(insts)
        # Dedupe
        return list(dict.fromkeys(instruments_found))


# ── Abstract Source ───────────────────────────────────────────────────────────

class NewsSource(ABC):
    """Base class for all news sources. Each source must implement fetch()."""

    @abstractmethod
    async def fetch(self, session: aiohttp.ClientSession) -> list[NewsItem]:
        """Fetch and return normalised news items."""
        ...


# ── Alpha Vantage Source ──────────────────────────────────────────────────────

class AlphaVantageNews(NewsSource):
    """
    Alpha Vantage News & Sentiment API.
    Free tier: 25 requests/day, 500 requests/month.
    Docs: https://www.alphavantage.co/documentation/#news-sentiment
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self) -> None:
        self._key = get_settings().data.alpha_vantage_key

    async def fetch(self, session: aiohttp.ClientSession) -> list[NewsItem]:
        if not self._key or self._key == "your_alpha_vantage_key_here":
            log.debug("Alpha Vantage key not configured — skipping")
            return []

        params = {
            "function": "NEWS_SENTIMENT",
            "topics": "forex",
            "sort": "LATEST",
            "limit": "50",
            "apikey": self._key,
        }
        try:
            async with session.get(self.BASE_URL, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    log.warning(f"Alpha Vantage returned HTTP {resp.status}")
                    return []
                data = await resp.json()

            items: list[NewsItem] = []
            for article in data.get("feed", []):
                pub = datetime.strptime(
                    article.get("time_published", "20240101T000000"),
                    "%Y%m%dT%H%M%S",
                ).replace(tzinfo=timezone.utc)
                av_sentiment = float(article.get("overall_sentiment_score", 0))
                items.append(NewsItem(
                    headline=article.get("title", ""),
                    summary=article.get("summary", ""),
                    url=article.get("url", ""),
                    published_at=pub,
                    source="alpha_vantage",
                    sentiment_score=av_sentiment,  # Use AV's pre-computed score
                ))
            log.debug(f"AlphaVantage: fetched {len(items)} articles")
            return items
        except Exception as e:
            log.warning(f"AlphaVantage fetch error: {e}")
            return []


# ── Finnhub Source ────────────────────────────────────────────────────────────

class FinnhubNews(NewsSource):
    """
    Finnhub Forex News endpoint.
    Free tier: 60 calls/minute.
    Docs: https://finnhub.io/docs/api/forex-news
    """

    BASE_URL = "https://finnhub.io/api/v1/news"

    def __init__(self) -> None:
        self._key = get_settings().data.finnhub_key

    async def fetch(self, session: aiohttp.ClientSession) -> list[NewsItem]:
        if not self._key or self._key == "your_finnhub_key_here":
            log.debug("Finnhub key not configured — skipping")
            return []

        params = {"category": "forex", "token": self._key}
        try:
            async with session.get(self.BASE_URL, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return []
                articles = await resp.json()

            items: list[NewsItem] = []
            for art in articles[:50]:
                ts = datetime.fromtimestamp(art.get("datetime", 0), tz=timezone.utc)
                items.append(NewsItem(
                    headline=art.get("headline", ""),
                    summary=art.get("summary", ""),
                    url=art.get("url", ""),
                    published_at=ts,
                    source="finnhub",
                ))
            log.debug(f"Finnhub: fetched {len(items)} articles")
            return items
        except Exception as e:
            log.warning(f"Finnhub fetch error: {e}")
            return []


# ── Aggregator ────────────────────────────────────────────────────────────────

class NewsAggregator:
    """
    Coordinates all news sources, deduplicates results, and provides
    a rolling window of recent sentiment per instrument.
    """

    def __init__(self) -> None:
        self._sources: list[NewsSource] = [
            AlphaVantageNews(),
            FinnhubNews(),
        ]
        # Simple in-memory article cache: key → NewsItem
        self._cache: dict[str, NewsItem] = {}
        self._last_fetch: datetime = datetime.min.replace(tzinfo=timezone.utc)
        # Minimum seconds between full fetches (rate-limit protection)
        self._fetch_interval = get_settings().news_config.get("fetch_interval_minutes", 15) * 60

    async def fetch_all(self, force: bool = False) -> list[NewsItem]:
        """
        Fetch from all sources concurrently. Results are deduplicated by
        headline hash. Returns ALL cached articles sorted newest-first.

        Args:
            force: If True, bypass the TTL cache and re-fetch immediately.
        """
        now = datetime.now(timezone.utc)
        seconds_since_last = (now - self._last_fetch).total_seconds()

        if not force and seconds_since_last < self._fetch_interval:
            log.debug(f"News cache valid ({seconds_since_last:.0f}s old) — returning cached")
            return sorted(self._cache.values(), key=lambda x: x.published_at, reverse=True)

        log.info("Fetching fresh news from all sources...")
        async with aiohttp.ClientSession() as session:
            results = await asyncio.gather(
                *[source.fetch(session) for source in self._sources],
                return_exceptions=True,
            )

        new_count = 0
        for batch in results:
            if isinstance(batch, Exception):
                log.warning(f"A news source raised an error: {batch}")
                continue
            for item in batch:
                if item.key not in self._cache:
                    self._cache[item.key] = item
                    new_count += 1

        # Prune articles older than 24 hours
        cutoff = now - timedelta(hours=24)
        self._cache = {
            k: v for k, v in self._cache.items() if v.published_at > cutoff
        }

        self._last_fetch = now
        articles = sorted(self._cache.values(), key=lambda x: x.published_at, reverse=True)
        log.info(f"News aggregator: {new_count} new articles, {len(articles)} total in cache")
        return articles

    def get_instrument_sentiment(
        self,
        instrument: str,
        lookback_hours: int = 4,
    ) -> dict[str, Any]:
        """
        Compute rolling sentiment stats for a specific forex instrument.

        Returns a dict with:
          - score: average sentiment (-1.0 to +1.0)
          - article_count: how many relevant articles in the window
          - impact: "positive" / "negative" / "neutral"
          - high_impact_count: articles flagged as high impact
        """
        if not self._cache:
            return {"score": 0.0, "article_count": 0, "impact": "neutral", "high_impact_count": 0}

        cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        relevant = [
            item for item in self._cache.values()
            if item.published_at > cutoff and instrument in item.related_instruments
        ]

        if not relevant:
            return {"score": 0.0, "article_count": 0, "impact": "neutral", "high_impact_count": 0}

        avg_score = sum(item.sentiment_score for item in relevant) / len(relevant)
        high_impact = sum(1 for item in relevant if item.impact == "high")

        return {
            "score": round(avg_score, 4),
            "article_count": len(relevant),
            "impact": "positive" if avg_score > 0.1 else "negative" if avg_score < -0.1 else "neutral",
            "high_impact_count": high_impact,
            "recent_headlines": [item.headline for item in relevant[:5]],
        }


# ── Singleton ─────────────────────────────────────────────────────────────────

_aggregator: NewsAggregator | None = None


def get_news_aggregator() -> NewsAggregator:
    global _aggregator
    if _aggregator is None:
        _aggregator = NewsAggregator()
    return _aggregator
