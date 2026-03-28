"""
ForexMind — Async Database Manager
=====================================
SQLAlchemy 2.0 async engine + session factory.
Uses SQLite with aiosqlite for zero-config setup.

Advanced Python concepts:
  - async context managers (__aenter__ / __aexit__)
  - AsyncSession with SQLAlchemy 2.0
  - Dependency injection pattern (get_db() for FastAPI)
  - @asynccontextmanager decorator
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from forexmind.config.settings import get_settings
from forexmind.data.models import Base
from forexmind.utils.logger import get_logger

log = get_logger(__name__)

# Module-level engine & session factory (created once, reused)
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _get_engine() -> AsyncEngine:
    """Lazily create the async database engine."""
    global _engine
    if _engine is None:
        cfg = get_settings()
        url = cfg.data.database_url
        _engine = create_async_engine(
            url,
            echo=False,          # Set True to log all SQL (useful for debugging)
            pool_pre_ping=True,  # Test connections before use
        )
        log.debug(f"Database engine created: {url}")
    return _engine


def _get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Lazily create the session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=_get_engine(),
            expire_on_commit=False,   # Don't re-load objects after commit
            class_=AsyncSession,
        )
    return _session_factory


async def init_db() -> None:
    """
    Create all tables defined in models.py if they don't exist.
    Call once at application startup.
    """
    engine = _get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("Database initialised (tables created if missing)")


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager that yields a database session with
    automatic commit on success and rollback on exception.

    Usage:
        async with get_session() as session:
            session.add(candle)
            # commit happens automatically on __aexit__
    """
    factory = _get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency injection style session provider.

    Usage in FastAPI routes:
        @router.get("/signals")
        async def get_signals(db: AsyncSession = Depends(get_db)):
            ...
    """
    factory = _get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def close_db() -> None:
    """Dispose of the engine connection pool. Call at application shutdown."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        log.debug("Database engine disposed")
