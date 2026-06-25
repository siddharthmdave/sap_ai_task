# app/core/database.py
"""
Database Engine and Session Management Module.

Provides async SQLAlchemy engine, session factory, and dependency injection
helpers for FastAPI. Supports SQLite (default) and can be extended to
PostgreSQL/MySQL by changing DATABASE_URL.

Features:
    - Async engine with connection pooling
    - Per-request session lifecycle management
    - Automatic table creation on startup
    - Health-check utility

Usage:
    # In FastAPI endpoint (dependency injection):
    async def endpoint(db: AsyncSession = Depends(get_db)):
        result = await db.execute(select(Order))
   ...
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# — ORM Base ——————————————————————————————————————

class Base(DeclarativeBase):
    """
    Declarative base class for all SQLAlchemy ORM models.

    All model classes must inherit from this Base to be included in
    metadata.create_all() calls during startup.
    """
    pass

# — Engine Factory ——————————————————————————————————————

def _build_engine() -> AsyncEngine:
    """
    Build and configure the async SQLAlchemy engine.

    For SQLite: enables WAL mode and foreign key enforcement via event hooks.
    For other databases: connection pool settings are applied.

    Returns:
        Configured AsyncEngine instance.
    """
    connect_args: dict = {}
    engine_kwargs: dict = {
        "echo": settings.DATABASE_ECHO,
        "future": True,
    }

    if "sqlite" in settings.DATABASE_URL:
        # SQLite requires check_same_thread=False for async usage
        connect_args["check_same_thread"] = False
        # SQLite does not support pool_size/max_overflow
    else:
        engine_kwargs["pool_size"] = 10
        engine_kwargs["max_overflow"] = 5
        engine_kwargs["pool_pre_ping"] = True
        engine_kwargs["pool_recycle"] = 3600

    engine = create_async_engine(
        settings.DATABASE_URL,
        connect_args=connect_args,
        **engine_kwargs,
    )

    # Enable WAL mode and foreign keys for SQLite
    if "sqlite" in settings.DATABASE_URL:
        @event.listens_for(engine.sync_engine, "connect") # type: ignore[misc]
        def set_sqlite_pragma(dbapi_conn, connection_record): # noqa: ARG001
            """Apply SQLite performance and integrity pragmas on each new connection."""
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.close()

    return engine

# — Module-level singletons ——————————————————————————————————————

engine: AsyncEngine = _build_engine()

AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# — Lifecycle Helpers ——————————————————————————————————————

async def create_all_tables() -> None:
    """
    Create all ORM-mapped tables in the database if they do not exist.

    Called during application startup. Safe to call multiple times (idempotent).
    Imports all models to ensure they are registered with Base.metadata.
    """
    # Import models to register them with Base.metadata before create_all
    import app.models.order # noqa: F401
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    logger.info("database_tables_created", url=settings.DATABASE_URL)

async def dispose_engine() -> None:
    """
    Dispose the database engine connection pool.

    Called during application shutdown to cleanly release all connections.
    """
    await engine.dispose()
    logger.info("database_engine_disposed")

# — Dependency Injection ——————————————————————————————————————

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides a per-request async database session.

    Automatically commits on success and rolls back on exception.
    The session is closed after the request completes.

    Yields:
        AsyncSession: Active database session for the current request.

    Example:
        @router.get("/orders")
        async def list_orders(db: AsyncSession = Depends(get_db)):
           ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for database sessions outside of FastAPI requests.

    Used in ETL scripts, background tasks, and CLI commands where FastAPI
    dependency injection is not available.

    Example:
        async with get_db_context() as db:
            await db.execute(insert(Order).values(...))
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# — Health Check ——————————————————————————————————————

async def check_database_health() -> bool:
    """
    Verify database connectivity by executing a lightweight query.

    Returns:
        True if the database is reachable, False otherwise.
    """
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error("database_health_check_failed", error=str(exc))
        return False