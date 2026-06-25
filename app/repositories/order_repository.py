# app/repositories/order_repository.py
"""
Order Repository - Data Access Layer.

Encapsulates all database queries for the orders domain.
The repository pattern separates SQL/ORM logic from business logic,
making the service layer testable with mock repositories.

All methods are async and accept an AsyncSession injected by FastAPI's
dependency injection system.

Usage:
    repo = OrderRepository(db)
    orders = await repo.get_by_customer("C001")
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from sqlalchemy import func, select, text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import CustomerNotFoundError, DatabaseError
from app.core.logging import get_logger
from app.models.order import Order

logger = get_logger(__name__)


class OrderRepository:
    """
    Repository for all Order data access operations.

    Provides CRUD and query methods that abstract SQLAlchemy ORM details
    from the service layer. All methods raise domain exceptions on failure.

    Args:
        db: Active async SQLAlchemy session (injected via FastAPI Depends).
    """

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    # — Read Operations ——————————————————————————————————————

    async def get_by_customer(self, customer_id: str) -> List[Order]:
        """
        Retrieve all orders for a given customer, ordered by date descending.

        Args:
            customer_id: The alphanumeric customer identifier.

        Returns:
            List of Order ORM objects. Empty list if customer has no orders.

        Raises:
            CustomerNotFoundError: If no orders exist for the customer.
            DatabaseError: On unexpected database failure.
        """
        try:
            stmt = (
                select(Order)
               .where(Order.customer_id == customer_id)
               .order_by(Order.order_date.desc())
            )
            result = await self._db.execute(stmt)
            orders = list(result.scalars().all())

            if not orders:
                raise CustomerNotFoundError(customer_id=customer_id)

            logger.debug(
                "orders_fetched_by_customer",
                customer_id=customer_id,
                count=len(orders),
            )
            return orders

        except CustomerNotFoundError:
            raise
        except Exception as exc:
            logger.error("db_get_by_customer_failed", customer_id=customer_id, error=str(exc))
            raise DatabaseError(f"Failed to fetch orders for customer '{customer_id}': {exc}") from exc

    async def get_recent(self, days: int) -> List[Order]:
        """
        Retrieve all orders placed within the last N days.

        Args:
            days: Number of days to look back from today (inclusive).

        Returns:
            List of Order ORM objects ordered by date descending.

        Raises:
            DatabaseError: On unexpected database failure.
        """
        try:
            cutoff_date = date.today() - timedelta(days=days)
            stmt = (
                select(Order)
              .where(Order.order_date >= cutoff_date)
              .order_by(Order.order_date.desc())
            )
            result = await self._db.execute(stmt)
            orders = list(result.scalars().all())

            logger.debug(
                "recent_orders_fetched",
                days=days,
                cutoff_date=str(cutoff_date),
                count=len(orders),
            )
            return orders

        except Exception as exc:
            logger.error("db_get_recent_failed", days=days, error=str(exc))
            raise DatabaseError(f"Failed to fetch recent orders: {exc}") from exc

    async def get_stats(self) -> Dict:
        """
        Compute aggregated order statistics in a single database query.

        Returns a dict with:
            - total_revenue: Float
            - avg_order_value: Float
            - order_count: int
            - orders_per_day: Dict[str, int]
            - currency_breakdown: Dict[str, int]

        Raises:
            DatabaseError: On unexpected database failure.
        """
        try:
            # Aggregate query: total revenue, avg, count
            agg_stmt = select(
                func.coalesce(func.sum(Order.amount_usd), 0.0).label("total_revenue"),
                func.coalesce(func.avg(Order.amount_usd), 0.0).label("avg_order_value"),
                func.count(Order.order_id).label("order_count"),
            )
            agg_result = await self._db.execute(agg_stmt)
            agg_row = agg_result.one()

            # Per-day order counts
            day_stmt = select(
                Order.order_date.label("day"),
                func.count(Order.order_id).label("cnt"),
            ).group_by(Order.order_date).order_by(Order.order_date)
            day_result = await self._db.execute(day_stmt)
            orders_per_day: Dict[str, int] = {
                str(row.day): row.cnt for row in day_result.all()
            }

            # Currency breakdown
            currency_stmt = select(
                Order.currency.label("currency"),
                func.count(Order.order_id).label("cnt"),
            ).group_by(Order.currency)
            currency_result = await self._db.execute(currency_stmt)
            currency_breakdown: Dict[str, int] = {
                row.currency: row.cnt for row in currency_result.all()
            }

            return {
                "total_revenue": float(agg_row.total_revenue),
                "avg_order_value": float(agg_row.avg_order_value),
                "order_count": int(agg_row.order_count),
                "orders_per_day": orders_per_day,
                "currency_breakdown": currency_breakdown,
            }

        except Exception as exc:
            logger.error("db_get_stats_failed", error=str(exc))
            raise DatabaseError(f"Failed to compute order statistics: {exc}") from exc

    async def get_all(self) -> List[Order]:
        """
        Retrieve all orders from the database.

        Used by the embedding service to build the FAISS index on startup.

        Returns:
            List of all Order ORM objects.

        Raises:
            DatabaseError: On unexpected database failure.
        """
        try:
            stmt = select(Order).order_by(Order.order_date.desc())
            result = await self._db.execute(stmt)
            orders = list(result.scalars().all())
            logger.debug("all_orders_fetched", count=len(orders))
            return orders
        except Exception as exc:
            logger.error("db_get_all_failed", error=str(exc))
            raise DatabaseError(f"Failed to fetch all orders: {exc}") from exc

    async def get_by_ids(self, order_ids: List[str]) -> List[Order]:
        """
        Retrieve orders by a list of order IDs.

        Used by the semantic search service to hydrate FAISS search results.

        Args:
            order_ids: List of order_id strings to fetch.

        Returns:
            List of matching Order ORM objects (may be shorter than order_ids
            if some IDs are not found).

        Raises:
            DatabaseError: On unexpected database failure.
        """
        try:
            if not order_ids:
                return []
            stmt = select(Order).where(Order.order_id.in_(order_ids))
            result = await self._db.execute(stmt)
            return list(result.scalars().all())
        except Exception as exc:
            logger.error("db_get_by_ids_failed", order_ids=order_ids, error=str(exc))
            raise DatabaseError(f"Failed to fetch orders by IDs: {exc}") from exc

    async def execute_raw_sql(self, sql: str) -> Tuple[List[Dict], List[str]]:
        """
        Execute a raw SQL SELECT statement and return rows as dicts.

        Used exclusively by the NL2SQL query endpoint. Only SELECT statements
        are permitted; any other statement type raises a ValueError.

        Args:
            sql: Raw SQL SELECT statement generated by the LLM.

        Returns:
            Tuple of (rows as list of dicts, column names list).

        Raises:
            ValueError: If the SQL is not a SELECT statement.
            DatabaseError: On SQL execution failure.
        """
        # Security: only allow SELECT statements
        normalized = sql.strip().upper()
        if not normalized.startswith("SELECT"):
            raise ValueError(
                f"Only SELECT statements are permitted. Got: {sql[:50]}..."
            )

        try:
            result = await self._db.execute(text(sql))
            columns = list(result.keys())
            rows = [dict(zip(columns, row)) for row in result.fetchall()]
            return rows, columns
        except Exception as exc:
            raise DatabaseError(f"SQL execution failed: {exc}") from exc

    # — Write Operations ——————————————————————————————————————

    async def upsert_batch(self, orders: List[Dict]) -> Tuple[int, int]:
        """
        Upsert a batch of order records using INSERT OR REPLACE semantics.

        For SQLite: uses INSERT OR REPLACE (via sqlite_insert with on_conflict_do_update).
        For other databases: falls back to individual upserts.

        Args:
            orders: List of dicts with keys matching Order column names.

        Returns:
            Tuple of (inserted_count, updated_count).

        Raises:
            DatabaseError: On batch write failure.
        """
        if not orders:
            return 0, 0

        try:
            # Use SQLite's INSERT OR REPLACE for efficient upsert
            stmt = sqlite_insert(Order).values(orders)
            stmt = stmt.on_conflict_do_update(
                index_elements=["order_id"],
                set_={
                    "customer_id": stmt.excluded.customer_id,
                    "order_date": stmt.excluded.order_date,
                    "amount_usd": stmt.excluded.amount_usd,
                    "original_amount": stmt.excluded.original_amount,
                    "currency": stmt.excluded.currency,
                    "updated_at": func.now(),
                },
            )
            await self._db.execute(stmt)
            await self._db.flush()

            logger.debug("batch_upserted", count=len(orders))
            # SQLite doesn't easily distinguish inserts vs updates in bulk;
            # return total as inserted for simplicity
            return len(orders), 0

        except Exception as exc:
            logger.error("db_upsert_batch_failed", count=len(orders), error=str(exc))
            raise DatabaseError(f"Batch upsert failed: {exc}") from exc

    async def count(self) -> int:
        """
        Return the total number of orders in the database.

        Returns:
            Integer count of all order records.
        """
        try:
            stmt = select(func.count(Order.order_id))
            result = await self._db.execute(stmt)
            return result.scalar_one()
        except Exception as exc:
            raise DatabaseError(f"Failed to count orders: {exc}") from exc