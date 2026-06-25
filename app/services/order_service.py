# app/services/order_service.py
"""
Order Business Logic Service.

Implements the business logic layer for order queries, sitting between
the API endpoints (controllers) and the data access layer (repository).

Responsibilities:
    - Validate business rules (e.g., days parameter range)
    - Coordinate repository calls
    - Apply in-memory caching for expensive aggregations
    - Map ORM models to response schemas

Caching strategy:
    - /orders/stats is cached with a TTL of CACHE_TTL_SECONDS (default 60s)
    - Cache is invalidated when the ETL pipeline runs
    - Uses cachetools.TTLCache for thread-safe in-memory caching

Usage:
  service = OrderService(repository)
  orders = await service.get_customer_orders("C001")
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

from cachetools import TTLCache

from app.core.config import settings
from app.core.exceptions import ValidationError
from app.core.logging import get_logger
from app.repositories.order_repository import OrderRepository
from app.schemas.order import OrderListResponse, OrderResponse, OrderStatsResponse

logger = get_logger(__name__)

# — In-memory Cache —————————————————————————————————————————————————

# TTLCache for /orders/stats - evicts entries after CACHE_TTL_SECONDS
_stats_cache: TTLCache = TTLCache(
    maxsize=settings.CACHE_MAX_SIZE,
    ttl=settings.CACHE_TTL_SECONDS,
)
_STATS_CACHE_KEY = "global_stats"


class OrderService:
    """
    Business logic service for order domain operations.

    Args:
        repository: OrderRepository instance for data access.
    """

    def __init__(self, repository: OrderRepository) -> None:
        self._repo = repository

    # — Customer Orders ———————————————————————————————————————————————

    async def get_customer_orders(self, customer_id: str) -> OrderListResponse:
        """
        Retrieve all orders for a given customer.

        Validates the customer_id format before querying the database.

        Args:
            customer_id: Alphanumeric customer identifier (e.g. "C001").

        Returns:
            OrderListResponse with the list of orders and total count.

        Raises:
            ValidationError: If customer_id is empty or exceeds 50 characters.
            CustomerNotFoundError: If no orders exist for the customer.
        """
        # Business rule: customer_id must be non-empty and ≤ 50 chars
        customer_id = customer_id.strip()
        if not customer_id:
            raise ValidationError(
                message="customer_id must not be empty.",
                details={"field": "customer_id"},
            )
        if len(customer_id) > 50:
            raise ValidationError(
                message="customer_id must not exceed 50 characters.",
                details={"field": "customer_id", "max_length": 50},
            )

        orders = await self._repo.get_by_customer(customer_id)

        order_responses = [
            OrderResponse.model_validate(order) for order in orders
        ]

        logger.info(
            "customer_orders_retrieved",
            customer_id=customer_id,
            count=len(order_responses),
        )

        return OrderListResponse(orders=order_responses, count=len(order_responses))

    # — Statistics ————————————————————————————————————————————————————

    async def get_stats(self) -> OrderStatsResponse:
        """
        Compute and return aggregated order statistics.

        Results are cached in memory for CACHE_TTL_SECONDS to avoid
        repeated expensive aggregation queries.

        Returns:
            OrderStatsResponse with revenue, averages, and per-day counts.

        Raises:
            DatabaseError: On unexpected database failure.
        """
        # Check cache first
        cached = _stats_cache.get(_STATS_CACHE_KEY)
        if cached is not None:
            logger.debug("stats_cache_hit")
            return cached

        logger.debug("stats_cache_miss")

        stats_dict = await self._repo.get_stats()

        response = OrderStatsResponse(
            total_revenue=round(stats_dict["total_revenue"], 2),
            avg_order_value=round(stats_dict["avg_order_value"], 2),
            order_count=stats_dict["order_count"],
            orders_per_day=stats_dict["orders_per_day"],
            currency_breakdown=stats_dict["currency_breakdown"],
            computed_at=datetime.now(timezone.utc),
        )

        # Store in cache
        _stats_cache[_STATS_CACHE_KEY] = response

        logger.info(
            "stats_computed",
            total_revenue=response.total_revenue,
            order_count=response.order_count,
        )

        return response

    # — Recent Orders —————————————————————————————————————————————————

    async def get_recent_orders(self, days: int) -> OrderListResponse:
        """
        Retrieve all orders placed within the last N days.

        Args:
            days: Number of days to look back (1-3650).

        Returns:
            OrderListResponse with matching orders and count.

        Raises:
            ValidationError: If days is outside the allowed range.
            DatabaseError: On unexpected database failure.
        """
        if days < 1:
            raise ValidationError(
                message="days must be at least 1.",
                details={"field": "days", "min_value": 1},
            )
        if days > 3650:
            raise ValidationError(
                message="days must not exceed 3650 (10 years).",
                details={"field": "days", "max_value": 3650},
            )

        orders = await self._repo.get_recent(days)

        order_responses = [
            OrderResponse.model_validate(order) for order in orders
        ]

        logger.info(
            "recent_orders_retrieved",
            days=days,
            count=len(order_responses),
        )

        return OrderListResponse(orders=order_responses, count=len(order_responses))

    # — Cache Management ————————————————————————————————————————————————

    def invalidate_stats_cache(self) -> None:
        """
        Invalidate the statistics cache.

        Called by the ETL pipeline after a successful data load to ensure
        the next /orders/stats request reflects the updated data.
        """
        _stats_cache.pop(_STATS_CACHE_KEY, None)
        logger.info("stats_cache_invalidated")