# app/schemas/order.py

"""
Order Pydantic Schemas.

Defines all request and response schemas for the orders domain.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Order Read Models
# =============================================================================

class OrderResponse(BaseModel):
    """
    Single order read model.
    """

    model_config = {"from_attributes": True}

    order_id: str = Field(
        description="Unique order identifier"
    )

    customer_id: str = Field(
        description="Customer identifier"
    )

    order_date: date = Field(
        description="Order date"
    )

    amount_usd: float = Field(
        ge=0.0,
        description="Amount converted to USD"
    )

    original_amount: Optional[float] = Field(
        default=None,
        description="Original amount from source"
    )

    currency: str = Field(
        description="Original currency code"
    )

    created_at: Optional[datetime] = Field(
        default=None,
        description="ETL load timestamp"
    )


class OrderListResponse(BaseModel):
    """
    List of orders.
    """

    orders: List[OrderResponse] = Field(
        description="Orders"
    )

    count: int = Field(
        ge=0,
        description="Total records"
    )


# =============================================================================
# Statistics
# =============================================================================

class OrderStatsResponse(BaseModel):
    """
    Aggregated statistics.
    """

    total_revenue: float = Field(
        ge=0.0,
        description="Total revenue in USD"
    )

    avg_order_value: float = Field(
        ge=0.0,
        description="Average order value"
    )

    order_count: int = Field(
        ge=0,
        description="Total order count"
    )

    orders_per_day: Dict[str, int] = Field(
        default_factory=dict,
        description="Orders grouped by day"
    )

    currency_breakdown: Dict[str, int] = Field(
        default_factory=dict,
        description="Orders grouped by currency"
    )

    computed_at: datetime = Field(
        description="Statistics computation timestamp"
    )


# =============================================================================
# Natural Language Query
# =============================================================================

class NLQueryRequest(BaseModel):
    """
    Request body for AI query endpoint.
    """

    question: str = Field(
        min_length=5,
        max_length=500,
        description="Natural language question",
        examples=[
            "What is the total revenue from customer C001 in the last 30 days?"
        ],
    )

    @field_validator("question")
    @classmethod
    def sanitize_question(cls, value: str) -> str:
        """
        Prevent SQL injection and prompt injection attempts.
        """

        value = value.strip()

        sql_patterns = [
            r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|TRUNCATE|EXECUTE)\b",
            r"(--|;|/\*|\*/)",
            r"\bUNION\s+SELECT\b",
            r"\bOR\s+1\s*=\s*1\b",
        ]

        for pattern in sql_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                raise ValueError(
                    "Question contains disallowed SQL patterns."
                )

        prompt_patterns = [
            r"ignore\s+(previous|all|above)\s+instructions",
            r"forget\s+(everything|all|previous)",
            r"you\s+are\s+now\s+a",
            r"act\s+as\s+a[n]?\s+",
            r"system\s+prompt",
            r"jailbreak",
        ]

        for pattern in prompt_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                raise ValueError(
                    "Question contains disallowed prompt injection patterns."
                )

        return value


class NLQueryResponse(BaseModel):
    """
    AI query response.
    """

    answer: str = Field(
        description="Generated answer"
    )

    sql_used: str = Field(
        description="SQL query executed"
    )

    rows: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Raw query results"
    )

    token_count: int = Field(
        ge=0,
        description="LLM tokens consumed"
    )

    retry_count: int = Field(
        ge=0,
        description="Number of retries"
    )

    model_used: str = Field(
        description="LLM model used"
    )


# =============================================================================
# Semantic Search
# =============================================================================

class SemanticSearchResult(BaseModel):
    """
    Individual semantic search hit.
    """

    order_id: str = Field(
        description="Order identifier"
    )

    customer_id: str = Field(
        description="Customer identifier"
    )

    amount_usd: float = Field(
        ge=0.0,
        description="Amount in USD"
    )

    order_date: str = Field(
        description="Order date"
    )

    currency: str = Field(
        description="Currency code"
    )

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Similarity score"
    )


class SemanticSearchResponse(BaseModel):
    """
    Semantic search response.
    """

    results: List[SemanticSearchResult] = Field(
        default_factory=list,
        description="Search results"
    )

    query: str = Field(
        description="Original query"
    )

    top_k: int = Field(
        ge=1,
        description="Requested result count"
    )

    index_size: int = Field(
        ge=0,
        description="Size of embedding index"
    )


# =============================================================================
# ETL Response
# =============================================================================

class ETLRunResponse(BaseModel):
    """
    ETL execution response.
    """

    status: str = Field(
        description="success | partial | failed"
    )

    file_path: str = Field(
        description="Processed CSV file"
    )

    rows_read: int = Field(
        ge=0,
        description="Rows read"
    )

    rows_loaded: int = Field(
        ge=0,
        description="Rows loaded"
    )

    rows_skipped: int = Field(
        ge=0,
        description="Rows skipped"
    )

    rows_updated: int = Field(
        ge=0,
        description="Rows updated"
    )

    duration_seconds: float = Field(
        ge=0.0,
        description="Processing time"
    )

    errors: List[str] = Field(
        default_factory=list,
        description="ETL errors"
    )