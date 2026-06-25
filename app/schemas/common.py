# app/schemas/common.py
"""
Common Pydantic Schemas.

Defines reusable base schemas for API responses, pagination, and error
envelopes. All API responses are wrapped in a consistent envelope to
ensure uniform client parsing.

Response Envelope:
    {
        "success": true,
        "data": {... },
        "meta": { "request_id": "...", "timestamp": "..." }
    }

Error Envelope:
    {
        "success": false,
        "error": {
            "error_code": "ORDER_NOT_FOUND",
            "message": "Order '1001' not found.",
            "details": {}
        },
        "meta": { "request_id": "...", "timestamp": "..." }
    }
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field

# Generic type variable for the data payload
T = TypeVar("T")

# — Meta ——————————————————————————————————————

class ResponseMeta(BaseModel):
    """
    Metadata attached to every API response.

    Attributes:
        request_id: Correlation ID for distributed tracing.
        timestamp: ISO 8601 UTC timestamp of the response.
        version: API version string.
    """

    request_id: Optional[str] = Field(default=None, description="Request correlation ID")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response generation timestamp (UTC)",
    )
    version: str = Field(default="v1", description="API version")

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}

# — Success Envelope ——————————————————————————————————————

class SuccessResponse(BaseModel, Generic[T]):
    """
    Generic success response envelope.

    Wraps any data payload with a consistent structure.

    Example:
        SuccessResponse[OrderResponse](data=order, meta=meta)
    """
    success: bool = Field(default=True, description="Always True for success responses")
    data: T = Field(description="Response payload")
    meta: ResponseMeta = Field(default_factory=ResponseMeta, description="Response metadata")

# — Error Envelope ——————————————————————————————————————

class ErrorDetail(BaseModel):
    """
    Structured error detail object.

    Attributes:
        error_code: Machine-readable error identifier.
        message: Human-readable error description.
        details: Optional additional context (field errors, etc.).
    """
    error_code: str = Field(description="Machine-readable error code")
    message: str = Field(description="Human-readable error description")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional error context")

class ErrorResponse(BaseModel):
    """
    Standard error response envelope.

    All error responses from the API use this structure.
    """
    success: bool = Field(default=False, description="Always False for error responses")
    error: ErrorDetail = Field(description="Error details")
    meta: ResponseMeta = Field(default_factory=ResponseMeta, description="Response metadata")

# — Validation Error Detail ——————————————————————————————————————

class FieldValidationError(BaseModel):
    """
    Single field validation error detail.

    Used in 422 Unprocessable Entity responses from Pydantic validation.
    """
    field: str = Field(description="Field path that failed validation")
    message: str = Field(description="Validation error message")
    invalid_value: Optional[Any] = Field(default=None, description="The value that failed validation")

class ValidationErrorResponse(BaseModel):
    """
    Response body for 422 Pydantic validation errors.

    Provides field-level error details for client-side form validation.
    """
    success: bool = Field(default=False)
    error: ErrorDetail = Field(
        default_factory=lambda: ErrorDetail(
            error_code="VALIDATION_ERROR",
            message="Request validation failed.",
        )
    )
    field_errors: List[FieldValidationError] = Field(
        default_factory=list,
        description="Per-field validation errors",
    )
    meta: ResponseMeta = Field(default_factory=ResponseMeta)

# — Health Check ——————————————————————————————————————

class HealthCheckResponse(BaseModel):
    """
    Response schema for the /healthz liveness endpoint.

    Attributes:
        status: "ok" when healthy, "degraded" when some components are unhealthy.
        database: Database connectivity status.
        embedding_index: FAISS index readiness status.
        version: Application version string.
        uptime_seconds: Seconds since application startup.
    """
    status: str = Field(description="Overall health status: ok | degraded | unhealthy")
    database: str = Field(description="Database connectivity: ok | error")
    embedding_index: str = Field(description="FAISS index status: ready | not_ready | error")
    version: str = Field(description="Application version")
    uptime_seconds: float = Field(description="Seconds since application startup")

# — Pagination ——————————————————————————————————————

class PaginationParams(BaseModel):
    """
    Common pagination query parameters.

    Attributes:
        page: 1-based page number.
        page_size: Number of items per page (max 500).
    """
    page: int = Field(default=1, ge=1, description="Page number (1-based)")
    page_size: int = Field(default=50, ge=1, le=500, description="Items per page")

    @property
    def offset(self) -> int:
        """Calculate SQL OFFSET from page and page_size."""
        return (self.page - 1) * self.page_size

class PaginatedResponse(BaseModel, Generic[T]):
    """
    Paginated list response envelope.

    Attributes:
        items: List of items for the current page.
        total: Total number of items across all pages.
        page: Current page number.
        page_size: Items per page.
        total_pages: Total number of pages.
    """
    items: List[T] = Field(description="Items for the current page")
    total: int = Field(description="Total item count")
    page: int = Field(description="Current page number")
    page_size: int = Field(description="Items per page")
    total_pages: int = Field(description="Total number of pages")

    @classmethod
    def create(cls, items: List[T], total: int, page: int, page_size: int) -> "PaginatedResponse[T]":
        """
        Factory method to construct a paginated response.

        Args:
            items: Items for the current page.
            total: Total item count.
            page: Current page number.
            page_size: Items per page.

        Returns:
            Populated PaginatedResponse instance.
        """
        import math
        return cls(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=math.ceil(total / page_size) if page_size > 0 else 0,
        )