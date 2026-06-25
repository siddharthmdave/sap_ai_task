# app/core/exceptions.py

"""
Application Exception Hierarchy Module.

Defines a structured exception hierarchy for the ETL service.
All custom exceptions carry an HTTP status code, a machine-readable
error code, and a human-readable message.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


# =============================================================================
# Base Exception
# =============================================================================

class AppBaseException(Exception):
    """
    Root exception for all application-specific errors.
    """

    http_status_code: int = 500
    error_code: str = "INTERNAL_ERROR"

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None,
    ) -> None:
        super().__init__(message)

        self.message = message
        self.details = details or {}

        if error_code:
            self.error_code = error_code

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        }


# =============================================================================
# 400 Bad Request
# =============================================================================

class ValidationError(AppBaseException):
    """
    Business validation error.
    """

    http_status_code = 400
    error_code = "VALIDATION_ERROR"


class InvalidDateFormatError(ValidationError):
    error_code = "INVALID_DATE_FORMAT"

    def __init__(self, value: str, field: str = "date") -> None:
        super().__init__(
            message=f"Cannot parse '{value}' as a valid date for field '{field}'.",
            details={
                "field": field,
                "value": value,
            },
        )


class InvalidCurrencyError(ValidationError):
    error_code = "INVALID_CURRENCY"

    def __init__(self, currency: str) -> None:
        super().__init__(
            message=f"Unsupported currency '{currency}'. Supported: USD, EUR.",
            details={
                "currency": currency,
            },
        )


class InvalidQueryError(ValidationError):
    error_code = "INVALID_QUERY"

    def __init__(
        self,
        question: str,
        reason: str,
    ) -> None:
        super().__init__(
            message=f"Cannot answer question from available schema: {reason}",
            details={
                "question": question,
                "reason": reason,
            },
        )


# =============================================================================
# 404 Not Found
# =============================================================================

class NotFoundError(AppBaseException):
    http_status_code = 404
    error_code = "NOT_FOUND"


class OrderNotFoundError(NotFoundError):
    error_code = "ORDER_NOT_FOUND"

    def __init__(self, order_id: str) -> None:
        super().__init__(
            message=f"Order '{order_id}' not found.",
            details={
                "order_id": order_id,
            },
        )


class CustomerNotFoundError(NotFoundError):
    error_code = "CUSTOMER_NOT_FOUND"

    def __init__(self, customer_id: str) -> None:
        super().__init__(
            message=f"No orders found for customer '{customer_id}'.",
            details={
                "customer_id": customer_id,
            },
        )


# =============================================================================
# 409 Conflict
# =============================================================================

class DuplicateOrderError(AppBaseException):
    http_status_code = 409
    error_code = "DUPLICATE_ORDER"

    def __init__(self, order_id: str) -> None:
        super().__init__(
            message=f"Order '{order_id}' already exists.",
            details={
                "order_id": order_id,
            },
        )


# =============================================================================
# 422 Unprocessable Entity
# =============================================================================

class ETLProcessingError(AppBaseException):
    http_status_code = 422
    error_code = "ETL_PROCESSING_ERROR"

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        row: Optional[int] = None,
    ) -> None:

        details: Dict[str, Any] = {}

        if file_path:
            details["file_path"] = file_path

        if row is not None:
            details["row"] = row

        super().__init__(
            message=message,
            details=details,
        )


# =============================================================================
# 429 Too Many Requests
# =============================================================================

class RateLimitExceededError(AppBaseException):
    http_status_code = 429
    error_code = "RATE_LIMIT_EXCEEDED"

    def __init__(
        self,
        limit: int,
        window: str = "minute",
    ) -> None:
        super().__init__(
            message=f"Rate limit of {limit} requests per {window} exceeded.",
            details={
                "limit": limit,
                "window": window,
            },
        )


# =============================================================================
# 500 Internal Server Error
# =============================================================================

class DatabaseError(AppBaseException):
    http_status_code = 500
    error_code = "DATABASE_ERROR"

    def __init__(self, message: str = "Database operation failed.") -> None:
        super().__init__(message=message)


class InternalError(AppBaseException):
    http_status_code = 500
    error_code = "INTERNAL_ERROR"

    def __init__(self, message: str = "Internal server error.") -> None:
        super().__init__(message=message)


# =============================================================================
# 503 Service Unavailable
# =============================================================================

class AIServiceUnavailableError(AppBaseException):
    """
    Raised when AI provider is unavailable or not configured.
    """

    http_status_code = 503
    error_code = "AI_SERVICE_UNAVAILABLE"

    def __init__(
        self,
        provider: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        message = reason or "AI service is unavailable."
        details: Dict[str, Any] = {}

        if provider:
            details["provider"] = provider
            if not reason:
                message = (
                    f"AI provider '{provider}' is unavailable or not configured."
                )

        super().__init__(message=message, details=details)


class AIServiceError(AppBaseException):
    """Raised when an AI provider returns an unexpected error."""

    http_status_code = 502
    error_code = "AI_SERVICE_ERROR"

    def __init__(
        self,
        provider: Optional[str] = None,
        reason: str = "AI service error.",
    ) -> None:
        details: Dict[str, Any] = {}
        if provider:
            details["provider"] = provider
        super().__init__(message=reason, details=details)


class SQLGenerationError(AppBaseException):
    """Raised when SQL generation fails after all retries."""

    http_status_code = 400
    error_code = "SQL_GENERATION_ERROR"

    def __init__(
        self,
        question: str,
        last_sql: str = "",
        last_error: str = "Unknown error",
    ) -> None:
        super().__init__(
            message=f"Failed to generate valid SQL: {last_error}",
            details={
                "question": question,
                "last_sql": last_sql,
                "last_error": last_error,
            },
        )


class EmbeddingServiceError(AppBaseException):
    """
    Raised when semantic search index is unavailable.
    """

    http_status_code = 503
    error_code = "EMBEDDING_SERVICE_UNAVAILABLE"

    def __init__(
        self,
        message: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        super().__init__(
            message=reason or message or (
                "Embedding index is unavailable. "
                "Run ETL and embedding generation first."
            ),
        )