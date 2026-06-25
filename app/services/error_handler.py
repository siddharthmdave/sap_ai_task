# app/services/error_handler.py

"""
Centralized Error Handler Service.

This service is the single point of truth for converting exceptions into
structured HTTP responses. It registers handlers on the FastAPI application
for all exception types used in the codebase.

Design:
    - AppBaseException subclasses + structured JSON with error_code + message
    - Pydantic RequestValidationError + 422 with per-field error details
    - HTTPException + pass-through with envelope wrapping
    - Unhandled Exception + 500 with sanitized message (no stack traces in prod)

All responses follow the ErrorResponse envelope defined in app/schemas/common.py.

Usage:
    from app.services.error_handler import register_exception_handlers
    register_exception_handlers(app)
"""

from __future__ import annotations

import traceback
from typing import Any, Dict, Union

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.exceptions import AppBaseException
from app.core.logging import get_logger
from app.schemas.common import (
    ErrorDetail,
    ErrorResponse,
    FieldValidationError,
    ResponseMeta,
    ValidationErrorResponse,
)

logger = get_logger(__name__)


# — Response Builders ————————————————————————————————————————————————

def _build_error_response(
    request: Request,
    status_code: int,
    error_code: str,
    message: str,
    details: Dict[str, Any] | None = None,
) -> JSONResponse:
    """
    Build a standardized JSON error response.

    Args:
        request: The incoming FastAPI request (used to extract request_id).
        status_code: HTTP status code for the response.
        error_code: Machine-readable error identifier.
        message: Human-readable error description.
        details: Optional additional context dict.

    Returns:
        JSONResponse with the ErrorResponse envelope.
    """
    request_id = request.headers.get("X-Request-ID") or request.state.__dict__.get("request_id")

    body = ErrorResponse(
        error=ErrorDetail(
            error_code=error_code,
            message=message,
            details=details or {},
        ),
        meta=ResponseMeta(request_id=request_id),
    )

    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json"),
    )

def _build_validation_error_response(
    request: Request,
    field_errors: list[FieldValidationError],
) -> JSONResponse:
    """
    Build a 422 validation error response with per-field details.

    Args:
        request: The incoming FastAPI request.
        field_errors: List of field-level validation errors.

    Returns:
        JSONResponse with the ValidationErrorResponse envelope.
    """
    request_id = request.headers.get("X-Request-ID") or request.state.__dict__.get("request_id")

    body = ValidationErrorResponse(
        error=ErrorDetail(
            error_code="VALIDATION_ERROR",
            message=f"Request validation failed with {len(field_errors)} error(s).",
        ),
        field_errors=field_errors,
        meta=ResponseMeta(request_id=request_id),
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=body.model_dump(mode="json"),
    )


# — Exception Handlers ——————————————————————————————————————————————

async def handle_app_exception(
    request: Request,
    exc: AppBaseException,
) -> JSONResponse:
    """
    Handler for all AppBaseException subclasses.

    Converts domain exceptions into structured HTTP responses using the
    http_status_code and error_code defined on each exception class.

    Args:
        request: The incoming FastAPI request.
        exc: The AppBaseException instance.

    Returns:
        JSONResponse with the appropriate status code and error envelope.
    """
    logger.warning(
        "app_exception",
        error_code=exc.error_code,
        message=exc.message,
        status_code=exc.http_status_code,
        path=str(request.url),
        method=request.method,
        details=exc.details,
    )

    return _build_error_response(
        request=request,
        status_code=exc.http_status_code,
        error_code=exc.error_code,
        message=exc.message,
        details=exc.details,
    )

async def handle_http_exception(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    """
    Handler for Starlette/FastAPI HTTPException.

    Wraps standard HTTP exceptions in the error envelope so clients
    always receive a consistent response structure.

    Args:
        request: The incoming FastAPI request.
        exc: The StarletteHTTPException instance.

    Returns:
        JSONResponse with the error envelope.
    """
    # Map common HTTP status codes to error codes
    error_code_map: Dict[int, str] = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        405: "METHOD_NOT_ALLOWED",
        408: "REQUEST_TIMEOUT",
        409: "CONFLICT",
        429: "RATE_LIMIT_EXCEEDED",
        500: "INTERNAL_ERROR",
        502: "BAD_GATEWAY",
        503: "SERVICE_UNAVAILABLE",
    }

    error_code = error_code_map.get(exc.status_code, f"HTTP_{exc.status_code}")
    message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)

    logger.warning(
        "http_exception",
        status_code=exc.status_code,
        error_code=error_code,
        message=message,
        path=str(request.url),
        method=request.method,
    )

    return _build_error_response(
        request=request,
        status_code=exc.status_code,
        error_code=error_code,
        message=message,
    )


async def handle_request_validation_error(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """
    Handler for Pydantic RequestValidationError (422 Unprocessable Entity).

    Extracts per-field error details from the Pydantic error list and
    returns them in a structured format for client-side form validation.

    Args:
        request: The incoming FastAPI request.
        exc: The RequestValidationError instance from FastAPI/Pydantic.

    Returns:
        JSONResponse with 422 status and per-field error details.
    """
    field_errors: list[FieldValidationError] = []

    for error in exc.errors():
        # Build a dot-notation field path from the error location tuple
        loc = error.get("loc", ())
        # Skip the first element if it's "body", "query", or "path"
        field_parts = [str(p) for p in loc if p not in ("body", "query", "path")]
        field_path = ".".join(field_parts) if field_parts else str(loc)

        field_errors.append(
            FieldValidationError(
                field=field_path,
                message=error.get("msg", "Validation error"),
                invalid_value=error.get("input"),
            )
        )

    logger.warning(
        "request_validation_error",
        path=str(request.url),
        method=request.method,
        error_count=len(field_errors),
        errors=[e.model_dump() for e in field_errors],
    )

    return _build_validation_error_response(request=request, field_errors=field_errors)

async def handle_unhandled_exception(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """
    Catch-all handler for unhandled exceptions.

    Logs the full stack trace internally but returns a sanitized error
    message to the client (no internal details exposed in production).

    Args:
        request: The incoming FastAPI request.
        exc: The unhandled exception.

    Returns:
        JSONResponse with 500 status and a generic error message.
    """
    from app.core.config import settings

    # Always log the full traceback internally
    logger.error(
        "unhandled_exception",
        exc_type=type(exc).__name__,
        exc_message=str(exc),
        path=str(request.url),
        method=request.method,
        traceback=traceback.format_exc(),
    )

    # In development, include the exception message for easier debugging
    if settings.is_development:
        message = f"Internal server error: {type(exc).__name__}: {exc}"
        details: Dict[str, Any] = {"traceback": traceback.format_exc().splitlines()[:-5]}
    else:
        # In production, never expose internal details
        message = "An unexpected internal error occurred. Please try again later."
        details = {}

    return _build_error_response(
        request=request,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code="INTERNAL_ERROR",
        message=message,
        details=details,
    )


# — Registration ————————————————————————————————————————————————

def register_exception_handlers(app: FastAPI) -> None:
    """
    Register all exception handlers on the FastAPI application instance.

    This function must be called during application startup, after the
    FastAPI app is created but before it starts serving requests.

    Handler registration order matters: more specific handlers are registered
    before more general ones to ensure correct dispatch.

    Args:
        app: The FastAPI application instance.

    Example:
        app = FastAPI()
        register_exception_handlers(app)
    """
    # 1. Domain exceptions (most specific - registered first)
    app.add_exception_handler(AppBaseException, handle_app_exception)  # type: ignore[arg-type]

    # 2. Pydantic request validation errors (422)
    app.add_exception_handler(RequestValidationError, handle_request_validation_error)  # type: ignore[arg-type]

    # 3. Standard HTTP exceptions
    app.add_exception_handler(StarletteHTTPException, handle_http_exception)  # type: ignore[arg-type]

    # 4. Catch-all for unhandled exceptions (least specific - registered last)
    app.add_exception_handler(Exception, handle_unhandled_exception)  # type: ignore[arg-type]

    logger.info("exception_handlers_registered", handler_count=4)
