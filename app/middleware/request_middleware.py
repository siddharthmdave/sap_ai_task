# app/middleware/request_middleware.py

"""
Request Processing Middleware.

Provides:

1. RequestIDMiddleware
2. RequestLoggingMiddleware
3. SecurityHeadersMiddleware
4. TrustedHostMiddleware
"""

from __future__ import annotations

import time
import uuid

from starlette.middleware.base import (
    BaseHTTPMiddleware,
    RequestResponseEndpoint,
)
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from app.core.logging import get_logger, set_request_id

logger = get_logger(__name__)


# =============================================================================
# Request ID Middleware
# =============================================================================

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Inject a correlation ID into every request and response.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:

        request_id = (
            request.headers.get("X-Request-ID")
            or str(uuid.uuid4())
        )

        request.state.request_id = request_id

        set_request_id(request_id)

        response = await call_next(request)

        response.headers["X-Request-ID"] = request_id

        return response


# =============================================================================
# Request Logging Middleware
# =============================================================================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Structured request logging.
    """

    DEFAULT_SKIP_PATHS = [
        "/healthz",
        "/metrics",
        "/favicon.ico",
    ]

    def __init__(
        self,
        app: ASGIApp,
        skip_paths: list[str] | None = None,
    ) -> None:
        super().__init__(app)
        self.skip_paths = skip_paths or self.DEFAULT_SKIP_PATHS

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:

        if any(
            request.url.path.startswith(path)
            for path in self.skip_paths
        ):
            return await call_next(request)

        start_time = time.perf_counter()
        client_ip = self._get_client_ip(request)

        try:
            response = await call_next(request)

            duration_ms = (
                time.perf_counter() - start_time
            ) * 1000

            logger.info(
                "http_request",
                method=request.method,
                path=request.url.path,
                query=request.url.query or None,
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
                client_ip=client_ip,
                request_id=getattr(
                    request.state,
                    "request_id",
                    None,
                ),
            )

            response.headers["X-Response-Time-Ms"] = str(
                round(duration_ms, 2)
            )

            return response

        except Exception as exc:
            duration_ms = (
                time.perf_counter() - start_time
            ) * 1000

            logger.error(
                "http_request_error",
                method=request.method,
                path=request.url.path,
                duration_ms=round(duration_ms, 2),
                client_ip=client_ip,
                error=str(exc),
            )

            raise

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        """
        Get client IP respecting proxy headers.
        """

        forwarded_for = request.headers.get(
            "X-Forwarded-For"
        )

        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        return (
            request.client.host
            if request.client
            else "unknown"
        )


# =============================================================================
# Security Headers Middleware
# =============================================================================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add common HTTP security headers.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:

        response = await call_next(request)

        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-XSS-Protection"] = "1; mode=block"

        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "frame-ancestors 'none'; "
            "object-src 'none';"
        )

        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; "
                "includeSubDomains"
            )

        return response


# =============================================================================
# Trusted Host Middleware
# =============================================================================

class TrustedHostMiddleware(BaseHTTPMiddleware):
    """
    Validate Host header against an allowlist.
    """

    def __init__(
        self,
        app: ASGIApp,
        allowed_hosts: list[str] | None = None,
    ) -> None:
        super().__init__(app)

        self.allowed_hosts = [
            host.lower()
            for host in (
                allowed_hosts
                or ["localhost", "127.0.0.1"]
            )
        ]

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:

        host = (
            request.headers.get("host", "")
            .split(":")[0]
            .lower()
        )

        if self.allowed_hosts and host not in self.allowed_hosts:

            logger.warning(
                "rejected_host_header",
                host=host,
                allowed_hosts=self.allowed_hosts,
            )

            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": {
                        "error_code": "INVALID_HOST",
                        "message": (
                            f"Host '{host}' "
                            "is not allowed."
                        ),
                        "details": {},
                    },
                },
            )

        return await call_next(request)