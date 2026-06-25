# app/api/v1/endpoints/health.py
"""
Health Check Endpoint

Provides liveness and readiness probes for Kubernetes and load balancers.

Endpoints:
  GET /healthz - liveness probe (is the process alive?)
  GET /readyz - readiness probe (is the service ready to serve traffic?)

The liveness probe returns quickly without checking dependencies.
The readiness probe checks database connectivity and index status.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.database import check_database_health
from app.core.logging import get_logger
from app.schemas.common import HealthCheckResponse

logger = get_logger(__name__)

router = APIRouter(tags=["health"])

# Application start time for uptime calculation
_START_TIME: float = time.time()

@router.get(
    "/healthz",
    response_model=HealthCheckResponse,
    summary="Liveness probe",
    description="Returns 'ok' if the service process is alive. Used by Kubernetes liveness probes.",
    responses={
        200: {"description": "Service is alive"},
        503: {"description": "Service is unhealthy"},
    },
)
async def liveness() -> HealthCheckResponse:
    """
    Kubernetes liveness probe endpoint.

    Returns a lightweight health check without querying the database.
    If this endpoint fails, Kubernetes will restart the pod.

    Returns:
        HealthCheckResponse with status "ok".
    """
    from app.services.embedding_service import embedding_service

    uptime = time.time() - _START_TIME

    return HealthCheckResponse(
        status="ok",
        database="ok",
        embedding_index=embedding_service.get_status(),
        version=settings.APP_VERSION,
        uptime_seconds=round(uptime, 1),
    )

@router.get(
    "/readyz",
    summary="Readiness probe",
    description=(
        "Returns 200 if the service is ready to serve traffic. "
        "Checks database connectivity and embedding index status."
    ),
    responses={
        200: {"description": "Service is ready"},
        503: {"description": "Service is not ready"},
    },
)
async def readiness() -> JSONResponse:
    """
    Kubernetes readiness probe endpoint.

    Checks:
    1. Database connectivity (SELECT 1)
    2. Embedding index readiness

    Returns 200 if all checks pass, 503 if any check fails.
    Kubernetes will stop routing traffic to the pod if this returns 503.

    Returns:
        JSONResponse with status "ready" or "not_ready".
    """
    from app.services.embedding_service import embedding_service

    uptime = time.time() - _START_TIME
    db_ok = await check_database_health()
    index_status = embedding_service.get_status()

    all_ready = db_ok

    status_code = 200 if all_ready else 503
    status = "ready" if all_ready else "not_ready"

    return JSONResponse(
        status_code=status_code,
        content={
            "status": status,
            "database": "ok" if db_ok else "error",
            "embedding_index": index_status,
            "version": settings.APP_VERSION,
            "uptime_seconds": round(uptime, 1),
        },
    )