# main.py
"""
FastAPI Application Entry Point - ETL Order Service.

Enterprise-grade microservice for customer order data with:
 - Full OpenAPI 3.1 / Swagger UI documentation (auto-generated)
 - CORS, security-headers, request-ID, and logging middleware
 - Rate limiting and Prometheus metrics
 - AI-augmented natural-language query and semantic-search endpoints
 - Structured error handling with RFC 7807 Problem Details

Usage:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from app.core.config import settings
from app.core.database import create_all_tables, dispose_engine
from app.core.logging import configure_logging, get_logger

configure_logging(
    log_level=settings.LOG_LEVEL.value,
    is_development=settings.is_development,
)

logger = get_logger(__name__)

_OPENAPI_TAGS: list = [
    {
        "name": "Health",
        "description": (
            "Liveness and readiness probes. "
            "Used by Kubernetes / load-balancers to determine service health."
        ),
    },
    {
        "name": "Orders",
        "description": (
            "CRUD operations on customer orders loaded by the ETL pipeline. "
            "Supports filtering, pagination, and aggregation."
        ),
    },
    {
        "name": "AI Query",
        "description": (
            "**LLM-powered natural-language query layer (Part 4).** "
            "Send a plain-English question and receive structured JSON results."
        ),
        "externalDocs": {
            "description": "AI Model Configuration Standard",
            "url": "https://github.com/your-org/etl-service/blob/main/docs/AI_MODEL_STANDARDS.md",
        },
    },
    {
        "name": "Semantic Search",
        "description": (
            "**Embedding-based semantic search (Part 4).** "
            "Returns the *k* most semantically similar orders."
        ),
    },
    {
        "name": "ETL",
        "description": (
            "Trigger or inspect the ETL pipeline that ingests `data/orders.csv` "
            "into the SQLite database and rebuilds the FAISS embedding index."
        ),
    },
]

_OPENAPI_DESCRIPTION = """
## ETL Order Service - Enterprise REST API

This service ingests customer order data via an **ETL pipeline**, exposes a
**REST API** for querying and aggregating orders, and provides an
**AI-augmented query layer** powered by configurable LLM providers.

---

### Quick-start

```bash
# 1. Run ETL to load data
python etl.py load data/orders.csv

# 2. Start the API server
uvicorn main:app --reload

# 3. Open interactive docs
open http://localhost:8000/docs
```

### Core endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /orders/customer/{id}` | Orders for a customer |
| `GET /orders/stats` | Revenue and aggregation stats |
| `GET /orders/recent?days=N` | Orders from the last N days |
| `POST /orders/ask` | Natural-language SQL query (AI) |
| `GET /orders/semantic_search` | Vector similarity search (AI) |
| `GET /healthz` | Liveness probe |
"""

_OPENAPI_CONTACT = {
    "name": "ETL Order Service",
    "email": "support@example.com",
}

_OPENAPI_LICENSE = {
    "name": "MIT",
    "url": "https://opensource.org/licenses/MIT",
}

_OPENAPI_SERVERS = [
    {"url": "http://localhost:8000", "description": "Local development"},
    {"url": "https://etl-service-staging.example.com", "description": "Staging"},
    {"url": "https://etl-service.example.com", "description": "Production"},
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown lifecycle."""
    startup_start = time.perf_counter()
    logger.info(
        "application_starting",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        env=settings.APP_ENV.value,
    )

    await create_all_tables()

    try:
        from app.core.database import get_db_context
        from app.repositories.order_repository import OrderRepository
        from app.services.embedding_service import embedding_service

        async with get_db_context() as db:
            repo = OrderRepository(db)
            orders = await repo.get_all()

        if orders:
            await embedding_service.build_index(orders)
            logger.info(
                "embedding_index_ready",
                order_count=len(orders),
                index_size=embedding_service.index_size,
            )
        else:
            logger.warning(
                "embedding_index_skipped",
                reason="No orders in database. Run ETL pipeline first.",
            )
    except Exception as exc:
        logger.warning(
            "embedding_index_build_failed_on_startup",
            error=str(exc),
            note="Semantic search will be unavailable until ETL runs.",
        )

    startup_duration = time.perf_counter() - startup_start
    logger.info(
        "application_started",
        startup_duration_seconds=round(startup_duration, 2),
        host=settings.HOST,
        port=settings.PORT,
    )

    yield

    logger.info("application_shutting_down")
    await dispose_engine()
    logger.info("application_shutdown_complete")


def create_app() -> FastAPI:
    """Application factory that creates and configures the FastAPI instance."""
    _docs_url = "/docs" if not settings.is_production else None
    _redoc_url = "/redoc" if not settings.is_production else None
    _openapi_url = "/openapi.json" if not settings.is_production else None

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=_OPENAPI_DESCRIPTION,
        openapi_tags=_OPENAPI_TAGS,
        contact=_OPENAPI_CONTACT,
        license_info=_OPENAPI_LICENSE,
        servers=_OPENAPI_SERVERS,
        docs_url=_docs_url,
        redoc_url=_redoc_url,
        openapi_url=_openapi_url,
        lifespan=lifespan,
        swagger_ui_parameters={
            "defaultModelsExpandDepth": 2,
            "defaultModelExpandDepth": 3,
            "docExpansion": "list",
            "filter": True,
            "showExtensions": True,
            "showCommonExtensions": True,
            "tryItOutEnabled": True,
            "persistAuthorization": True,
            "displayRequestDuration": True,
            "syntaxHighlight.theme": "monokai",
        },
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time-Ms"],
    )

    from app.middleware.request_middleware import SecurityHeadersMiddleware
    app.add_middleware(SecurityHeadersMiddleware)

    from app.middleware.request_middleware import RequestLoggingMiddleware
    app.add_middleware(RequestLoggingMiddleware)

    from app.middleware.request_middleware import RequestIDMiddleware
    app.add_middleware(RequestIDMiddleware)

    from app.services.error_handler import register_exception_handlers
    register_exception_handlers(app)

    from app.api.v1.endpoints.health import router as health_router
    from app.api.v1.endpoints.orders import router as orders_router

    app.include_router(health_router)
    app.include_router(orders_router, prefix="/api/v1")
    app.include_router(orders_router, prefix="")

    if settings.ENABLE_METRICS:
        try:
            from prometheus_client import make_asgi_app
            metrics_app = make_asgi_app()
            app.mount(settings.METRICS_PATH, metrics_app)
            logger.info("prometheus_metrics_enabled", path=settings.METRICS_PATH)
        except ImportError:
            logger.warning("prometheus_client_not_installed", note="Metrics disabled")

    _patch_openapi(app)

    logger.info(
        "application_configured",
        routes=_list_route_paths(app),
        docs_url=_docs_url,
        redoc_url=_redoc_url,
    )

    return app


def _list_route_paths(app: FastAPI) -> list[str]:
    """Collect registered HTTP route paths for startup logging."""
    paths: list[str] = []
    for route in app.routes:
        path = getattr(route, "path", None)
        if path:
            paths.append(path)
    return sorted(paths)


def _patch_openapi(app: FastAPI) -> None:
    """Inject security schemes and Swagger UI branding into the OpenAPI schema."""

    def custom_openapi() -> dict:
        if app.openapi_schema:
            return app.openapi_schema

        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            tags=app.openapi_tags,
            routes=app.routes,
            contact=app.contact,
            license_info=app.license_info,
            servers=app.servers,
        )

        schema.setdefault("components", {})
        schema["components"]["securitySchemes"] = {
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "JWT access token.",
            },
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": "Static API key for service-to-service calls.",
            },
        }

        schema["security"] = [
            {"BearerAuth": []},
            {"ApiKeyAuth": []},
        ]

        schema.setdefault("info", {})
        schema["info"]["x-logo"] = {
            "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png",
            "altText": "ETL Order Service",
        }

        app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = custom_openapi  # type: ignore[method-assign]


app: FastAPI = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        workers=settings.WORKERS,
        log_level=settings.LOG_LEVEL.value.lower(),
        access_log=False,
    )
