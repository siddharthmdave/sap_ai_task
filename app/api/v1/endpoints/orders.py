# app/api/v1/endpoints/orders.py
"""
Orders API Endpoints – Controller Layer.

Implements all order-related REST API endpoints following the MVC pattern.
Controllers are thin: they validate HTTP-layer concerns, delegate to services,
and format responses. Business logic lives in the service layer.

Endpoints:
  GET /orders/customer/{customer_id} - Orders for a specific customer
  GET /orders/stats - Aggregated revenue statistics
  GET /orders/recent - Orders from the last N days
  POST /orders/ask - Natural language query (AI)
  GET /orders/semantic_search - Semantic vector search (AI)

All responses are wrapped in the SuccessResponse envelope.
All errors are handled by the centralized error handler service.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.logging import get_logger
from app.repositories.order_repository import OrderRepository

from app.schemas.common import (
    SuccessResponse,
    ResponseMeta,
)

from app.schemas.order import (
    NLQueryRequest,
    NLQueryResponse,
    OrderListResponse,
    OrderStatsResponse,
    SemanticSearchResponse,
    SemanticSearchResult,
)

from app.services.ai_service import AIService
from app.services.embedding_service import embedding_service
from app.services.order_service import OrderService

logger = get_logger(__name__)

router = APIRouter(prefix="/orders", tags=["Orders"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def success_response(data):
    """Create a standard success response envelope."""
    return SuccessResponse(
        data=data,
        meta=ResponseMeta(),
    )


# ---------------------------------------------------------------------------
# Dependency Factories
# ---------------------------------------------------------------------------

def get_order_service(
    db: AsyncSession = Depends(get_db),
) -> OrderService:
    """
    Construct an OrderService using the current request DB session.
    """
    return OrderService(
        repository=OrderRepository(db)
    )


def get_ai_service(
    db: AsyncSession = Depends(get_db),
) -> AIService:
    """
    Construct an AIService using the current request DB session.
    """
    return AIService(
        repository=OrderRepository(db)
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/customer/{customer_id}",
    response_model=SuccessResponse[OrderListResponse],
    summary="Get orders by customer",
    description=(
        "Returns all orders for the specified customer ID, "
        "ordered by date descending."
    ),
    responses={
        200: {"description": "Orders retrieved successfully"},
        400: {"description": "Invalid customer_id format"},
        404: {"description": "No orders found for the customer"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def get_customer_orders(
    customer_id: str,
    service: Annotated[
        OrderService,
        Depends(get_order_service),
    ],
) -> SuccessResponse[OrderListResponse]:
    """
    Retrieve all orders for a given customer.
    """
    result = await service.get_customer_orders(customer_id)
    return success_response(result)


@router.get(
    "/stats",
    response_model=SuccessResponse[OrderStatsResponse],
    summary="Get order statistics",
    description=(
        "Returns aggregated order statistics including "
        "revenue, average order value, counts, and breakdowns."
    ),
    responses={
        200: {"description": "Statistics computed successfully"},
        500: {"description": "Database error"},
    },
)
async def get_order_stats(
    service: Annotated[
        OrderService,
        Depends(get_order_service),
    ],
) -> SuccessResponse[OrderStatsResponse]:
    """
    Retrieve aggregated order statistics.
    """
    result = await service.get_stats()
    return success_response(result)


@router.get(
    "/recent",
    response_model=SuccessResponse[OrderListResponse],
    summary="Get recent orders",
    description="Returns all orders placed within the last N days.",
    responses={
        200: {"description": "Recent orders retrieved successfully"},
        400: {"description": "Invalid days parameter"},
        500: {"description": "Database error"},
    },
)
async def get_recent_orders(
    days: Annotated[
        int,
        Query(
            ge=1,
            le=3650,
            description="Number of days to look back (1-3650)",
            examples=[30],
        ),
    ] = 30,
    service: Annotated[
        OrderService,
        Depends(get_order_service),
    ] = None,
) -> SuccessResponse[OrderListResponse]:
    """
    Retrieve orders placed within the last N days.
    """
    result = await service.get_recent_orders(days)
    return success_response(result)


@router.post(
    "/ask",
    response_model=SuccessResponse[NLQueryResponse],
    summary="Natural language query",
    description=(
        "Convert a natural language question into SQL, "
        "execute it, and return the answer."
    ),
    responses={
        200: {"description": "Query answered successfully"},
        400: {"description": "Question invalid or SQL generation failed"},
        422: {"description": "Request validation failed"},
        503: {"description": "AI provider unavailable"},
    },
)
async def ask_question(
    request: NLQueryRequest,
    service: Annotated[
        AIService,
        Depends(get_ai_service),
    ],
) -> SuccessResponse[NLQueryResponse]:
    """
    Answer a natural language question about order data.
    """
    result_dict = await service.answer_question(
        request.question
    )

    response = NLQueryResponse(**result_dict)

    return success_response(response)


@router.get(
    "/semantic_search",
    response_model=SuccessResponse[SemanticSearchResponse],
    summary="Semantic order search",
    description=(
        "Search orders using vector embeddings and return "
        "the most semantically similar results."
    ),
    responses={
        200: {"description": "Search completed successfully"},
        400: {"description": "Invalid query parameters"},
        503: {"description": "Embedding index not ready"},
    },
)
async def semantic_search(
    q: Annotated[
        str,
        Query(
            min_length=2,
            max_length=200,
            description=(
                "Free-text search query "
                "(e.g. 'high value recent orders')"
            ),
            examples=["high value recent orders"],
        ),
    ],
    top_k: Annotated[
        int,
        Query(
            ge=1,
            le=100,
            description="Number of results to return (1-100)",
            examples=[5],
        ),
    ] = 5,
) -> SuccessResponse[SemanticSearchResponse]:
    """
    Search orders semantically using vector embeddings.
    """
    results = await embedding_service.search(
        query=q,
        top_k=top_k,
    )

    search_results = [
        SemanticSearchResult(**result)
        for result in results
    ]

    response = SemanticSearchResponse(
        results=search_results,
        query=q,
        top_k=top_k,
        index_size=embedding_service.index_size,
    )

    return success_response(response)