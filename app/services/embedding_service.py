# app/services/embedding_service.py
"""
Semantic Search Embedding Service.

Implements vector embeddings and FAISS-based semantic search for
the GET /orders/semantic_search endpoint.

Architecture:
    - Model: all-MiniLM-L6-v2
    - Index: FAISS IndexFlatIP (cosine similarity via normalized vectors)
    - Atomic index swapping during rebuilds
    - Async-safe CPU offloading using thread executors
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from app.core.config import settings
from app.core.exceptions import EmbeddingServiceError
from app.core.logging import get_logger
from app.utils.embedding import order_to_embedding_text

logger = get_logger(__name__)


class EmbeddingService:
    """
    Manages sentence embeddings and FAISS vector index
    for semantic order search.
    """

    def __init__(self) -> None:
        self._model = None
        self._model_lock = threading.Lock()

        self._index = None
        self._order_metadata: List[Dict[str, Any]] = []

        self._index_lock = threading.RLock()

        self._is_ready = False
        self._index_size = 0

    # ------------------------------------------------------------------
    # Model Loading
    # ------------------------------------------------------------------

    def _load_model(self) -> Any:
        """
        Lazy-load the SentenceTransformer model.

        Returns:
            SentenceTransformer instance.

        Raises:
            EmbeddingServiceError
        """
        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is not None:
                return self._model

            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingServiceError(
                    reason=(
                        "sentence-transformers is not installed. "
                        "Run: pip install sentence-transformers"
                    )
                ) from exc

            logger.info(
                "embedding_model_loading",
                model=settings.EMBEDDING_MODEL,
            )

            start = time.perf_counter()

            self._model = SentenceTransformer(
                settings.EMBEDDING_MODEL,
                device=settings.EMBEDDING_DEVICE,
            )

            duration = time.perf_counter() - start

            logger.info(
                "embedding_model_loaded",
                model=settings.EMBEDDING_MODEL,
                device=settings.EMBEDDING_DEVICE,
                duration_seconds=round(duration, 2),
            )

        return self._model

    # ------------------------------------------------------------------
    # Index Building
    # ------------------------------------------------------------------

    async def build_index(self, orders: List[Any]) -> None:
        """
        Build or rebuild the FAISS index.
        """
        if not orders:
            logger.warning(
                "embedding_index_build_skipped",
                reason="no orders provided",
            )
            return

        logger.info(
            "embedding_index_build_started",
            order_count=len(orders),
        )

        start = time.perf_counter()

        try:
            loop = asyncio.get_running_loop()

            new_index, new_metadata = await loop.run_in_executor(
                None,
                self._build_index_sync,
                orders,
            )

        except Exception as exc:
            logger.exception(
                "embedding_index_build_failed",
                error=str(exc),
            )
            raise EmbeddingServiceError(
                reason=f"Index build failed: {exc}"
            ) from exc

        with self._index_lock:
            self._index = new_index
            self._order_metadata = new_metadata
            self._index_size = len(new_metadata)
            self._is_ready = True

        duration = time.perf_counter() - start

        logger.info(
            "embedding_index_build_complete",
            order_count=len(orders),
            index_size=self._index_size,
            duration_seconds=round(duration, 2),
        )

    def _build_index_sync(
        self,
        orders: List[Any],
    ) -> Tuple[Any, List[Dict[str, Any]]]:
        """
        Synchronous FAISS index construction.
        """
        try:
            import faiss
        except ImportError as exc:
            raise EmbeddingServiceError(
                reason="faiss-cpu is not installed. Run: pip install faiss-cpu"
            ) from exc

        model = self._load_model()

        texts = [order_to_embedding_text(order) for order in orders]

        logger.debug(
            "generating_embeddings",
            count=len(texts),
            batch_size=settings.EMBEDDING_BATCH_SIZE,
        )

        embeddings = model.encode(
            texts,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        embeddings = embeddings.astype(np.float32)

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        metadata = [
            {
                "order_id": order.order_id,
                "customer_id": order.customer_id,
                "amount_usd": order.amount_usd,
                "order_date": str(order.order_date),
                "currency": order.currency,
            }
            for order in orders
        ]

        return index, metadata

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search.
        """
        query = query.strip()

        if not query:
            return []

        with self._index_lock:
            if not self._is_ready or self._index is None:
                raise EmbeddingServiceError(
                    reason=(
                        "FAISS index is not ready. "
                        "Run the ETL pipeline first."
                    )
                )

            index = self._index
            metadata = self._order_metadata.copy()

        effective_k = min(max(top_k, 1), len(metadata))

        if effective_k == 0:
            return []

        loop = asyncio.get_running_loop()

        results = await loop.run_in_executor(
            None,
            self._search_sync,
            query,
            effective_k,
            index,
            metadata,
        )

        logger.info(
            "semantic_search_complete",
            query=query,
            top_k=top_k,
            results_returned=len(results),
        )

        return results

    def _search_sync(
        self,
        query: str,
        top_k: int,
        index: Any,
        metadata: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Synchronous FAISS search.
        """
        model = self._load_model()

        query_embedding = model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        scores, indices = index.search(
            query_embedding,
            top_k,
        )

        results: List[Dict[str, Any]] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(metadata):
                continue

            if np.isnan(score):
                continue

            result = metadata[idx].copy()

            result["score"] = float(
                max(
                    0.0,
                    min(1.0, float(score)),
                )
            )

            results.append(result)

        results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return results

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def index_size(self) -> int:
        return self._index_size

    def get_status(self) -> str:
        return "ready" if self._is_ready else "not_ready"


# ----------------------------------------------------------------------
# Singleton
# ----------------------------------------------------------------------

embedding_service = EmbeddingService()