# app/utils/embedding.py
"""Helpers for building embedding text from order records."""

from __future__ import annotations

from typing import Any


def order_to_embedding_text(order: Any) -> str:
    """
    Serialize an order record to a short text string for embedding.

  Supports ORM Order models (with to_embedding_text) and plain dicts.
    """
    if hasattr(order, "to_embedding_text"):
        return order.to_embedding_text()

    customer_id = getattr(order, "customer_id", order.get("customer_id", ""))
    amount_usd = getattr(order, "amount_usd", order.get("amount_usd", 0.0))
    order_date = getattr(order, "order_date", order.get("order_date", ""))

    return f"customer {customer_id}, ${float(amount_usd):.2f} USD, {order_date}"
