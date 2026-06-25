# app/services/ai_service.py
"""
AI Service - Natural Language Query Orchestrator.

Orchestrates the NL->SQL pipeline using the Strategy + Factory pattern.
Supports LangGraph agent (Part 4c) with manual retry loop fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.exceptions import (
    InvalidQueryError,
    SQLGenerationError,
)
from app.core.logging import get_logger
from app.repositories.order_repository import OrderRepository
from app.services.llm.base import LLMMessage, LLMResponse
from app.services.llm.factory import get_llm_provider

logger = get_logger(__name__)

_SCHEMA_DESCRIPTION: str = """
Table: orders
Columns:
  order_id           TEXT  PRIMARY KEY   - unique order identifier
  customer_id        TEXT  NOT NULL      - alphanumeric customer identifier
  order_date         DATE  NOT NULL      - ISO 8601 date (YYYY-MM-DD)
  amount_usd         REAL  NOT NULL      - order amount in USD (after currency conversion)
  original_amount    REAL                - raw amount from source CSV
  currency           TEXT  NOT NULL      - original currency code (USD or EUR)
  created_at         DATETIME            - ETL load timestamp
  updated_at         DATETIME            - last update timestamp
""".strip()

_SYSTEM_PROMPT_TEMPLATE: str = """
You are a SQL expert assistant for a customer order database.

DATABASE SCHEMA:
{schema}

RULES - follow these exactly:
1. Return ONLY a single valid SQLite SELECT statement. No explanation, no markdown.
2. Use only the columns listed in the schema above.
3. Do NOT use DROP, DELETE, INSERT, UPDATE, ALTER, CREATE, or TRUNCATE.
4. Do NOT use subqueries that reference tables other than 'orders'.
5. If the question cannot be answered from this schema, reply with exactly:
   OUT_OF_SCOPE: <brief reason>
6. Always alias aggregates for clarity (e.g. SUM(amount_usd) AS total_revenue).
7. Use parameterized-style literals - quote string values with single quotes.
""".strip()


@dataclass
class NLQueryResult:
    """Structured result returned by AIService.answer_question()."""

    answer: str
    sql_used: str
    rows: List[Dict[str, Any]]
    token_count: int
    retry_count: int
    model_used: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "sql_used": self.sql_used,
            "rows": self.rows,
            "token_count": self.token_count,
            "retry_count": self.retry_count,
            "model_used": self.model_used,
        }


class AIService:
    """Orchestrates the natural language + SQL + answer pipeline."""

    def __init__(self, repository: OrderRepository) -> None:
        self._repository = repository

    async def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Convert a natural language question to SQL, execute it, and return
        a structured answer.
        """
        logger.info("ai_service_question_received", question=question[:100])

        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(schema=_SCHEMA_DESCRIPTION)

        try:
            from app.services.llm.graph import _SQL_AGENT_GRAPH, run_sql_agent

            if _SQL_AGENT_GRAPH is not None:
                logger.debug("ai_service_using_langgraph")
                return await run_sql_agent(
                    question=question,
                    system_prompt=system_prompt,
                    schema=_SCHEMA_DESCRIPTION,
                    repository=self._repository,
                )
        except ImportError:
            logger.debug("ai_service_langgraph_unavailable_using_manual_loop")

        logger.debug("ai_service_using_manual_loop")
        return await self._manual_retry_loop(question, system_prompt)

    async def _manual_retry_loop(
        self,
        question: str,
        system_prompt: str,
    ) -> Dict[str, Any]:
        """Fallback retry loop when LangGraph is unavailable."""
        provider = get_llm_provider()
        messages: List[LLMMessage] = [LLMMessage(role="user", content=question)]

        total_tokens = 0
        retry_count = 0
        last_error: Optional[str] = None
        last_sql: Optional[str] = None
        last_response: Optional[LLMResponse] = None
        max_attempts = settings.AI_MAX_RETRIES + 1

        logger.info(
            "ai_service_prompt_built",
            system_prompt_length=len(system_prompt),
            question=question[:100],
        )

        for attempt in range(max_attempts):
            logger.debug(
                "ai_service_attempt",
                attempt=attempt + 1,
                max_attempts=max_attempts,
                provider=provider.provider_name,
            )

            if attempt > 0 and last_error:
                messages.append(
                    LLMMessage(
                        role="assistant",
                        content=last_response.text if last_response else "",
                    )
                )
                messages.append(
                    LLMMessage(
                        role="user",
                        content=(
                            f"The SQL you generated caused this error:\n"
                            f"{last_error}\n\n"
                            "Please fix the SQL and return only the corrected "
                            "SELECT statement."
                        ),
                    )
                )
                retry_count += 1

            llm_response = await provider.complete(
                system_prompt=system_prompt,
                messages=messages,
            )
            last_response = llm_response
            total_tokens += llm_response.total_tokens

            raw_text = llm_response.text.strip()
            logger.debug(
                "ai_service_llm_response",
                attempt=attempt + 1,
                raw_text=raw_text[:200],
                tokens=llm_response.total_tokens,
            )

            if raw_text.upper().startswith("OUT_OF_SCOPE"):
                reason = (
                    raw_text.split(":", 1)[1].strip()
                    if ":" in raw_text
                    else raw_text
                )
                raise InvalidQueryError(question=question, reason=reason)

            sql = self._extract_sql(raw_text)
            if not sql:
                last_error = "No valid SQL statement found in the response."
                logger.warning(
                    "ai_service_no_sql_extracted",
                    attempt=attempt + 1,
                    raw_text=raw_text[:200],
                )
                continue

            if not sql.strip().upper().startswith("SELECT"):
                last_error = f"Only SELECT statements are allowed. Got: {sql[:80]}"
                last_sql = sql
                logger.warning(
                    "ai_service_non_select_sql",
                    attempt=attempt + 1,
                    sql=sql[:80],
                )
                continue

            last_sql = sql
            logger.info(
                "ai_service_sql_generated",
                attempt=attempt + 1,
                sql=sql,
                token_count=total_tokens,
            )

            try:
                rows, _ = await self._repository.execute_raw_sql(sql)
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "ai_service_sql_execution_error",
                    attempt=attempt + 1,
                    sql=sql[:200],
                    error=last_error,
                )
                continue

            answer = self._format_answer(rows)
            result = NLQueryResult(
                answer=answer,
                sql_used=sql,
                rows=rows,
                token_count=total_tokens,
                retry_count=retry_count,
                model_used=llm_response.model or settings.AI_MODEL,
            )

            logger.info(
                "ai_service_success",
                retry_count=retry_count,
                total_tokens=total_tokens,
                row_count=len(rows),
                model=result.model_used,
            )
            return result.to_dict()

        logger.error(
            "ai_service_all_attempts_failed",
            max_attempts=max_attempts,
            last_error=last_error,
        )
        raise SQLGenerationError(
            question=question,
            last_sql=last_sql or "",
            last_error=last_error or "Unknown error",
        )

    @staticmethod
    def _extract_sql(text: str) -> Optional[str]:
        """Extract a SQL SELECT statement from the LLM's raw text response."""
        if not text:
            return None

        fenced = re.search(
            r"```(?:sql)?\s*\n?(.*?)\n?```",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if fenced:
            return fenced.group(1).strip()

        select_match = re.search(
            r"(SELECT\s+.+)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if select_match:
            return select_match.group(1).strip()

        return None

    @staticmethod
    def _format_answer(rows: List[Dict[str, Any]]) -> str:
        """Format raw SQL result rows into a human-readable answer string."""
        if not rows:
            return "No results found."

        if len(rows) == 1:
            row = rows[0]
            if len(row) == 1:
                value = next(iter(row.values()))
                if isinstance(value, float):
                    return f"Result: ${value:,.2f}"
                return f"Result: {value}"
            parts = [f"{k}: {v}" for k, v in row.items()]
            return ", ".join(parts)

        return f"Found {len(rows)} results."
