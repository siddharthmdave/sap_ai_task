# app/services/llm/graph.py
"""
LangGraph SQL Agent - Two-Node Graph Implementation (Part 4c Bonus).

Replaces the manual retry loop in AIService with an explicit LangGraph
state machine. The graph has two nodes:

    sql_writer  - calls the LLM to generate SQL from the question + schema
    sql_executor - executes the SQL; on failure routes back to sql_writer
                   with the error appended (up to AI_MAX_RETRIES retries)

Graph topology:

    START
      |
      v
    sql_writer -----------------------------
      |                                     |
      v                                     |
    sql_executor                            |
      |                                     |
      |--- success ------------------------>| END
      |                                     |
      |--- error + retries_remaining > 0 --|
      |                                     |
      |--- error + retries_remaining == 0 -> END (raises)

Usage:
    from app.services.llm.graph import build_sql_agent, run_sql_agent

    graph = build_sql_agent()
    result = await run_sql_agent(graph, question="...", repository=repo)

The graph is compiled once at module level and reused across requests
(LangGraph compiled graphs are stateless and thread-safe).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app.core.logging import get_logger

logger = get_logger(__name__)

# — State Definition ——————————————————————————————————————

class SQLAgentState(dict):
    """
    Typed state dictionary passed between LangGraph nodes.

    Fields:
        question        - original natural language question
        schema          - database schema string injected into system prompt
        system_prompt   - fully rendered system prompt (schema injected)
        messages        - conversation history (user/assistant turns)
        sql             - most recently generated SQL (or None)
        rows            - query result rows on success (or None)
        error           - last SQL execution error message (or None)
        retry_count     - number of correction retries performed so far
        total_tokens    - cumulative LLM token count across all attempts
        model_used      - LLM model identifier from the last response
        success         - True when SQL executed successfully
        out_of_scope    - True when LLM returned OUT_OF_SCOPE sentinel
        out_of_scope_reason - reason string from OUT_OF_SCOPE response
    """

def _initial_state(
    question: str,
    schema: str,
    system_prompt: str,
) -> SQLAgentState:
    """Build the initial state for a new agent run."""
    return SQLAgentState(
        question=question,
        schema=schema,
        system_prompt=system_prompt,
        messages=[{"role": "user", "content": question}],
        sql=None,
        rows=None,
        error=None,
        retry_count=0,
        total_tokens=0,
        model_used="",
        success=False,
        out_of_scope=False,
        out_of_scope_reason="",
    )

# — Node: sql_writer ——————————————————————————————————————

async def sql_writer_node(state: SQLAgentState) -> SQLAgentState:
    """
    Node 1 - sql_writer.

    Calls the configured LLM provider with the current conversation history
    and extracts a SQL SELECT statement from the response.

    On retry (state["retry_count"] > 0), the previous error has already been
    appended to state["messages"] by the sql_executor node, so the LLM
    receives the full error context automatically.

    State mutations:
        - Appends assistant turn to messages (the raw LLM response)
        - Sets state["sql"] to the extracted SQL (or None)
        - Accumulates state["total_tokens"]
        - Sets state["model_used"]
        - Sets state["out_of_scope"] / state["out_of_scope_reason"] if applicable
    """
    from app.services.llm.factory import get_llm_provider
    from app.services.llm.base import LLMMessage

    attempt = state["retry_count"] + 1
    logger.debug(
        "langgraph_sql_writer",
        attempt=attempt,
        message_count=len(state["messages"]),
    )

    provider = get_llm_provider()

    # Convert plain dicts back to LLMMessage objects
    llm_messages = [
        LLMMessage(role=m["role"], content=m["content"])
        for m in state["messages"]
    ]

    llm_response = await provider.complete(
        system_prompt=state["system_prompt"],
        messages=llm_messages,
    )

    raw_text = llm_response.text.strip()
    state["total_tokens"] = state["total_tokens"] + llm_response.total_tokens
    state["model_used"] = llm_response.model or ""

    logger.debug(
        "langgraph_sql_writer_response",
        attempt=attempt,
        raw_text=raw_text[:200],
        tokens=llm_response.total_tokens,
    )

    # Append assistant turn to conversation history
    state["messages"] = state["messages"] + [
        {"role": "assistant", "content": raw_text}
    ]

    # Check for OUT_OF_SCOPE sentinel
    if raw_text.upper().startswith("OUT_OF_SCOPE"):
        reason = raw_text.split(":", 1)[1].strip() if ":" in raw_text else raw_text
        state["out_of_scope"] = True
        state["out_of_scope_reason"] = reason
        state["sql"] = None
        return state

    # Extract SQL from response
    sql = _extract_sql(raw_text)
    state["sql"] = sql

    if not sql:
        state["error"] = "No valid SQL statement found in the LLM response."
        logger.warning(
            "langgraph_sql_writer_no_sql",
            attempt=attempt,
            raw_text=raw_text[:200],
        )
    elif not sql.strip().upper().startswith("SELECT"):
        state["error"] = f"Only SELECT statements are allowed. Got: {sql[:80]}"
        state["sql"] = None
        logger.warning(
            "langgraph_sql_writer_non_select",
            attempt=attempt,
            sql=sql[:80],
        )
    else:
        # Clear previous error - new SQL is ready for execution
        state["error"] = None

    return state

# — Node: sql_executor ——————————————————————————————————————

async def sql_executor_node(state: SQLAgentState) -> SQLAgentState:
    """
    Node 2 - sql_executor.

    Executes the SQL in state["sql"] against the database via the repository
    that was injected into the state before the graph was invoked.

    On success:
        - Sets state["rows"] and state["success"] = True

    On failure:
        - Sets state["error"] with the exception message
        - Appends a user turn to messages asking the LLM to fix the SQL
        - Increments state["retry_count"]

    State mutations:
        - state["rows"], state["success"] on success
        - state["error"], state["messages"], state["retry_count"] on failure
    """
    from app.core.config import settings

    # If sql_writer produced no SQL (or OUT_OF_SCOPE), skip execution
    if state.get("out_of_scope") or not state.get("sql"):
        return state

    sql = state["sql"]
    repository = state["_repository"]  # injected before graph.invoke()
    attempt = state["retry_count"] + 1

    logger.debug(
        "langgraph_sql_executor",
        attempt=attempt,
        sql=sql[:200],
    )

    try:
        rows, _ = await repository.execute_raw_sql(sql)
        state["rows"] = rows
        state["success"] = True
        state["error"] = None

        logger.info(
            "langgraph_sql_executor_success",
            attempt=attempt,
            row_count=len(rows),
            total_tokens=state["total_tokens"],
            retry_count=state["retry_count"],
        )

    except Exception as exc:
        error_msg = str(exc)
        state["error"] = error_msg
        state["success"] = False

        logger.warning(
            "langgraph_sql_executor_error",
            attempt=attempt,
            sql=sql[:80],
            error=error_msg,
        )

        # Append error context as a new user turn so sql_writer can self-correct
        state["messages"] = state["messages"] + [
            {
                "role": "user",
                "content": (
                    f"The SQL you generated caused this error:\n"
                    f"  {error_msg}\n\n"
                    f"Please fix the SQL and return only the corrected "
                    f"SELECT statement."
                ),
            }
        ]
        state["retry_count"] = state["retry_count"] + 1

    return state

# — Routing Function ——————————————————————————————————————

def route_after_executor(state: SQLAgentState) -> str:
    """
    Conditional edge function called after sql_executor.

    Returns:
        "done"  - if SQL succeeded, or OUT_OF_SCOPE, or retries exhausted
        "retry" - if SQL failed and retries remain
    """
    from app.core.config import settings

    if state.get("success"):
        return "done"

    if state.get("out_of_scope"):
        return "done"

    max_attempts = settings.AI_MAX_RETRIES + 1
    if state["retry_count"] >= max_attempts:
        return "done"  # exhausted - caller will raise SQLGenerationError

    return "retry"

# — Graph Builder ——————————————————————————————————————

def build_sql_agent():
    """
    Build and compile the two-node LangGraph SQL agent.

    Graph definition:

        START → sql_writer → sql_executor
                      ↑           |
                      | retry     | success / exhausted
                      └───────────┘
                                 ↓
                                END

    The compiled graph is stateless and can be reused across requests.

    Returns:
        Compiled LangGraph StateGraph (CompiledGraph).
    """
    try:
        from langgraph.graph import StateGraph, END
    except ImportError as exc:
        raise ImportError(
            "langgraph is required for the SQL agent graph. "
            "Install it with: pip install langgraph"
        ) from exc

    graph = StateGraph(SQLAgentState)

    # Register nodes
    graph.add_node("sql_writer", sql_writer_node)
    graph.add_node("sql_executor", sql_executor_node)

    # Entry point
    graph.set_entry_point("sql_writer")

    # sql_writer always flows to sql_executor
    graph.add_edge("sql_writer", "sql_executor")

    # sql_executor routes conditionally
    graph.add_conditional_edges(
        "sql_executor",
        route_after_executor,
        {
            "retry": "sql_writer",  # error + retries remaining → back to writer
            "done": END,            # success or exhausted → end
        },
    )

    return graph.compile()

# — Module-level compiled graph (singleton) ——————————————————————————

# Compiled once at import time; reused across all requests.
# Falls back to None if langgraph is not installed (AIService will use
# the manual loop fallback in that case).
try:
    _SQL_AGENT_GRAPH = build_sql_agent()
    logger.debug("langgraph_sql_agent_compiled")
except ImportError:
    _SQL_AGENT_GRAPH = None
    logger.warning(
        "langgraph_not_installed",
        note="SQL agent will use manual retry loop. "
             "Install langgraph for graph-based execution.",
    )

# — Public Runner ——————————————————————————————————————

async def run_sql_agent(
    question: str,
    system_prompt: str,
    schema: str,
    repository: Any,
) -> Dict[str, Any]:
    """
    Run the LangGraph SQL agent for a single question.

    This is the primary entry point called by AIService. It:
        1. Builds the initial state
        2. Injects the repository (needed by sql_executor_node)
        3. Invokes the compiled graph
        4. Extracts and returns the result

    Args:
        question:        Validated natural language question.
        system_prompt:   Fully rendered system prompt with schema.
        schema:          Schema description string (for state context).
        repository:      OrderRepository bound to the current DB session.

    Returns:
        Dict with keys: answer, sql_used, rows, total_tokens, retry_count, model_used

    Raises:
        InvalidQueryError:     If LLM returned OUT_OF_SCOPE.
        SQLGenerationError:    If all retries exhausted without valid SQL.
        ImportError:           If langgraph is not installed.
    """
    from app.core.exceptions import InvalidQueryError, SQLGenerationError

    if _SQL_AGENT_GRAPH is None:
        raise ImportError(
            "langgraph is not installed. Run: pip install langgraph"
        )

    # Build initial state and inject repository
    state = _initial_state(question=question, schema=schema, system_prompt=system_prompt)
    state["_repository"] = repository  # injected dependency (not part of typed state)

    logger.info(
        "langgraph_agent_start",
        question=question[:100],
    )

    # Invoke the graph - runs until END node is reached
    final_state = await _SQL_AGENT_GRAPH.ainvoke(state)

    logger.info(
        "langgraph_agent_complete",
        success=final_state.get("success"),
        retry_count=final_state.get("retry_count", 0),
        total_tokens=final_state.get("total_tokens", 0),
    )

    # — Handle OUT_OF_SCOPE ——————————————————————————————————————
    if final_state.get("out_of_scope"):
        raise InvalidQueryError(
            question=question,
            reason=final_state.get("out_of_scope_reason", "Question out of scope"),
        )

    # — Handle exhausted retries ——————————————————————————————————————
    if not final_state.get("success"):
        raise SQLGenerationError(
            question=question,
            last_sql=final_state.get("sql") or "",
            last_error=final_state.get("error") or "Unknown error",
        )

    # — Format answer ——————————————————————————————————————
    rows = final_state.get("rows") or []
    answer = _format_answer(rows)

    return {
        "answer": answer,
        "sql_used": final_state.get("sql", ""),
        "rows": rows,
        "token_count": final_state.get("total_tokens", 0),
        "retry_count": final_state.get("retry_count", 0),
        "model_used": final_state.get("model_used", ""),
    }

# — Helpers ——————————————————————————————————————

def _extract_sql(text: str) -> Optional[str]:
    """Extract a SQL SELECT statement from raw LLM text."""
    if not text:
        return None

    # Strip markdown code fences
    fenced = re.search(r"```(?:sql)?\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    # Bare SELECT statement
    select_match = re.search(r"(SELECT\s+.+)", text, re.DOTALL | re.IGNORECASE)
    if select_match:
        return select_match.group(1).strip()

    return None

def _format_answer(rows: List[Dict[str, Any]]) -> str:
    """Format SQL result rows into a human-readable answer string."""
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