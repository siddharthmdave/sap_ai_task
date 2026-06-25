# AI Model Configuration Standards

**ETL Order Service — AI/LLM Integration Guide**

| | |
|---|---|
| **Version** | 1.0.0 |
| **Last updated** | 2026-06-25 |
| **Scope** | NL→SQL (`POST /orders/ask`), semantic search (`GET /orders/semantic_search`) |

---

## Table of Contents

1. [Overview](#1-overview)
2. [Provider Selection](#2-provider-selection)
3. [Configuration Parameters](#3-configuration-parameters)
4. [System Prompt Standard](#4-system-prompt-standard)
5. [Retry Loop](#5-retry-loop)
6. [Embedding Model Standards](#6-embedding-model-standards)
7. [Security and PII Guardrails](#7-security-and-pii-guardrails)
8. [Observability](#8-observability)
9. [Strategy + Factory Architecture](#9-strategy--factory-architecture)
10. [Model Upgrade Checklist](#10-model-upgrade-checklist)

---

## 1. Overview

This document defines standards for all AI integrations in the ETL Order Service:

- **NL→SQL** — `AIService` converts natural language to SQLite `SELECT` queries via a configurable LLM
- **Semantic search** — `EmbeddingService` encodes orders and queries with `sentence-transformers` + FAISS

All provider-specific code is isolated under `app/services/llm/`. The rest of the application depends only on the abstract `LLMProvider` interface.

---

## 2. Provider Selection

### 2.1 Default: OpenAI `gpt-4o-mini`

| Criterion | Rationale |
|-----------|-----------|
| **SQL accuracy** | Reliable on single-table schemas when schema is injected in the system prompt |
| **Cost** | ~$0.15/1M input tokens — economical for high query volume |
| **Latency** | ~500 ms median — acceptable for interactive use |
| **Context window** | 128K tokens — sufficient for schema + question + retry history |
| **Determinism** | `temperature=0.0` produces consistent SQL for identical questions |
| **Compatibility** | OpenAI SDK also powers Azure OpenAI and Ollama (OpenAI-compatible endpoint) |

### 2.2 Supported alternatives

| Provider | Env value | Typical model | Use case |
|----------|-----------|---------------|----------|
| OpenAI | `openai` | `gpt-4o-mini` | Default; development and production |
| Anthropic | `anthropic` | `claude-3-5-sonnet-20241022` | Strong reasoning; alternative cloud API |
| Azure OpenAI | `azure_openai` | deployment name | Data residency (EU / US / KSA) |
| Ollama | `ollama` | `llama3` | On-premise / air-gapped |

Switch provider by changing `AI_PROVIDER` in `.env` — no code changes required.

---

## 3. Configuration Parameters

All AI settings are environment-driven (see `.env.example` and `app/core/config.py`).

```bash
# Provider
AI_PROVIDER=openai
AI_MODEL=gpt-4o-mini

# Generation
AI_TEMPERATURE=0.0        # Must be 0.0 for deterministic SQL
AI_MAX_TOKENS=1024
AI_TIMEOUT_SECONDS=30
AI_MAX_RETRIES=2          # SQL correction attempts after execution failure

# Provider credentials
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3

# Embeddings
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=64
```

### Parameter rationale

| Parameter | Value | Why |
|-----------|-------|-----|
| `AI_TEMPERATURE` | `0.0` | Non-zero temperature adds randomness — undesirable for SQL generation |
| `AI_MAX_TOKENS` | `1024` | Single-table queries rarely exceed 200 tokens; headroom for complex aggregations |
| `AI_MAX_RETRIES` | `2` | One retry fixes most column-name errors; more retries increase cost without proportional gain |
| `AI_TIMEOUT_SECONDS` | `30` | 10× headroom over P99 LLM latency |

---

## 4. System Prompt Standard

**Source of truth:** `app/services/ai_service.py` — constants `_SCHEMA_DESCRIPTION` and `_SYSTEM_PROMPT_TEMPLATE`.

The schema block is injected at runtime and must stay aligned with `app/models/order.py`.

### Template

```
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
```

### Design principles

| Principle | Implementation |
|-----------|----------------|
| Schema in sync with ORM | `_SCHEMA_DESCRIPTION` mirrors `Order` model columns |
| Out-of-scope detection | `OUT_OF_SCOPE:` sentinel → HTTP 400 `INVALID_QUERY` |
| Defense in depth | Prompt forbids DDL/DML; repository enforces `SELECT`-only |
| Dialect hints | SQLite `date('now', '-N days')` encouraged in AI_MODEL_STANDARDS examples |
| Provider agnostic | Same template used by all `LLMProvider` strategies and LangGraph agent |

---

## 5. Retry Loop

### Execution modes

1. **Primary (Part 4c):** LangGraph agent in `app/services/llm/graph.py`
   - Node `sql_writer` — calls LLM, extracts SQL
   - Node `sql_executor` — runs SQL; on failure routes back to `sql_writer` with error appended
2. **Fallback:** Manual loop in `AIService._manual_retry_loop()` — identical semantics when LangGraph is unavailable

Max attempts = `AI_MAX_RETRIES + 1` (default: 3 total attempts).

### Example trace — wrong column name

**Question:** "What is the total revenue from customer C001 in the last 30 days?"

**Attempt 1**

```
LLM output:
  SELECT SUM(amount) FROM orders
  WHERE customer_id = 'C001'
  AND order_date >= date('now', '-30 days')

SQL error: no such column: amount
```

**Attempt 2** (error appended to conversation)

```
LLM output:
  SELECT SUM(amount_usd) AS total_revenue FROM orders
  WHERE customer_id = 'C001'
  AND order_date >= date('now', '-30 days')

Result: $4,230.00
retry_count: 1 | token_count: ~400
```

---

## 6. Embedding Model Standards

### 6.1 Chosen model: `all-MiniLM-L6-v2`

| Property | Value |
|----------|-------|
| Dimensions | 384 |
| Model size | ~80 MB |
| Inference speed | ~14,000 sentences/sec on CPU |
| MTEB (short text) | 56.26 |
| License | Apache 2.0 |

**Why it suits order records:** Trained on 1B+ sentence pairs; captures semantic similarity in short structured strings such as:

```
customer C001, $320.00 USD, 2024-03-15
```

Queries like `"high value recent orders"` surface records with large `amount_usd` and recent `order_date` without exact keyword matching.

### 6.2 Alternatives considered

| Model | Dim | Size | Not chosen because |
|-------|-----|------|-------------------|
| `all-mpnet-base-v2` | 768 | 420 MB | 5× larger; marginal gain for structured text |
| `paraphrase-MiniLM-L3-v2` | 384 | 61 MB | Lower accuracy |
| `text-embedding-3-small` (OpenAI) | 1536 | API | Per-query API latency and cost |

### 6.3 FAISS index

| Setting | Value |
|---------|-------|
| Index type | `IndexFlatIP` (inner product on L2-normalized vectors = cosine similarity) |
| Storage | In-memory (per process) |
| Rebuild trigger | `etl.py load` completion + API startup (`main.py` lifespan) |
| Concurrency | Atomic index swap under `threading.RLock`; searches hold a snapshot of the previous index during rebuild |

| Trade-off | `IndexFlatIP` | `IndexIVFFlat` |
|-----------|---------------|----------------|
| Accuracy | Exact (100%) | Approximate (~95%) |
| Query time | O(n) | O(√n) |
| Training | None | Requires training pass |
| Suitable for | < 100K vectors | > 100K vectors |

At ~5,000 orders, `IndexFlatIP` completes in < 1 ms on CPU.

---

## 7. Security and PII Guardrails

### 7.1 Pre-LLM validation

**Layer 1 — Pydantic (`NLQueryRequest.sanitize_question`):**

- Blocks SQL injection patterns (`DROP`, `DELETE`, `UNION SELECT`, `--`, etc.)
- Blocks prompt injection patterns (`ignore previous instructions`, `jailbreak`, etc.)
- Enforces 5–500 character length

**Layer 2 — Service (`AIService`):**

- Detects `OUT_OF_SCOPE:` from LLM before execution
- Returns HTTP 400 without running SQL

### 7.2 Post-LLM validation

| Control | Location |
|---------|----------|
| SELECT-only enforcement | `OrderRepository.execute_raw_sql()` |
| SQL type check | `AIService._extract_sql()` + repository guard |
| Result bounding | Repository returns all rows (consider `LIMIT` for production) |

### 7.3 Cloud API vs on-premise

| Concern | Cloud API (OpenAI / Anthropic) | On-premise (Ollama / private Llama) |
|---------|-------------------------------|-------------------------------------|
| PII in prompts | Customer IDs and amounts leave the network | Data stays within tenant boundary |
| Data residency | Depends on provider region / DPA | Fully controlled |
| Mitigation | Tokenize customer IDs; use Azure with regional endpoint | No additional masking required |
| Recommended for | Dev, non-regulated workloads | GDPR, HIPAA, KSA PDPL |

**Example PII masking (cloud path):**

```python
# Pseudonymize customer IDs before sending to external LLM
masked = question.replace(customer_id, f"CUST_{hash(customer_id) % 10000:04d}")
```

---

## 8. Observability

Every NL query logs (via structlog in `AIService`):

| Field | Description |
|-------|-------------|
| `question` | Sanitized user question (truncated) |
| `sql` | Generated SQL on success |
| `token_count` | Cumulative tokens across retry attempts |
| `retry_count` | Number of correction retries |
| `model_used` | LLM model identifier |
| `row_count` | Result set size |

**Suggested alerting thresholds:**

| Condition | Severity |
|-----------|----------|
| `retry_count > 0` | Warning — SQL needed correction |
| `token_count > 2000` | Warning — unusually large prompt |
| `duration_ms > 10000` | Error — timeout risk |

Semantic search logs `semantic_search_complete` with `query`, `top_k`, and `results_returned`.

---

## 9. Strategy + Factory Architecture

```
app/services/llm/
├── base.py          LLMProvider (abstract) + LLMMessage + LLMResponse
├── strategies.py    OpenAIProvider, AnthropicProvider, AzureOpenAIProvider, OllamaProvider
├── factory.py       get_llm_provider() — reads AI_PROVIDER from settings
└── graph.py         LangGraph sql_writer → sql_executor agent

app/services/ai_service.py
└── AIService.answer_question()
      ├── provider = get_llm_provider()     # factory
      └── response = await provider.complete(...)  # strategy
```

### Adding a new provider

1. Add enum value to `AIProvider` in `app/core/config.py`
2. Create `XxxProvider(LLMProvider)` in `strategies.py`
3. Register in `_build_registry()` in `factory.py`
4. Add env vars to `.env.example`

No changes required in `AIService`, API endpoints, or routers.

### Runtime provider switch

```bash
AI_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
AI_MODEL=claude-3-5-sonnet-20241022
```

---

## 10. Model Upgrade Checklist

When changing `AI_MODEL` or embedding model:

- [ ] Update `AI_MODEL` / `EMBEDDING_MODEL` in `.env` and `k8s/configmap.yaml`
- [ ] Verify retry loop on intentionally bad SQL (wrong column name)
- [ ] Rebuild FAISS index after embedding model change (`python etl.py load data/orders.csv`)
- [ ] Compare token usage and latency
- [ ] Update cost estimates in this document
- [ ] Update README API examples if response format changes
