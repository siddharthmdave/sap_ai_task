
# ETL Order Service

Enterprise-grade ETL pipeline and REST API for customer order data, with an optional AI-augmented query layer (natural language → SQL and semantic search).

| Part | Description | Status |
|------|-------------|--------|
| **Part 1** | ETL: Extract CSV → Transform → Load SQLite | Complete |
| **Part 2** | REST API: orders, stats, recent, health | Complete |
| **Part 3** | Docker + Kubernetes manifests | Complete |
| **Part 4a** | `POST /orders/ask` — NL→SQL with retry loop | Complete |
| **Part 4b** | `GET /orders/semantic_search` — FAISS + embeddings | Complete |
| **Part 4c** | LangGraph two-node SQL agent (bonus) | Complete |
| **Part 4d** | Multi-tenant architecture write-up | See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |

**Tech stack:** FastAPI · SQLAlchemy (async) · SQLite · pandas · sentence-transformers · FAISS · LangGraph · structlog

---

## Table of Contents

1. [Architecture](#1-architecture)
2. [Quick Start](#2-quick-start)
3. [Project Structure](#3-project-structure)
4. [Part 1 — ETL Pipeline](#4-part-1--etl-pipeline)
5. [Part 2 — REST API](#5-part-2--rest-api)
6. [Part 3 — Deployment](#6-part-3--deployment)
7. [Part 4 — AI Layer](#7-part-4--ai-layer)
8. [Configuration](#8-configuration)
9. [Security & Observability](#9-security--observability)

---

## 1. Architecture

```
┌─────────────┐     ┌──────────────────────────────────────────────────┐
│   etl.py    │     │              FastAPI (main.py)                    │
│   (CLI)     │     │  Middleware → Routers → Services → Repository    │
└──────┬──────┘     └────────────────────────┬─────────────────────────┘
       │                                     │
       ▼                                     ▼
┌─────────────┐     ┌──────────────┐   ┌─────────────────┐
│ ETLService  │────▶│   SQLite     │   │  FAISS Index    │
│ (transform) │     │  (orders)    │   │  (in-memory)    │
└─────────────┘     └──────────────┘   └─────────────────┘
                                              ▲
                                     ┌────────┴────────┐
                                     │ EmbeddingService │
                                     │ all-MiniLM-L6-v2 │
                                     └─────────────────┘

AI query path:  POST /orders/ask → AIService → LLM (via factory) → SQL → SQLite
Semantic path:  GET /orders/semantic_search → EmbeddingService → FAISS
```

**Layering (MVC-style):**

| Layer | Location | Responsibility |
|-------|----------|----------------|
| Controller | `app/api/v1/endpoints/` | HTTP routing, request validation |
| Service | `app/services/` | Business logic, AI orchestration |
| Repository | `app/repositories/` | Database access |
| Model | `app/models/` | SQLAlchemy ORM |
| Schema | `app/schemas/` | Pydantic request/response models |

Further design detail: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · AI standards: [`docs/AI_MODEL_STANDARDS.md`](docs/AI_MODEL_STANDARDS.md)

---

## 2. Quick Start

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | **3.11.x or 3.12.x** (3.13+ not recommended — `faiss-cpu`, `torch` wheels may be unavailable) |
| pip | 23+ |

### Installation

```bash
# 1. Enter project directory
cd ETL_TASK

# 2. Create virtual environment
python3.11 -m venv .venv
# If ensurepip fails on 3.12, try:
#   python3.11 -m venv .venv
#   or: python3 -m venv .venv --without-pip && curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python

# 3. Activate
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 4. Upgrade pip and install dependencies
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 5. Configure environment
cp .env.example .env                 # edit OPENAI_API_KEY for AI endpoints

# 6. Load sample data
python etl.py load data/orders.csv

# 7. Start API
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Verify

```bash
curl http://localhost:8000/healthz
curl http://localhost:8000/orders/stats
curl http://localhost:8000/orders/customer/C123
open http://localhost:8000/docs
```

### Docker

```bash
docker-compose up --build
# or
docker build -t etl-service:latest .
docker run -p 8000:8000 --env-file .env -v $(pwd)/data:/app/data etl-service:latest
```

---

## 3. Project Structure

```
ETL_TASK/
├── main.py                          # FastAPI app factory + lifespan
├── etl.py                           # CLI: load, show-stats, validate
├── requirements.txt
├── Dockerfile                       # Multi-stage, non-root, healthcheck
├── docker-compose.yml
├── .env.example
│
├── app/
│   ├── api/v1/endpoints/
│   │   ├── health.py                # /healthz, /readyz
│   │   └── orders.py                # Order + AI endpoints
│   ├── core/
│   │   ├── config.py                # Pydantic settings
│   │   ├── database.py              # Async SQLAlchemy engine
│   │   ├── exceptions.py            # Domain exceptions
│   │   └── logging.py               # structlog
│   ├── middleware/
│   │   └── request_middleware.py    # Request ID, logging, security headers
│   ├── models/
│   │   └── order.py                 # ORM model (source of truth for schema)
│   ├── repositories/
│   │   └── order_repository.py      # All SQL / upsert / raw SQL execution
│   ├── schemas/
│   │   ├── common.py                # Response envelopes, health schema
│   │   └── order.py                 # Order, stats, NL query, semantic search
│   ├── services/
│   │   ├── etl_service.py           # ETL transform + load
│   │   ├── order_service.py         # Order queries + stats cache
│   │   ├── ai_service.py            # NL→SQL orchestrator
│   │   ├── embedding_service.py     # FAISS index + semantic search
│   │   ├── error_handler.py         # Global exception handlers
│   │   └── llm/
│   │       ├── base.py              # LLMProvider interface
│   │       ├── strategies.py        # OpenAI, Anthropic, Azure, Ollama
│   │       ├── factory.py           # get_llm_provider()
│   │       └── graph.py             # LangGraph SQL agent (Part 4c)
│   └── utils/
│       └── embedding.py             # Order → embedding text helper
│
├── data/
│   └── orders.csv                   # Sample input
├── docs/
│   ├── AI_MODEL_STANDARDS.md
│   └── ARCHITECTURE.md
└── k8s/
    ├── configmap.yaml
    ├── deployment.yaml
    ├── service.yaml
    └── secret.yaml.example
```

---

## 4. Part 1 — ETL Pipeline

### Run

```bash
python etl.py load data/orders.csv
python etl.py show-stats
python etl.py validate data/orders.csv
```

### Transform rules

| Field | Rule |
|-------|------|
| `order_id` | Required — row dropped if missing |
| `customer_id` | Required — row dropped if missing |
| `order_date` | Normalized to `YYYY-MM-DD`; unparseable dates dropped |
| `amount` | Invalid/missing → `0.0` |
| `currency` | Missing/unknown → `USD` |
| `amount_usd` | `amount × rate` (EUR = 1.1 USD, USD = 1.0) |

### Supported date formats

`YYYY-MM-DD` · `MM/DD/YYYY` · `DD-MM-YYYY` · `DD/MM/YYYY` · `MM-DD-YYYY` · `YYYY/MM/DD`

Data is stored in SQLite table `orders` (see `app/models/order.py`).

---

## 5. Part 2 — REST API

Both **spec paths** and **versioned paths** are registered:

| Method | Spec path | Versioned path |
|--------|-----------|----------------|
| `GET` | `/healthz` | `/healthz` |
| `GET` | `/readyz` | `/readyz` |
| `GET` | `/orders/customer/{customer_id}` | `/api/v1/orders/customer/{customer_id}` |
| `GET` | `/orders/stats` | `/api/v1/orders/stats` |
| `GET` | `/orders/recent?days=N` | `/api/v1/orders/recent?days=N` |
| `POST` | `/orders/ask` | `/api/v1/orders/ask` |
| `GET` | `/orders/semantic_search` | `/api/v1/orders/semantic_search` |

### Stats response example

```json
{
  "success": true,
  "data": {
    "total_revenue": 12345.67,
    "avg_order_value": 87.5,
    "order_count": 150,
    "orders_per_day": {
      "2020-01-01": 15,
      "2020-01-02": 20
    },
    "currency_breakdown": { "USD": 120, "EUR": 30 },
    "computed_at": "2026-06-25T10:00:00Z"
  },
  "meta": { "request_id": "...", "timestamp": "...", "version": "v1" }
}
```

All success responses use the `SuccessResponse` envelope; errors use `ErrorResponse` with `error_code` and `message`.

---

## 6. Part 3 — Deployment

### Docker

| Requirement | Implementation |
|-------------|----------------|
| Multi-stage build | `builder` → `runtime` in `Dockerfile` |
| Non-root user | `appuser` (UID/GID 1001) |
| Expose port | `8000` |
| Healthcheck | `curl -f http://localhost:8000/healthz` |

### Kubernetes

```bash
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/service.yaml
kubectl create secret generic etl-service-secrets \
  --from-literal=SECRET_KEY=$(openssl rand -hex 32) \
  --from-literal=OPENAI_API_KEY="sk-..."
kubectl apply -f k8s/deployment.yaml
```

Manifests include liveness (`/healthz`) and readiness (`/readyz`) probes and a ConfigMap for `DATABASE_URL`, `APP_ENV`, etc.

---

## 7. Part 4 — AI Layer

### 7a. Natural Language Query — `POST /orders/ask`

**Request:**

```http
POST /orders/ask
Content-Type: application/json

{"question": "What is the total revenue from customer C001 in the last 30 days?"}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "answer": "Result: $4,230.00",
    "sql_used": "SELECT SUM(amount_usd) AS total_revenue FROM orders WHERE customer_id = 'C001' AND order_date >= date('now', '-30 days')",
    "rows": [{"total_revenue": 4230.0}],
    "token_count": 385,
    "retry_count": 1,
    "model_used": "gpt-4o-mini"
  }
}
```

#### LLM choice: OpenAI `gpt-4o-mini`

| Criterion | Rationale |
|-----------|-----------|
| SQL accuracy | Strong on single-table schemas with schema-in-prompt |
| Cost | ~10× cheaper than GPT-4o for equivalent SQL tasks |
| Latency | ~500 ms median — suitable for interactive queries |
| Determinism | `temperature=0.0` gives consistent SQL |
| Portability | Same `LLMProvider` interface supports Anthropic, Azure OpenAI, Ollama |

Switch provider via `.env`: `AI_PROVIDER=anthropic|azure_openai|ollama`

#### System prompt template

Defined in `app/services/ai_service.py` as `_SYSTEM_PROMPT_TEMPLATE`:

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

The `{schema}` placeholder is filled at runtime from `_SCHEMA_DESCRIPTION` (kept in sync with `app/models/order.py`).

#### Retry loop example

**Question:** "What is the total revenue from customer C001?"

| Attempt | SQL | Result |
|---------|-----|--------|
| 1 | `SELECT SUM(amount) FROM orders WHERE customer_id = 'C001'` | Error: `no such column: amount` |
| 2 | `SELECT SUM(amount_usd) AS total_revenue FROM orders WHERE customer_id = 'C001'` | Success → `$4,230.00` |

The error from attempt 1 is appended to the conversation; the LLM self-corrects on attempt 2. Max retries: `AI_MAX_RETRIES` (default 2).

Out-of-scope questions return **400** with `INVALID_QUERY`. Prompt, SQL, and token count are logged on every request.

Full AI configuration standards: [`docs/AI_MODEL_STANDARDS.md`](docs/AI_MODEL_STANDARDS.md)

---

### 7b. Semantic Search — `GET /orders/semantic_search`

```http
GET /orders/semantic_search?q=high+value+recent+orders&top_k=5
```

```json
{
  "success": true,
  "data": {
    "results": [
      {
        "order_id": "1001",
        "customer_id": "C001",
        "amount_usd": 320.0,
        "order_date": "2024-03-15",
        "currency": "USD",
        "score": 0.91
      }
    ],
    "query": "high value recent orders",
    "top_k": 5,
    "index_size": 5000
  }
}
```

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Embedding model | `all-MiniLM-L6-v2` | 384-dim, ~80 MB, fast on CPU, strong on short structured text |
| Index | FAISS `IndexFlatIP` + L2-normalized vectors | Exact cosine similarity; <1 ms for ~5K vectors |
| Record format | `"customer C001, $320.00 USD, 2024-03-15"` | See `Order.to_embedding_text()` |

**Index rebuild:** `etl.py load` rebuilds the index after a successful load. `main.py` lifespan also builds the index on API startup from existing DB data. Rebuild is synchronous and atomically swaps the in-memory index reference — in-flight searches continue using the previous index until the swap completes. Multi-worker deployments would need a shared vector store (see architecture doc).

---

### 7c. LangGraph SQL Agent (bonus)

Implemented in `app/services/llm/graph.py`. `AIService` uses LangGraph when installed; otherwise falls back to an identical manual retry loop.

```
START → sql_writer → sql_executor ──success──→ END
              ↑            │
              └── retry ───┘  (up to AI_MAX_RETRIES)
```

**Trace example:**

1. **sql_writer** → `SELECT SUM(amount) ...` (wrong column)
2. **sql_executor** → `no such column: amount` → route `retry`
3. **sql_writer** → `SELECT SUM(amount_usd) ...` (corrected)
4. **sql_executor** → success → `END`

---

### 7d. Scaling to 50 enterprise customers

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full Part 4d write-up covering:

- Vector index tenant isolation (per-tenant collection vs shared index)
- Per-tenant LLM routing (cloud vs on-premise)
- PII guardrails in the NL→SQL pipeline
- Highest-leverage architectural decision and accepted trade-offs

---

## 8. Configuration

Copy `.env.example` to `.env`. Key variables:

```bash
# Application
APP_ENV=development          # development | staging | production
SECRET_KEY=change-me-in-prod

# Database
DATABASE_URL=sqlite+aiosqlite:///./data/orders.db

# AI / LLM
AI_PROVIDER=openai           # openai | anthropic | azure_openai | ollama
AI_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
AI_MAX_RETRIES=2
AI_TEMPERATURE=0.0

# Embeddings
EMBEDDING_MODEL=all-MiniLM-L6-v2
FAISS_INDEX_PATH=./data/faiss_index

# Cache & metrics
CACHE_TTL_SECONDS=60
ENABLE_METRICS=true
METRICS_PATH=/metrics
```

---

## 9. Security & Observability

| Control | Implementation |
|---------|----------------|
| SQL injection | `NLQueryRequest` blocks dangerous patterns; repository allows `SELECT` only |
| Prompt injection | Pattern matching on NL query input |
| Request tracing | `X-Request-ID` on every request/response |
| Security headers | `X-Frame-Options`, CSP, `X-Content-Type-Options` |
| Structured logging | structlog JSON (production) / console (development) |
| Metrics | Prometheus at `/metrics` when `ENABLE_METRICS=true` |
| Health | `GET /healthz` (liveness), `GET /readyz` (readiness + DB check) |
| Container | Non-root user, multi-stage Docker build |

---

## License

MIT
