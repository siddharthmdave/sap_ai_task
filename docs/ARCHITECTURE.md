# Architectural Extension — Scaling to 50 Enterprise Customers

**ETL Order Service — Part 4d**

| | |
|---|---|
| **Version** | 1.0.0 |
| **Date** | 2026-06-25 |
| **Scope** | Written design only — not implemented in this repository |

---

## 1. Current State (Single-Tenant Prototype)

```
CSV → etl.py → SQLite → FastAPI → Clients
                    ↓
              FAISS index (in-process, in-memory)
                    ↓
              LLM API (OpenAI / Anthropic / Ollama)
```

The prototype delivers Parts 1–4 with three AI components:

| Component | Implementation | Location |
|-----------|----------------|----------|
| LLM (NL→SQL) | Configurable provider via factory | `app/services/ai_service.py`, `app/services/llm/` |
| Embedding model | `all-MiniLM-L6-v2` | `app/services/embedding_service.py` |
| Vector index | FAISS `IndexFlatIP` in memory | `app/services/embedding_service.py` |

**Prototype limitations at scale:**

- SQLite — single writer; no tenant isolation
- FAISS — per-process memory; not shared across replicas
- No per-tenant rate limits, audit trails, or data residency routing
- Synchronous index rebuild blocks the embedding swap window (mitigated by atomic reference swap)

---

## 2. Target: 50 Enterprise Tenants with Data Residency

Each tenant requires data to remain in a specific region:

| Region | Example tenants | Infrastructure |
|--------|-----------------|----------------|
| US | 20 tenants | `us-east-1` — cloud LLM + managed vector DB |
| EU | 20 tenants | `eu-west-1` — regional Azure OpenAI or on-premise Llama |
| KSA | 10 tenants | Local cloud — on-premise LLM mandatory; no US/EU data transfer |

**Target topology (per region):**

```
API Gateway (JWT + tenant_id + rate limit)
        │
   FastAPI pods (stateless)
        │
   ┌────┴────┬────────────┬──────────────┐
   ▼         ▼            ▼              ▼
PostgreSQL  Redis      Qdrant       LLM Router
(schema/    (cache +   (per-tenant   (OpenAI /
 tenant)     queue)     collection)   Azure / Ollama)
```

---

## 3. Part 4d Design Decisions

### 3.1 Tenant isolation for the vector index

**Options evaluated:**

| Approach | Memory | Latency | Data leakage risk |
|----------|--------|---------|-------------------|
| **Shared FAISS index + namespace filter** | Lowest (one index) | Fastest query; filter post-search | **High** — filter bugs or prompt injection could cross tenants |
| **One FAISS index per tenant (in-process)** | O(tenants × vectors) — 50 indexes × ~5K vectors ≈ manageable on one host | O(1) index selection by tenant_id | Low per process; **not shared across replicas** |
| **One Qdrant collection per tenant** | Higher ops overhead; persistent storage | REST/gRPC; ~5 ms p99 at 50 collections | **Lowest** — collection is hard boundary; GDPR erasure = `DELETE collection` |

**Decision: One Qdrant collection per tenant** (`orders_{tenant_id}`).

**Trade-off accepted:** Higher memory and operational overhead (50 collections, cluster management) in exchange for zero cross-tenant leakage blast radius and clean regulatory erasure. A shared FAISS index with payload filtering is rejected because a single filter omission exposes all tenant embeddings.

At 50 tenants × ~80K orders × 384 dims × 4 bytes ≈ **6 GB** vector data — fits a 3-node Qdrant cluster with replication.

---

### 3.2 LLM backend per tenant

Some tenants require on-premise Llama; others accept regional Azure OpenAI; dev tenants use OpenAI directly.

**Where routing lives:** A **tenant-aware `LLMRouter`** in the service layer (`app/services/llm/`), above concrete strategies:

```
AIService.answer_question()
    └── LLMRouter.resolve(tenant_id) → LLMProvider instance
            ├── policy: cloud_openai   → OpenAIProvider
            ├── policy: azure_eu       → AzureOpenAIProvider (eu-west endpoint)
            └── policy: on_prem_ollama → OllamaProvider (tenant-specific base URL)
```

**Model-agnostic prompt layer:** `_SYSTEM_PROMPT_TEMPLATE` and `_SCHEMA_DESCRIPTION` remain static strings in `ai_service.py` with no provider-specific tokens. Only `LLMProvider.complete(system_prompt, messages)` is called — the template never references model names or provider SDKs.

Tenant LLM policy is stored in a config service (or ConfigMap per deployment region):

```json
{
  "tenant_id": "acme-eu",
  "llm_policy": "azure_eu",
  "azure_deployment": "gpt-4o-mini-eu",
  "data_residency": "eu-west-1"
}
```

**Trade-off accepted:** Operational complexity of maintaining multiple LLM endpoints per region vs. forcing all tenants onto one cloud API (unacceptable for KSA and strict EU tenants).

---

### 3.3 PII in the NL→SQL pipeline

Order data contains **customer IDs** and **amounts** — both are PII in enterprise contexts.

**Guardrails before question + schema reach the LLM:**

| Guardrail | Cloud third-party API | On-premise LLM |
|-----------|----------------------|----------------|
| Input sanitization (SQL/prompt injection) | Required | Required |
| Customer ID pseudonymization in prompt | **Required** — replace `C001` with `CUST_a3f2` | Optional — data stays in network |
| Schema-only context (no raw rows in prompt) | Required | Required |
| Audit log: tenant_id, question hash, SQL, model | Required | Required |
| DPA / regional endpoint enforcement | Required (Azure EU, etc.) | N/A |
| Row-level result filtering post-SQL | Required — SQL scoped to tenant schema | Required |

**Does the answer change for cloud vs on-premise?**

Yes. For **third-party cloud APIs**, customer IDs should be tokenized before the LLM call and mapped back after SQL execution. Amounts in the question text are lower risk but may be rounded or bucketed for highly regulated tenants. For **on-premise**, the full question and schema can reach the model without masking because inference never leaves the tenant's network boundary — but SQL allow-listing and audit logging still apply.

---

### 3.4 Highest-leverage decision

**Decision:** **Per-tenant isolated vector collections** (Qdrant) over a shared index with namespace filtering.

**Why highest leverage:** This is the only decision that simultaneously affects:

1. **Vector DB** — collection boundary is the isolation primitive
2. **ETL pipeline** — each tenant load writes to `orders_{tenant_id}` collection only
3. **API query path** — `tenant_id` from JWT selects collection before any search
4. **Compliance** — right-to-erasure is a single `DELETE collection`; no orphan vectors in a shared index

**Trade-off accepted:** ~2× infrastructure cost and multi-collection ops vs. a single shared index that would be cheaper but carries catastrophic cross-tenant leakage risk that is difficult to audit and impossible to fully reverse after an incident.

---

## 4. Supporting Scale Decisions (Summary)

| Concern | Decision |
|---------|----------|
| **Relational DB** | PostgreSQL schema-per-tenant (not shared `tenant_id` column) |
| **ETL** | Async Celery workers per tenant; S3 upload → SQS → worker → tenant schema + vector rebuild |
| **Cache** | Redis shared stats cache; invalidation via Pub/Sub after ETL |
| **API** | Stateless FastAPI pods behind regional API gateway with JWT `tenant_id` claim |
| **Observability** | structlog + `tenant_id` on every line; Prometheus metrics per tenant |

### Migration roadmap (8 weeks)

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| 1 | 2 weeks | PostgreSQL, schema-per-tenant, JWT auth |
| 2 | 2 weeks | Redis cache, Celery ETL workers |
| 3 | 2 weeks | Qdrant per-tenant collections |
| 4 | 1 week | Observability dashboards |
| 5 | 1 week | Load test, security review, go-live |

### Cost estimate (50 tenants)

| Component | Monthly (est.) |
|-----------|----------------|
| PostgreSQL (Multi-AZ) | $350 |
| API + ETL workers (K8s) | $240 |
| Redis | $180 |
| Qdrant cluster | $280 |
| LLM API usage | $75 |
| **Total** | **~$1,125 (~$22.50/tenant)** |

---

## 5. References

- Implementation: `app/services/embedding_service.py`, `app/services/ai_service.py`
- AI standards: [`AI_MODEL_STANDARDS.md`](AI_MODEL_STANDARDS.md)
- Setup and API docs: [`../README.md`](../README.md)
