# Design Pack — Hybrid RAG + Text-to-SQL Q&A Assistant

This document walks through the architecture, the key design decisions and their
trade-offs, the current limitations, and how the prototype would evolve into a
production-ready enterprise system.

---

## 1. Problem

Build an assistant that answers natural-language questions grounded in **two very
different kinds of knowledge**:

- **Unstructured** policy documents (returns, delivery, click & collect, promotions, product care, warranty).
- **Structured** relational data (products, orders, inventory, promotions, stores).

…and that **clearly attributes its sources**, decides **where** to answer from, and
behaves reliably and securely.

---

## 2. Architecture overview

The system is a **LangGraph state machine**. A shared state flows through nodes; the
router decides the path via conditional edges.

```
START → router → ┬─ smalltalk ─────────────→ END
                 ├─ documents → docs ──────→ finalize → END
                 ├─ database  → sql ───────→ finalize → END
                 └─ both      → docs → sql → finalize → END
```

(See `architecture_graph.png`.)

**Nodes**
- **router** — an LLM classifier that returns `documents | database | both | smalltalk` with a reason.
- **docs** — document RAG: semantic retrieval from ChromaDB → LLM reranking → grounded answer.
- **sql** — Text-to-SQL: schema-aware SQL generation → safety check → execute → phrase the answer.
- **smalltalk** — conversational LLM reply for greetings / general advice.
- **finalize** — returns the single-source answer, or synthesizes the doc + SQL answers for `both`.

---

## 3. Key design decisions & trade-offs

### 3.1 Routing (documents vs database vs both)
- **Decision:** an LLM classifier with explicit source descriptions and few-shot examples.
- **Why:** flexible, handles paraphrase and intent; easy to explain and tune.
- **Trade-off:** an extra LLM call per query (latency/cost) and non-determinism on
  genuinely ambiguous questions. Mitigated with clear rules/examples and validated by the eval set.
- **Alternative considered:** keyword/rules routing (fast, deterministic, but brittle);
  or an agent that picks tools itself (more flexible, harder to control/observe).

### 3.2 Retrieval design (RAG)
- **Chunking:** ~1000 chars with 150 overlap (`RecursiveCharacterTextSplitter`) to keep context coherent.
- **Embeddings:** `text-embedding-3-small` — strong quality/cost balance.
- **Retrieve-then-rerank:** retrieve top-15 by similarity, then LLM-rerank to the best 4.
  - **Why:** similarity alone mixes in near-duplicates from other docs (seen as Precision@4 ≈ 0.67);
    reranking improves the precision of what actually reaches the answer prompt.
  - **Trade-off:** one extra LLM call; acceptable for the accuracy gain.

### 3.3 Text-to-SQL
- **Decision:** feed the live schema to the LLM, generate one `SELECT`, execute, then phrase the result.
- **Date-awareness:** the prompt injects today's date so "active promotions" filters by `date('now')`.
- **Safety:** a guard rejects anything that isn't a single read-only `SELECT`.
- **Trade-off:** LLM-generated SQL can be wrong on complex joins; mitigated by a small,
  well-described schema and the eval set. Production would add query validation / allow-lists.

### 3.4 Source attribution
- Documents: filename + page + the retrieved chunk text + an openable copy of the file.
- Database: the exact SQL query **and** the returned rows.
- **Why:** trust and auditability — the user can verify *how* every answer was produced.

### 3.5 Orchestration with LangGraph
- **Decision:** model the flow as an explicit graph rather than ad-hoc `if/else`.
- **Why:** a clear execution graph, easy conditional branching, and a foundation for
  checkpointed memory and human-in-the-loop interrupts.

---

## 4. Reliability & evaluation

`evaluate.py` runs a labelled test set and reports:
- **Router accuracy** — predicted vs expected route.
- **Answer keyword match** — a lightweight correctness proxy.
- **Precision@k / Recall@k / Hit-rate** — retrieval quality against the expected source document.

**Latest results (12-question set):** Router 12/12 (100%), Answer 10/10 (100%),
Hit-rate@4 100%, Recall@4 0.88, Precision@4 0.67.

The evaluation caught a real routing weakness (ambiguous "click & collect" questions),
which was fixed by refining the router prompt — demonstrating an evaluate → diagnose →
iterate loop. Production would add LLM-as-judge scoring and a larger, versioned dataset.

---

## 5. Security & governance

- **Read-only SQL:** only single `SELECT` statements execute; DML/DDL and stacked queries are blocked.
- **PII redaction:** customer emails are masked before the LLM or UI ever see them.
- **Secrets management:** API keys live only in `.env` (git-ignored); `.env.example` documents them.
- **Grounding:** the RAG prompt forbids using outside knowledge, reducing hallucination.

---

## 6. Limitations

- SQLite substitutes for BigQuery in the prototype.
- Store-assistant escalation is simulated (no real ticket/live-chat integration).
- PII masking covers emails only.
- Conversation memory is per-session (no persistence across restarts).
- Small corpus means retrieval metrics are indicative, not exhaustive.

---

## 7. Path to production

- **Data:** swap SQLite → **BigQuery/Postgres**; keep the Text-to-SQL layer, add query
  validation, cost guards, and **row-level access control** so users only see permitted data.
- **Vector store:** move ChromaDB → a managed/hosted vector DB; add metadata filtering and hybrid (keyword + vector) search; consider a dedicated reranker model.
- **Governance:** integrate a PII engine (e.g. Presidio) for names/phones/addresses; add audit logging, prompt-injection defenses, and content guardrails.
- **Reliability:** larger versioned eval sets, LLM-as-judge, regression tests in CI, and online feedback capture.
- **Ops:** authentication/SSO, caching, rate limiting, autoscaling, and dashboards/alerts built on LangSmith traces.
- **UX:** persistent, checkpointed conversation memory and a real human-in-the-loop handoff (LangGraph interrupts).
