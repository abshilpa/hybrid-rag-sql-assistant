# Design Pack — Hybrid RAG + Text-to-SQL Q&A Assistant

This document walks through the architecture, the key design decisions and their
trade-offs, the current limitations, and how the prototype would evolve into a
production-ready enterprise system.

> A sister repository, **Hybrid-RAG-SQL-Assistant-with-memory**, extends this baseline with
> three-layer conversation memory. This document describes the pre-memory version.

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

The system is a **LangGraph state machine**. A shared state flows through nodes, and the
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
- **docs** — document RAG: structure-aware chunks in ChromaDB, multi-query + hybrid retrieval, LLM reranking, grounded answer with answer-grounded source attribution.
- **sql** — Text-to-SQL: schema-aware SQL generation, safety check, execute, phrase the answer.
- **smalltalk** — conversational LLM reply for greetings and general advice.
- **finalize** — returns the single-source answer, or synthesizes the doc and SQL answers for `both`.

---

## 3. Key design decisions & trade-offs

### 3.1 Routing (documents vs database vs both)
- **Decision:** an LLM classifier with explicit source descriptions and few-shot examples.
- **Why:** flexible, handles paraphrase and intent, and is easy to explain and tune.
- **Intent disambiguation:** the same product name routes differently by intent. "How much is X" goes to the database, "what is X made of" goes to documents, and a general "tell me about X" goes to **both** — the description from documents and the live price and stock from the database.
- **Trade-off:** an extra LLM call per query (latency and cost) and some non-determinism on genuinely ambiguous questions. Mitigated with ordered rules, examples, and validation by the eval set.
- **Alternative considered:** keyword or rules routing (fast and deterministic but brittle), or an agent that picks tools itself (more flexible but harder to control and observe).

### 3.2 Retrieval design (RAG)
- **Structure-aware chunking:** documents are split at their natural section and entry boundaries, so each policy section and each catalogue item (each product, each menu dish) becomes its own chunk. A heading heuristic detects section titles, runs of list items are kept together, and any oversized section falls back to recursive splitting. This replaced fixed-size chunking, which mixed multiple entries into one chunk and blurred retrieval.
- **Embeddings:** `text-embedding-3-small` for a strong quality and cost balance.
- **Retrieve broadly, filter precisely:** multi-query expansion (paraphrases of the question), then hybrid retrieval combining vector search with MMR and BM25 keyword search, then an LLM reranker keeps the most useful chunks.
- **Answer-grounded source attribution:** after the answer is written, the displayed source is the retrieved chunk whose content best overlaps the answer, so the shown evidence always matches what the answer used. This replaced a model-self-citation approach that could show an unrelated chunk.
- **Freshness:** each chunk is hashed, so re-ingest embeds only new or changed chunks and deletes removed ones.
- **Trade-off:** multi-query and reranking add LLM calls, which is acceptable for the retrieval-quality gain.

### 3.3 Text-to-SQL
- **Decision:** feed the live schema to the LLM, generate one `SELECT`, execute, then phrase the result.
- **Value hints:** distinct values of low-cardinality columns are injected so the LLM maps user wording to the real enum values (for example "in process" to "Processing").
- **Date-awareness:** the prompt injects today's date so "active promotions" filter by `date('now')`.
- **Safety:** a guard rejects anything that is not a single read-only `SELECT`.
- **Trade-off:** LLM-generated SQL can be wrong on complex joins, mitigated by a small, well-described schema and the eval set. Production would add query validation and allow-lists.

### 3.4 Source attribution
- Documents: filename, page, the retrieved chunk text, and an openable copy of the file. The chunk shown is the one whose content best matches the answer.
- Database: the exact SQL query **and** the returned rows.
- **Why:** trust and auditability, so the user can verify how every answer was produced.

### 3.5 Orchestration with LangGraph
- **Decision:** model the flow as an explicit graph rather than ad-hoc `if/else`.
- **Why:** a clear execution graph, easy conditional branching, and a foundation for checkpointed memory and human-in-the-loop interrupts.

---

## 4. Reliability & evaluation

`evaluate.py` runs a labelled test set and reports:
- **Router accuracy** — predicted vs expected route.
- **Answer keyword match** — a lightweight correctness proxy.
- **Precision@k / Recall@k / Hit-rate** — retrieval quality against the expected source document.

The evaluation drives an evaluate → diagnose → iterate loop. For example, it surfaced a
routing weakness on ambiguous "click & collect" questions, and separately, manual testing
showed fixed-size chunks mixing catalogue entries, which motivated the move to structure-aware
chunking and answer-grounded source attribution. Production would add LLM-as-judge scoring and
a larger, versioned dataset.

---

## 5. Security & governance

- **Read-only SQL:** only single `SELECT` statements execute. DML, DDL, and stacked queries are blocked.
- **RBAC PII masking:** customer names and emails are masked for non-admin roles before the LLM or UI ever see them, while an admin role can see them to resolve tickets.
- **PII refusal:** a non-admin query that tries to read customer names or emails is politely declined at the query layer, rather than run and masked. This is a defense-in-depth layer on top of masking.
- **Secrets management:** API keys live only in `.env` (git-ignored), and `.env.example` documents them.
- **Grounding:** the RAG prompt forbids using outside knowledge, which reduces hallucination.

---

## 6. Limitations

- SQLite substitutes for BigQuery in the prototype.
- Store-assistant escalation is simulated, with no real ticket or live-chat integration.
- PII masking covers names and emails, not a general entity recogniser.
- Conversation memory is per-session. Persistent memory is the focus of the sister with-memory repository.
- The small corpus means retrieval metrics are indicative rather than exhaustive.

---

## 7. Path to production

- **Data:** swap SQLite for **BigQuery or Postgres**, keep the Text-to-SQL layer, and add query validation, cost guards, and **row-level access control** so users only see permitted data.
- **Vector store:** move ChromaDB to a managed or hosted vector DB and add metadata filtering. Hybrid retrieval and reranking are already in place, and a dedicated reranker model could be added.
- **Governance:** integrate a PII engine (for example Presidio) for phones and addresses, and add audit logging, prompt-injection defenses, and content guardrails.
- **Reliability:** larger versioned eval sets, LLM-as-judge, regression tests in CI, and online feedback capture.
- **Ops:** authentication and SSO, caching, rate limiting, autoscaling, and dashboards and alerts built on LangSmith traces.
- **UX:** persistent, checkpointed conversation memory and a real human-in-the-loop handoff (LangGraph interrupts). Conversation memory is implemented in the sister repository.