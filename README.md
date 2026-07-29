# 🛍️ Retail Q&A Assistant — Hybrid RAG + Text-to-SQL

A prototype Q&A assistant that answers natural-language questions using **both**
uploaded policy documents **and** a live relational database, and clearly shows the
**sources** behind every answer.

Built for a retail scenario (JD Retail): policy PDFs / DOCX plus a SQLite retail database
(Products, Orders, Inventory, Promotions, Stores).

> A sister repository, **Hybrid-RAG-SQL-Assistant-with-memory**, extends this baseline with
> three-layer conversation memory. This repo is the pre-memory version.

---

## ✨ Features

- **Hybrid answering** — a question is answered from **documents** (RAG), the **database**
  (Text-to-SQL), or **both**, decided automatically by an LLM **router**. A general request
  for a product ("tell me about X") routes to **both** — description from documents, live
  price and stock from the database.
- **Document RAG** — PDF / DOCX / TXT / MD ingestion with **structure-aware chunking** (each
  policy section and each catalogue entry becomes its own chunk), then **multi-query + hybrid
  (vector + BM25) + MMR retrieval**, **LLM reranking**, and **answer-grounded source
  attribution** (the chunk shown is the one whose content best matches the answer).
- **Text-to-SQL** — natural language to a safe, read-only `SELECT` against the live database,
  date-aware for active promotions and value-hint-aware (maps "in process" to the real enum
  value "Processing").
- **Source attribution** — every answer shows the supporting document chunk (name, page,
  snippet, openable file) and/or the exact SQL query and result.
- **Small talk** — greetings and general shopping questions are answered conversationally.
- **Escalation** — when an answer isn't in the knowledge base, it asks a clarifying question
  and offers a (simulated) handoff to a store assistant with a ticket number.
- **Security / governance** — read-only SQL guard, **RBAC PII masking** (customer names and
  emails masked for non-admin roles), non-admin requests for customer PII are politely
  **refused at the query layer**, and secrets kept out of source control.
- **Orchestration** — a **LangGraph** state machine (router → docs / sql / smalltalk → finalize).
- **Monitoring** — full tracing of every request and component via **LangSmith**.
- **Evaluation** — a test set scoring router accuracy, answer correctness, and retrieval
  **Precision@k / Recall@k / Hit-rate**.


---

## 🏗️ Architecture

```
User ──> Streamlit UI ──> LangGraph orchestrator
                              │
                        ┌─────┴───────────────┐
                        │   Router (LLM)       │  documents | database | both | smalltalk
                        └─────┬───────────────┘
             ┌────────────────┼───────────────────┐
             ▼                ▼                    ▼
      Document RAG       Text-to-SQL          Small talk
   (Chroma + rerank)   (SQLite, read-only)   (LLM chat)
             │                │
             └──────┬─────────┘
                    ▼
             Finalize / synthesize  ──> Answer + Sources
```

See `architecture_graph.png` for the rendered LangGraph flow.

---

## 🧰 Tech stack

| Concern            | Choice                                   |
|--------------------|-------------------------------------------|
| Orchestration      | LangGraph (on LangChain)                  |
| LLM & embeddings   | OpenAI (`gpt-4o-mini`, `text-embedding-3-small`) |
| Vector store       | ChromaDB (persistent, local)              |
| Structured data    | SQLite (`jd_retail.db`)                    |
| UI                 | Streamlit                                 |
| Monitoring         | LangSmith                                 |

> Note: SQLite stands in for BigQuery in the prototype the SQL layer is
> swappable for BigQuery/Postgres in production.

---

## 📁 Project structure

```
qa-assistant/
├── data/
│   ├── documents/        # policy PDFs / DOCX (knowledge base)
│   └── db/jd_retail.db   # SQLite retail database
├── src/
│   ├── ingest.py         # load, structure-aware chunk, embed docs -> ChromaDB
│   ├── doc_qa.py         # document RAG (multi-query + hybrid + rerank + answer-grounded sources)
│   ├── sql_qa.py         # text-to-SQL (safe SELECT, value hints, RBAC PII masking + refusal)
│   ├── router.py         # LLM router (documents/database/both/smalltalk)
│   ├── assistant.py      # LangGraph orchestration
│   ├── app.py            # Streamlit UI
│   ├── evaluate.py       # evaluation harness (accuracy + P@k/R@k)
│   └── explore_db.py     # inspect the database schema
├── requirements.txt
├── .env.example
└── README.md
```

---

---

## 🚀 Setup & run

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/Scripts/activate        # Windows Git Bash
# source venv/bin/activate          # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure secrets
cp .env.example .env                # then fill in your keys

# 4. Build the vector store from the documents
python src/ingest.py

# 5. Run the app
streamlit run src/app.py
```

Open http://localhost:8501, ask a question, or upload a new document in the sidebar.

---

## 🧪 Evaluation

```bash
python src/evaluate.py
```

Reports router accuracy, answer keyword match, and retrieval **Precision@k / Recall@k /
Hit-rate** against a labelled test set.

---

## 🔐 Security & governance

- **Read-only SQL** — only `SELECT` queries are allowed, and DML / DDL is blocked.
- **RBAC PII masking** — customer names and emails are masked for non-admin roles before they
  reach the LLM or UI.
- **PII refusal** — a non-admin request that tries to read customer names or emails is politely
  declined at the query layer, rather than run and masked (defense in depth).
- **Secrets** — API keys live in `.env` (git-ignored), and a `.env.example` documents the
  required keys.

---

## ⚠️ Limitations & future work

- SQLite is a stand-in for BigQuery, so swap the SQL layer for production.
- Store-assistant handoff is simulated, and would integrate a real ticketing or live-chat system.
- PII masking covers names and emails, and production would use a dedicated PII engine
  (e.g. Presidio) for phones and addresses.
- Conversation history is per-session. Persistent, checkpointed memory is the focus of the
  sister **with-memory** repository.
- Add authentication, role-based row-level data access, caching, and rate limiting for production.