import os
import sys
import json
import uuid
import streamlit as st

sys.path.append(os.path.dirname(__file__))

from ingest import ingest_files, DOCS_DIR
from assistant import answer
from feedback import log_feedback


HISTORY_FILE = "chat_history.json"

st.set_page_config(page_title="Retail Q&A Assistant", page_icon="🛍️", layout="wide")
st.title("🛍️ Retail Q&A Assistant")
st.caption("Answers grounded in policy documents, the live retail database, or both.")


def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_history(history):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ---- per-session state ----
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]   # unique per browser session
if "messages" not in st.session_state:
    st.session_state.messages = []                        # FRESH each open (main chat)

# ---- Sidebar ----
with st.sidebar:
    st.header("⚙️ Settings")
    role = st.selectbox(
        "View as role", ["customer", "admin"],
        help="Admin sees customer name & email; customer sees them masked (RBAC).",
    )
    st.caption(f"Session: `{st.session_state.session_id}`")

    st.markdown("---")
    st.header("📄 Upload documents")
    uploaded = st.file_uploader("Add documents to the knowledge base",
                                type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)
    if uploaded and st.button("Ingest uploaded documents"):
        os.makedirs(DOCS_DIR, exist_ok=True)
        saved = []
        for f in uploaded:
            dest = os.path.join(DOCS_DIR, f.name)
            with open(dest, "wb") as out:
                out.write(f.getbuffer())
            saved.append(dest)
        with st.spinner("Ingesting into ChromaDB..."):
            n = ingest_files(saved, replace=True)
        st.success(f"Ingested {len(saved)} document(s) — {n} chunks.")

    st.markdown("---")
    st.header("🕘 Conversation history")
    past = load_history()
    if past:
        for turn in reversed(past[-20:]):        # most recent first
            st.caption(f"• {turn['question']}")
        if st.button("🗑️ Clear history"):
            save_history([])
            st.rerun()
    else:
        st.caption("No past questions yet.")



def render_result(result, idx):
    st.markdown(result["answer"])
    if result.get("cached"):
        st.caption(f"⚡ Served from semantic cache (similarity {result.get('cache_similarity')})")
    has_sources = result.get("doc_sources") or result.get("sql_query")
    if has_sources:
        with st.expander("🔍 Details & sources"):
            st.markdown(f"**Routing decision:** `{result['route']}` — {result['reason']}")
            if result.get("doc_sources"):
                st.markdown("**Supporting document chunks:**")
                unique_files = []
                for s in result["doc_sources"]:
                    if s["source"] not in unique_files:
                        unique_files.append(s["source"])
                    snippet = s["text"][:400] + ("..." if len(s["text"]) > 400 else "")
                    st.markdown(f"**{s['source']}** — page {s['page']}")
                    st.caption(snippet)
                st.markdown("**Open source documents:**")
                for k, fname in enumerate(unique_files):
                    path = os.path.join(DOCS_DIR, fname)
                    if os.path.exists(path):
                        with open(path, "rb") as fh:
                            st.download_button(f"📄 Open {fname}", fh.read(),
                                               file_name=fname, key=f"dl_{idx}_{k}")
            if result.get("sql_query"):
                st.markdown("**Database query:**")
                st.code(result["sql_query"], language="sql")
                if result.get("sql_result"):
                    st.markdown("**Query result:**")
                    st.code(result["sql_result"])
    if result.get("needs_escalation"):
        if st.button("🧑‍💼 Connect me to a store assistant", key=f"esc_{idx}"):
            ticket = "RS-" + uuid.uuid4().hex[:6].upper()
            st.success(f"✅ Thanks — I've raised ticket **#{ticket}**. A store assistant will "
                       f"review your query and get back to you by email with the details. (Simulated)")

    # ---- 👍 / 👎 feedback ----
    fb_key = result.get("run_id") or f"{st.session_state.session_id}-{idx}"
    st.session_state.setdefault("rated", {})
    if fb_key in st.session_state.rated:
        st.caption(f"✅ Thanks for your feedback ({st.session_state.rated[fb_key]}).")
    else:
        col_up, col_down, _ = st.columns([1, 1, 10])
        clicked = None
        if col_up.button("👍", key=f"up_{idx}", help="This answer was helpful"):
            clicked = "up"
        if col_down.button("👎", key=f"down_{idx}", help="This answer needs work"):
            clicked = "down"
        if clicked:
            log_feedback(
                question=result.get("question", ""),
                answer=result.get("answer", ""),
                route=result.get("route", ""),
                rating=clicked,
                run_id=result.get("run_id"),
                sources=[s.get("source") for s in result.get("doc_sources", [])],
                role=result.get("role", "customer"),
            )
            st.session_state.rated[fb_key] = "👍" if clicked == "up" else "👎"
            st.rerun()


# ---- Render CURRENT session's messages (empty on open) ----
for i, turn in enumerate(st.session_state.messages):
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        render_result(turn, i)

# ---- New question ----
question = st.chat_input("Ask a question...")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = answer(
                question,
                role=role,
                langsmith_extra={"metadata": {
                    "session_id": st.session_state.session_id,
                    "thread_id": st.session_state.session_id,
                    "role": role,
                }},
            )
        render_result(result, len(st.session_state.messages))
    st.session_state.messages.append(result)
    all_hist = load_history()
    all_hist.append({"question": question, "session_id": st.session_state.session_id})
    save_history(all_hist)