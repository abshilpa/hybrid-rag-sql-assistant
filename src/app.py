import os
import sys
import streamlit as st

sys.path.append(os.path.dirname(__file__))

from ingest import ingest_files, DOCS_DIR
from assistant import answer

st.set_page_config(page_title="Retail Q&A Assistant", page_icon="🛍️", layout="wide")
st.title("🛍️ Retail Q&A Assistant")
st.caption("Answers grounded in policy documents, the live retail database, or both.")

with st.sidebar:
    st.header("📄 Upload documents")
    uploaded = st.file_uploader(
        "Add documents to the knowledge base",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )
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
    st.caption("Documents are embedded into ChromaDB. Database answers are always live.")


def render_result(result, idx):
    st.markdown(f"**Route:** `{result['route']}` — {result['reason']}")
    st.markdown(result["answer"])

    has_sources = result["doc_sources"] or result["sql_query"]
    if has_sources:
        with st.expander("📌 Sources used"):
            if result["doc_sources"]:
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
            if result["sql_query"]:
                st.markdown("**Database query:**")
                st.code(result["sql_query"], language="sql")
                if result.get("sql_result"):
                    st.markdown("**Query result:**")
                    st.code(result["sql_result"])

    ans = result["answer"].lower()
    if "could not find" in ans or "couldn't find" in ans or "store assistant" in ans:
        if st.button("🧑‍💼 Connect me to a store assistant", key=f"esc_{idx}"):
            st.info("Connecting you to a live store assistant… (simulated). "
                    "I raised a support ticket for you our assistant will get back to you shortly.")


if "history" not in st.session_state:
    st.session_state.history = []

for i, turn in enumerate(st.session_state.history):
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        render_result(turn, i)

question = st.chat_input("Ask a question...")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = answer(question)
        render_result(result, len(st.session_state.history))
    st.session_state.history.append(result)