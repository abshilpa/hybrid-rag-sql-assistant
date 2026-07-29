import sys
import re
from dotenv import load_dotenv
from langsmith import traceable
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from ingest import get_vectorstore

load_dotenv()

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful retail support assistant answering questions about company policies.
Use ONLY the numbered context chunks below (never outside knowledge). Reason over them; if a
chunk partially addresses the question, use it.

Write your answer. Then, on a NEW line, write "SOURCES:" followed by the numbers of ONLY the
chunks you actually used to answer, comma-separated (e.g. "SOURCES: 2"). Do not list chunks you
did not use.

If NONE of the chunks are relevant, reply with "[[NO_ANSWER]]" on the first line, briefly say it
isn't covered and ask ONE clarifying question, then write "SOURCES:" with nothing after it.

Context:
{context}

Question: {question}

Answer:"""
)



RERANK_PROMPT = ChatPromptTemplate.from_template(
    """You are ranking chunks by how useful they are for answering the question.
Question: {question}

Chunks:
{chunks}

Return ONLY a comma-separated list of the numbers of the {top_k} MOST useful chunks,
most useful first. Prefer chunks that directly contain the answer. You may return fewer
than {top_k} if only a couple are useful.
"""
)



MULTIQUERY_PROMPT = ChatPromptTemplate.from_template(
    """Generate 3 alternative phrasings of the question below to improve document
retrieval (use synonyms and related wording, e.g. "return" ↔ "refund",
"reward points" ↔ "loyalty points"). One per line, no numbering.

Question: {question}
"""
)


def _all_chunks_as_documents(vs):
    stored = vs.get(include=["documents", "metadatas"])
    return [Document(page_content=t, metadata=m or {})
            for t, m in zip(stored["documents"], stored["metadatas"])]


@traceable(name="expand_queries")
def expand_queries(question):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    raw = llm.invoke(MULTIQUERY_PROMPT.format(question=question)).content
    variants = [line.strip(" -•*").strip() for line in raw.splitlines() if line.strip()]
    return [question] + variants[:3]      # original + 3 paraphrases


@traceable(name="hybrid_retrieve")
def hybrid_retrieve(question, retrieve_k=10):
    """Multi-query + hybrid: paraphrase the question, then vector (MMR) + BM25 for
    each variant, merged and de-duplicated by content."""
    vs = get_vectorstore()
    corpus = _all_chunks_as_documents(vs)
    bm25 = BM25Retriever.from_documents(corpus)
    bm25.k = retrieve_k
    vector = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": retrieve_k, "fetch_k": retrieve_k * 2, "lambda_mult": 0.5},
    )

    merged, seen = [], set()
    for q in expand_queries(question):        # Multi-Query Retrieval
        for d in vector.invoke(q) + bm25.invoke(q):
            key = d.page_content.strip()[:120]
            if key not in seen:
                seen.add(key)
                merged.append(d)
    return merged




@traceable(name="rerank")
def rerank(question, docs, top_k=4):
    if not docs:
        return []
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    listing = "\n\n".join(f"[{i}] {d.page_content[:600]}" for i, d in enumerate(docs))
    raw = llm.invoke(RERANK_PROMPT.format(question=question, chunks=listing, top_k=top_k)).content.strip()
    order = []
    for tok in re.findall(r"\d+", raw):
        i = int(tok)
        if 0 <= i < len(docs) and i not in order:
            order.append(i)
        if len(order) >= top_k:
            break
    return [docs[i] for i in order] if order else docs[:top_k]   # fallback: top candidates, never just 1



def format_docs(docs):
    blocks = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        blocks.append(f"[{i}] (Source: {src}, page {page})\n{d.page_content}")
    return "\n\n".join(blocks)



@traceable(name="docs_pipeline")
def answer_from_docs(question, retrieve_k=10, top_k=6):
    candidates = hybrid_retrieve(question, retrieve_k)
    docs = rerank(question, candidates, top_k=top_k)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(ANSWER_PROMPT.format(context=format_docs(docs), question=question)).content

    # Strip any trailing "SOURCES:" line the model may add
    m = re.search(r"\n?\s*SOURCES?\s*:.*$", response, flags=re.IGNORECASE | re.DOTALL)
    answer_text = (response[:m.start()] if m else response).strip()

    if answer_text.startswith("[[NO_ANSWER]]"):
        return answer_text, []

    # Attribute sources by matching the ANSWER back to the retrieved chunks (content overlap)
    STOP = set(("the a an is are was were of to and or in on for with it its this that these those "
                "be as at by from you your our we they them their have has had will can may not do "
                "does no yes if then than also more most into over under a s").split())
    def _tok(s):
        return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2 and w not in STOP}

    ans_tok = _tok(answer_text)
    if not ans_tok:
        return answer_text, docs[:1]
    scored = sorted(docs, key=lambda d: len(ans_tok & _tok(d.page_content)), reverse=True)
    best = len(ans_tok & _tok(scored[0].page_content))
    if best == 0:
        return answer_text, docs[:1]
    used = [d for d in scored if len(ans_tok & _tok(d.page_content)) >= max(2, best * 0.6)][:3]
    return answer_text, used or [scored[0]]



if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is your returns policy?"
    print(f"\nQ: {question}\n")
    answer, sources = answer_from_docs(question)
    print("A:", answer)
    print("\n--- Sources used (hybrid + dedup + rerank) ---")
    for d in sources:
        print(f"   - {d.metadata.get('source')} (page {d.metadata.get('page')})")