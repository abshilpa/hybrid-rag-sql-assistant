import sys
import re
from dotenv import load_dotenv
from langsmith import traceable
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from ingest import get_vectorstore

load_dotenv()

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful retail support assistant answering questions about company policies.
Use ONLY the information in the context below (never use outside knowledge), but you SHOULD
reason over it and be genuinely helpful:
- If the context describes a relevant policy, process, or set of conditions that addresses
  the question — even partially — then ANSWER using it. For example, a manufacturing-fault
  claim process answers "my item is faulty, what should I do?"; return conditions answer
  "can I return X?".
- ONLY if the context contains nothing relevant to the question at all, begin your reply
  with the exact token [[NO_ANSWER]], briefly say it isn't covered, and ask ONE clarifying
  question.

Context:
{context}

Question: {question}

Answer:"""
)



RERANK_PROMPT = ChatPromptTemplate.from_template(
    """You are selecting the chunks most useful for answering the question.
Question: {question}

Chunks:
{chunks}

Return ONLY a comma-separated list of the numbers of the chunks that could help
answer the question, most useful first, at most {top_k}. Include any chunk that
contains information related to the question (e.g. faults, claims, warranty coverage,
product care, returns, delivery). Only omit chunks that are clearly about a completely
different topic.
"""
)





@traceable(name="rerank")
def rerank(question, docs, top_k=4):
    if not docs:
        return []
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    listing = "\n\n".join(f"[{i}] {d.page_content[:500]}" for i, d in enumerate(docs))
    raw = llm.invoke(RERANK_PROMPT.format(question=question, chunks=listing, top_k=top_k)).content.strip()
    order = []
    for tok in re.findall(r"\d+", raw):
        i = int(tok)
        if 0 <= i < len(docs) and i not in order:
            order.append(i)
        if len(order) >= top_k:
            break
    # Safety fallback: never drop everything when we have candidates
    if not order:
        return docs[:2]
    return [docs[i] for i in order]


def format_docs(docs):
    blocks = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        blocks.append(f"[Source: {src}, page {page}]\n{d.page_content}")
    return "\n\n".join(blocks)


@traceable(name="docs_pipeline")
def answer_from_docs(question, retrieve_k=15, top_k=4):
    vs = get_vectorstore()
    candidates = vs.as_retriever(search_kwargs={"k": retrieve_k}).invoke(question)
    docs = rerank(question, candidates, top_k=top_k)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ANSWER_PROMPT.format(context=format_docs(docs), question=question)
    response = llm.invoke(prompt)
    return response.content, docs


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is your returns policy?"
    print(f"\nQ: {question}\n")
    answer, sources = answer_from_docs(question)
    print("A:", answer)
    print("\n--- Sources used (after reranking) ---")
    for d in sources:
        print(f"   - {d.metadata.get('source')} (page {d.metadata.get('page')})")