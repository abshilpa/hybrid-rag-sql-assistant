import sys
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from ingest import get_vectorstore

load_dotenv()

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful retail support assistant answering questions about company policies.
Use ONLY the information in the context below, but you MAY reason over it to reach a
conclusion (e.g. if the policy lists the conditions for a valid return, use them to decide
whether a specific item qualifies, and briefly explain why).

If the context genuinely does NOT cover the topic:
- Do not make anything up.
- Say you couldn't find it in the current policy documents.
- Ask ONE clarifying question that would help you assist better.
- Offer to connect the customer to a live store assistant.

Context:
{context}

Question: {question}

Answer:"""
)

RERANK_PROMPT = ChatPromptTemplate.from_template(
    """Rank the chunks by how well they help answer the question.
Question: {question}

Chunks:
{chunks}

Return ONLY a comma-separated list of the numbers of the {top_k} most relevant
chunks, best first. Example: 3,1,7,0
"""
)


def format_docs(docs):
    blocks = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        blocks.append(f"[Source: {src}, page {page}]\n{d.page_content}")
    return "\n\n".join(blocks)


def rerank(question, docs, top_k=4):
    if len(docs) <= top_k:
        return docs
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    listing = "\n\n".join(f"[{i}] {d.page_content[:500]}" for i, d in enumerate(docs))
    raw = llm.invoke(RERANK_PROMPT.format(question=question, chunks=listing, top_k=top_k)).content
    order = []
    for tok in re.findall(r"\d+", raw):
        i = int(tok)
        if 0 <= i < len(docs) and i not in order:
            order.append(i)
        if len(order) >= top_k:
            break
    return [docs[i] for i in order] if order else docs[:top_k]


def answer_from_docs(question, retrieve_k=15, top_k=4):
    vs = get_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": retrieve_k})
    candidates = retriever.invoke(question)
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