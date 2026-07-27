import sys
from typing import TypedDict, List, Optional
from dotenv import load_dotenv
from langsmith import traceable
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

from router import route_question
from doc_qa import answer_from_docs
from sql_qa import answer_from_sql

load_dotenv()

SYNTHESIS_PROMPT = ChatPromptTemplate.from_template(
    """Combine the two partial answers below into ONE clear, coherent answer to
the user's question. Don't repeat yourself. If a part is not relevant, ignore it.

Question: {question}

Answer from policy documents:
{doc_answer}

Answer from the database:
{sql_answer}

Final combined answer:"""
)

SMALLTALK_PROMPT = ChatPromptTemplate.from_template(
    """You are a friendly, helpful shopping assistant for JD Retail.
The user's message is a greeting, general chit-chat, or a general shopping question
that is NOT about a specific company policy or a specific order/product record.
Reply warmly and helpfully in 1-3 sentences. You may give general advice. Where natural,
mention you can also help with returns, delivery, orders, stock levels, and promotions.

User: {question}
Assistant:"""
)

NO_ANSWER = "[[NO_ANSWER]]"


class QAState(TypedDict):
    question: str
    route: str
    reason: str
    answer: str
    doc_answer: Optional[str]
    sql_answer: Optional[str]
    doc_sources: List[dict]
    sql_query: Optional[str]
    sql_result: Optional[str]
    needs_escalation: bool


@traceable(name="router_node")
def router_node(state: QAState):
    decision = route_question(state["question"])
    return {"route": decision.get("route", "both"), "reason": decision.get("reason", "")}


@traceable(name="smalltalk_node")
def smalltalk_node(state: QAState):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    return {"answer": llm.invoke(SMALLTALK_PROMPT.format(question=state["question"])).content}


@traceable(name="docs_node")
def docs_node(state: QAState):
    doc_answer, docs = answer_from_docs(state["question"])
    no_ans = doc_answer.strip().startswith(NO_ANSWER)
    if no_ans:
        doc_answer = doc_answer.replace(NO_ANSWER, "").strip()

    updates = {"doc_answer": doc_answer}
    if no_ans and state["route"] == "documents":
        updates["needs_escalation"] = True
        updates["doc_sources"] = []          # nothing was actually used
    else:
        updates["doc_sources"] = [
            {"source": d.metadata.get("source"),
             "page": d.metadata.get("page"),
             "text": d.page_content}
            for d in docs
        ]
    return updates


@traceable(name="sql_node")
def sql_node(state: QAState):
    sql_answer, sql, sql_result = answer_from_sql(state["question"])
    return {"sql_answer": sql_answer, "sql_query": sql, "sql_result": str(sql_result)}


@traceable(name="finalize_node")
def finalize_node(state: QAState):
    route = state["route"]
    if route == "documents":
        return {"answer": state.get("doc_answer") or ""}
    if route == "database":
        return {"answer": state.get("sql_answer") or ""}
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    combined = llm.invoke(
        SYNTHESIS_PROMPT.format(
            question=state["question"],
            doc_answer=state.get("doc_answer"),
            sql_answer=state.get("sql_answer"),
        )
    ).content
    return {"answer": combined}


def route_after_router(state: QAState):
    return state["route"]


def route_after_docs(state: QAState):
    return "sql" if state["route"] == "both" else "finalize"


def build_graph():
    g = StateGraph(QAState)
    g.add_node("router", router_node)
    g.add_node("smalltalk", smalltalk_node)
    g.add_node("docs", docs_node)
    g.add_node("sql", sql_node)
    g.add_node("finalize", finalize_node)
    g.add_edge(START, "router")
    g.add_conditional_edges("router", route_after_router, {
        "smalltalk": "smalltalk", "documents": "docs", "database": "sql", "both": "docs",
    })
    g.add_edge("smalltalk", END)
    g.add_conditional_edges("docs", route_after_docs, {"sql": "sql", "finalize": "finalize"})
    g.add_edge("sql", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


qa_graph = build_graph()


@traceable(name="qa_assistant")
def answer(question):
    initial: QAState = {
        "question": question, "route": "", "reason": "", "answer": "",
        "doc_answer": None, "sql_answer": None, "doc_sources": [],
        "sql_query": None, "sql_result": None, "needs_escalation": False,
    }
    return qa_graph.invoke(initial)


def print_result(r):
    print(f"\nQ: {r['question']}")
    print(f"Route: {r['route']}   (reason: {r['reason']})")
    print(f"Needs escalation: {r['needs_escalation']}\n")
    print("Answer:\n", r["answer"])
    print("\n--- Sources ---")
    for s in r["doc_sources"]:
        print(f"   [Document] {s['source']} (page {s['page']})")
    if r["sql_query"]:
        print(f"   [Database] SQL: {r['sql_query']}")
    if not r["doc_sources"] and not r["sql_query"]:
        print("   (none)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--graph":
        print(qa_graph.get_graph().draw_mermaid())
    else:
        q = " ".join(sys.argv[1:]) or "Do you offer gift wrapping?"
        print_result(answer(q))