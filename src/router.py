import sys
import json
from dotenv import load_dotenv
from langsmith import traceable
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from ingest import get_vectorstore

load_dotenv()


def get_ingested_docs():
    """Return the list of document filenames currently in the vector store."""
    try:
        vs = get_vectorstore()
        stored = vs.get(include=["metadatas"])
        names = sorted({m.get("source") for m in (stored.get("metadatas") or [])
                        if m and m.get("source")})
        return names
    except Exception:
        return []


ROUTER_PROMPT = ChatPromptTemplate.from_template(
    """You are a routing assistant for a retail Q&A system. Decide how to handle the message.

Sources:
1. DOCUMENTS (policy files). The knowledge base CURRENTLY contains these documents:
   {documents_list}
   Documents cover policies, capabilities, how things work, and the COST / FEE / RATE / RULES
   of SERVICES and PROGRAMS (e.g. delivery charges and times, gift-wrapping fees, click & collect,
   returns, warranty, and loyalty/reward-point earn-and-redeem rates).
2. DATABASE (SQL): live records only -> Products (a specific product's PRICE or STOCK),
   Orders (a specific order's status/tracking/delivery), Inventory (stock per store),
   Promotions (which product is discounted and by how much, with dates), Stores (a list of
   stores, cities, hours, click&collect availability, phone).

Choose ONE route using these rules IN ORDER:

RULE 1 - "smalltalk" is ONLY for greetings, thanks, and social pleasantries with NO question
  ("hi", "hello", "thanks", "how are you", "good morning"). If the message asks ANYTHING
  factual, it is NEVER smalltalk.

RULE 2 - The COST, FEE, RATE, DURATION, or RULES of a service/policy/program is a DOCUMENTS
  question, NOT a database question. Delivery cost/time, gift-wrap fee, reward-points earn rate,
  return windows -> "documents". (The database is only for a specific PRODUCT'S price/stock or a
  specific live record.)

RULE 3 - Use "database" only for specific live data: a count, a list of records, a specific
  order/product/store, a product's price or stock level, what is on promotion right now.

RULE 4 - Use "both" ONLY when a SINGLE question needs a policy explanation AND specific live
  records together (e.g. "can I return a SALE item, and what's on sale right now?"). If the whole
  question can be answered from documents alone, choose "documents" - do NOT add "database".

RULE 5 - For any other real question whose topic is not clearly live database data, default to
  "documents" (it will answer from policy or, if not covered, escalate). Never send a real
  question to "smalltalk".

Examples:
Q: "hi"                                                         -> {{"route": "smalltalk", "reason": "greeting"}}
Q: "thanks!"                                                    -> {{"route": "smalltalk", "reason": "social thanks"}}
Q: "What is your returns policy?"                               -> {{"route": "documents", "reason": "policy question"}}
Q: "How long does standard delivery take and what does it cost?"-> {{"route": "documents", "reason": "delivery time & cost are in the delivery policy"}}
Q: "How many reward points do I earn per £1, and do gift cards earn points?" -> {{"route": "documents", "reason": "reward-points earn rate is policy"}}
Q: "How much is gift wrapping?"                                 -> {{"route": "documents", "reason": "gift-wrap fee is a service cost in policy"}}
Q: "Do you have any sustainability initiatives?"               -> {{"route": "documents", "reason": "company/policy topic, not database"}}
Q: "Do you offer a student loan?"                              -> {{"route": "documents", "reason": "real question; answer from docs or escalate"}}
Q: "How much does the Air Max 270 cost?"                        -> {{"route": "database", "reason": "a specific product price"}}
Q: "How many products are there?"                              -> {{"route": "database", "reason": "a count of records"}}
Q: "Which stores offer click and collect?"                     -> {{"route": "database", "reason": "a list of stores"}}
Q: "What is the status of order 5?"                            -> {{"route": "database", "reason": "a specific order"}}
Q: "Can I return a sale item, and what's on sale right now?"    -> {{"route": "both", "reason": "policy plus live promotion data"}}

Respond with ONLY a JSON object.

Message: {question}
"""
)


@traceable(name="router")
def route_question(question):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    docs_list = ", ".join(get_ingested_docs()) or "(none yet)"
    raw = llm.invoke(ROUTER_PROMPT.format(question=question, documents_list=docs_list)).content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except Exception:
        return {"route": "documents", "reason": "Could not parse router output; defaulting to documents."}


if __name__ == "__main__":
    print("Ingested documents:", get_ingested_docs(), "\n")
    tests = [
        "hi",
        "How long does standard delivery take and what does it cost?",
        "How many reward points do I earn per £1, and do gift cards earn points?",
        "How much is gift wrapping?",
        "Do you have any sustainability initiatives?",
        "Do you offer a student loan?",
        "How many products are there?",
        "What is the status of order 5?",
        "Which stores offer click and collect?",
    ]
    if len(sys.argv) > 1:
        tests = [" ".join(sys.argv[1:])]
    for q in tests:
        d = route_question(q)
        print(f"Q: {q}\n   -> route = {d.get('route')}   (reason: {d.get('reason')})\n")