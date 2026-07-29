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
1. DOCUMENTS (knowledge base files). Currently contains these documents:
   {documents_list}
   Documents cover policies, capabilities, how things work, the COST / FEE / RATE / RULES of
   services and programs (delivery, gift wrapping, click & collect, returns, warranty, rewards),
   AND — if a product-information guide is present — product DESCRIPTIONS, materials, fit and
   sizing, and care instructions.
2. DATABASE (SQL): live records only -> Products (a product's PRICE or STOCK), Orders (a specific
   order's status/tracking), Inventory (stock per store), Promotions (what is discounted, with
   dates), Stores (list of stores, cities, hours, click & collect, phone).

Choose ONE route using these rules IN ORDER:

RULE 1 - "smalltalk" is ONLY for greetings, thanks, and social pleasantries with NO question.
  If the message asks ANYTHING factual, it is NEVER smalltalk.

RULE 2 - The COST, FEE, RATE, DURATION, or RULES of a service/policy/program is a DOCUMENTS
  question (delivery cost, gift-wrap fee, reward-point rates, return windows).

RULE 3 - Use "database" for specific LIVE data only: a count, a list of records, a specific
  order/product/store, a product's PRICE or STOCK level, what is on promotion right now.

RULE 4 - Use "both" when a question needs DOCUMENT content AND live records together. This
  includes: a policy plus live data, OR a GENERAL request for information / a description of a
  specific product ("tell me about X", "give me info on X", "describe X") when a product-
  information guide exists — the description comes from documents, the price/stock from the database.

RULE 5 - A question purely about a product's DESCRIPTION, materials, fit, or care (with no ask
  for price or stock) is "documents".

RULE 6 - For any other real question not clearly live database data, default to "documents".

Examples:
Q: "hi"                                                    -> {{"route": "smalltalk", "reason": "greeting"}}
Q: "How much does standard delivery cost?"                 -> {{"route": "documents", "reason": "service cost is policy"}}
Q: "How much is the Adidas Samba OG?"                       -> {{"route": "database", "reason": "a product price"}}
Q: "How many Nike products are there?"                      -> {{"route": "database", "reason": "a count of records"}}
Q: "What is the Adidas Samba OG made of?"                   -> {{"route": "documents", "reason": "product description in the guide"}}
Q: "Tell me about the Adidas Samba OG"                      -> {{"route": "both", "reason": "product description from docs plus price/stock from db"}}
Q: "Give me the info about the Nike Air Max 270"           -> {{"route": "both", "reason": "description from docs plus live price/stock"}}
Q: "What is your returns policy?"                           -> {{"route": "documents", "reason": "policy question"}}
Q: "Can I return a sale item, and what's on sale now?"     -> {{"route": "both", "reason": "policy plus live promotion data"}}

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