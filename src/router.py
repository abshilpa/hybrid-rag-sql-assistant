import sys
import json
from dotenv import load_dotenv
from langsmith import traceable
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

ROUTER_PROMPT = ChatPromptTemplate.from_template(
    """You are a routing assistant for a retail Q&A system. Decide how to handle
the user's message.

Sources available:
1. DOCUMENTS (policy PDFs): returns & refunds, delivery, click & collect,
   promotions guide (rules/terms), product care, warranty.
2. DATABASE (SQL): Products, Orders, Inventory, Promotions, Stores.

Routing rules:
- "smalltalk" -> greetings, thanks, general chit-chat, or general shopping advice
  (e.g. "hi", "how are you", "what can you do", "which running shoes suit rain?").
- "documents" -> asks about a POLICY, a capability, or HOW something works.
  Yes/No and "what is your ... policy" questions go here.
  Examples: "what is your returns policy?", "do you offer click and collect?",
  "how does delivery work?".
- "database" -> asks for SPECIFIC LIVE DATA: a count, a LIST, or a specific
  order/product/store. "which", "list", "how many", and named order/store/product
  questions go here.
  Examples: "which stores offer click and collect?", "how many products are there?",
  "what is the status of order 5?".
- "both" -> needs BOTH a policy AND live data.

Respond with ONLY a JSON object, e.g.:
{{"route": "documents", "reason": "asks whether click & collect is offered"}}

Message: {question}
"""
)


@traceable(name="router")
def route_question(question):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    raw = llm.invoke(ROUTER_PROMPT.format(question=question)).content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        decision = json.loads(raw)
    except Exception:
        decision = {"route": "both", "reason": "Could not parse router output; defaulting to both."}
    return decision


if __name__ == "__main__":
    tests = ["hi", "how are you?", "which running shoes are good in the rain?",
             "What is your returns policy?", "How many products are on promotion?",
             "What is the status of order 5?"]
    if len(sys.argv) > 1:
        tests = [" ".join(sys.argv[1:])]
    for q in tests:
        d = route_question(q)
        print(f"Q: {q}\n   -> route = {d.get('route')}   (reason: {d.get('reason')})\n")