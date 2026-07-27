import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from assistant import answer
from ingest import get_vectorstore

load_dotenv()

TOP_K = 4

# Golden set — expected answers are grounded in the actual policy documents.
EVAL_SET = [
    {"q": "What is your returns policy?", "route": "documents", "doc": "Returns Policy.pdf",
     "reference": "Eligible items can be returned within 30 days of purchase, in original condition with all tags and packaging, not worn outside, with proof of purchase."},
    {"q": "How long do I have to return an item?", "route": "documents", "doc": "Returns Policy.pdf",
     "reference": "Within 30 days of the purchase date."},
    {"q": "My Nike Air Max 270 has faulty stitching after one week. What should I do?", "route": "documents", "doc": "Warranty Policy.docx",
     "reference": "Faulty stitching is a manufacturing defect. Submit a quality claim with proof of purchase, order number, clear photos and a description. The Quality Assessment Team inspects it and normally communicates a decision within 7 business days; outcomes include repair, replacement or refund."},
    {"q": "What types of damage are not covered under the quality policy?", "route": "documents", "doc": "Warranty Policy.docx",
     "reference": "Normal wear and tear such as worn soles, natural creasing, colour fading, minor scuffs, damage during sport, and damage from incorrect washing/drying/storage; also accidental damage, misuse, unauthorised repairs and commercial use."},
    {"q": "How should I care for my trainers?", "route": "documents", "doc": "Warranty Policy.docx",
     "reference": "Clean with a soft brush, cloth and warm water, use mild soap, do not machine wash, air dry naturally, avoid heat sources, and store in a cool dry place away from sunlight."},
    {"q": "How long does standard delivery take and what does it cost?", "route": "documents", "doc": "Delivery Policy.pdf",
     "reference": "Standard delivery takes 3-5 business days and costs 3.99, free on orders over 75."},
    {"q": "How many products are there?", "route": "database", "doc": None,
     "reference": "There are 20 products."},
    {"q": "What is the status of order 5?", "route": "database", "doc": None,
     "reference": "The status of order 5 is In Transit."},
    {"q": "Which stores offer click and collect?", "route": "database", "doc": None,
     "reference": "The stores whose click_collect field is Yes (a list of the participating store names)."},
    {"q": "Which promotions are active right now?", "route": "database", "doc": None,
     "reference": "The promotions whose start_date and end_date span today's date."},
    {"q": "Can I return a discounted item, and which items are on sale now?", "route": "both", "doc": "Returns Policy.pdf",
     "reference": "Discounted items can be returned per the returns policy unless marked Final Sale, plus the list of currently active promotions from the database."},
    {"q": "Do you offer gift wrapping?", "route": "documents", "doc": None,
     "reference": "This is NOT covered in the policy documents. The assistant should say it cannot find it, ask a clarifying question, and offer to connect the customer to a store assistant."},
]

JUDGE_PROMPT = ChatPromptTemplate.from_template(
    """You are grading an AI assistant's answer on ONE dimension, from 1 (very poor)
to 5 (excellent).

Dimension: {dimension}
What it means: {definition}

Question: {question}
{extra}

Assistant's answer:
{answer}

Respond with ONLY the score number (1-5)."""
)


def judge(dimension, definition, question, answer_text, extra=""):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    raw = llm.invoke(JUDGE_PROMPT.format(
        dimension=dimension, definition=definition, question=question,
        answer=answer_text, extra=extra)).content
    m = re.search(r"[1-5]", raw)
    return int(m.group()) if m else 0


def retrieval_metrics(question, expected_doc, k=TOP_K):
    vs = get_vectorstore()
    docs = vs.as_retriever(search_kwargs={"k": k}).invoke(question)
    sources = [d.metadata.get("source") for d in docs]
    rel = sum(1 for s in sources if s == expected_doc)
    total = len(vs.get(where={"source": expected_doc}).get("ids", []))
    return rel / k, (rel / total if total else 0.0), rel > 0


def main():
    router_ok = 0
    precisions, recalls, hits = [], [], []
    corr, faith, rel = [], [], []

    print(f"{'route':10}{'r?':4}{'corr':6}{'faith':7}{'relev':7} question")
    print("-" * 90)

    for case in EVAL_SET:
        result = answer(case["q"])
        ans = result["answer"]

        route_ok = result["route"] == case["route"]
        router_ok += route_ok

        if case["doc"]:
            p, r, h = retrieval_metrics(case["q"], case["doc"])
            precisions.append(p); recalls.append(r); hits.append(h)

        context = " ".join(s["text"] for s in result.get("doc_sources", []))
        if result.get("sql_result"):
            context += " " + str(result["sql_result"])

        c = judge("Correctness", "how well the answer matches the reference facts",
                  case["q"], ans, f"Reference answer: {case['reference']}")
        f = judge("Faithfulness", "whether every claim in the answer is supported by the "
                  "retrieved context (no made-up facts)", case["q"], ans,
                  f"Retrieved context: {context or '(none)'}")
        rl = judge("Answer relevance", "how directly the answer addresses the question",
                   case["q"], ans)
        corr.append(c); faith.append(f); rel.append(rl)

        flag = "OK" if route_ok else "XX"
        print(f"{result['route']:10}{flag:4}{c:<6}{f:<7}{rl:<7} {case['q'][:45]}")

    n = len(EVAL_SET)
    print("\n================ SCORECARD ================")
    print(f"Router accuracy       : {router_ok}/{n} = {router_ok/n:.0%}")
    print(f"Mean Correctness      : {sum(corr)/len(corr):.2f} / 5")
    print(f"Mean Faithfulness     : {sum(faith)/len(faith):.2f} / 5")
    print(f"Mean Answer Relevance : {sum(rel)/len(rel):.2f} / 5")
    if precisions:
        print(f"Mean Precision@{TOP_K}     : {sum(precisions)/len(precisions):.2f}")
        print(f"Mean Recall@{TOP_K}        : {sum(recalls)/len(recalls):.2f}")
        print(f"Hit Rate@{TOP_K}           : {sum(hits)/len(hits):.0%}")
    print("===========================================")


if __name__ == "__main__":
    main()