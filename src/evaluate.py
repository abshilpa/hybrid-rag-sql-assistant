from dotenv import load_dotenv

from assistant import answer
from ingest import get_vectorstore

load_dotenv()

TOP_K = 4

# Test set: expected route, expected source doc (for retrieval eval), expected answer keywords.
# TIP: add your own questions from the "Returns Policy-Demo Questions" file here.
EVAL_SET = [
    {"q": "What is your returns policy?",              "route": "documents", "doc": "Returns Policy.pdf",         "keywords": ["30 days", "30"]},
    {"q": "How long do I have to return an item?",     "route": "documents", "doc": "Returns Policy.pdf",         "keywords": ["30"]},
    {"q": "Do you offer click and collect?",           "route": "documents", "doc": "Click & Collect Policy.pdf", "keywords": ["collect"]},
    {"q": "How long does delivery take?",              "route": "documents", "doc": "Delivery Policy.pdf",        "keywords": ["day"]},
    {"q": "How should I care for my trainers?",        "route": "documents", "doc": "Product Care Guide.pdf",     "keywords": ["clean"]},
    {"q": "How many products are there?",              "route": "database",  "doc": None,                        "keywords": ["20"]},
    {"q": "What is the status of order 5?",            "route": "database",  "doc": None,                        "keywords": ["transit", "order"]},
    {"q": "Which stores offer click and collect?",     "route": "database",  "doc": None,                        "keywords": ["London"]},
    {"q": "Which promotions are active right now?",    "route": "database",  "doc": None,                        "keywords": ["%", "sale", "off"]},
    {"q": "Can I return a discounted item, and which items are on sale now?", "route": "both", "doc": "Returns Policy.pdf", "keywords": ["final sale", "sale", "return"]},
    {"q": "hi",                                        "route": "smalltalk", "doc": None,                        "keywords": []},
    {"q": "what can you do for me?",                   "route": "smalltalk", "doc": None,                        "keywords": []},
]


def retrieval_metrics(question, expected_doc, k=TOP_K):
    """Precision@k, Recall@k and Hit-rate for the retriever (no reranking)."""
    vs = get_vectorstore()
    docs = vs.as_retriever(search_kwargs={"k": k}).invoke(question)
    sources = [d.metadata.get("source") for d in docs]
    relevant_in_topk = sum(1 for s in sources if s == expected_doc)
    total_relevant = len(vs.get(where={"source": expected_doc}).get("ids", []))
    precision = relevant_in_topk / k
    recall = relevant_in_topk / total_relevant if total_relevant else 0.0
    hit = relevant_in_topk > 0
    return precision, recall, hit


def main():
    router_correct = 0
    answer_correct = 0
    answer_total = 0
    precisions, recalls, hits = [], [], []

    print(f"{'route':10} {'ok':3} {'retrieval':30} {'ans':5} question")
    print("-" * 90)

    for case in EVAL_SET:
        result = answer(case["q"])
        pred_route = result["route"]
        route_ok = (pred_route == case["route"])
        router_correct += route_ok

        if case["doc"]:
            p, r, h = retrieval_metrics(case["q"], case["doc"])
            precisions.append(p); recalls.append(r); hits.append(h)
            retr = f"P@{TOP_K}={p:.2f} R@{TOP_K}={r:.2f} hit={'Y' if h else 'N'}"
        else:
            retr = "-"

        if case["keywords"]:
            ans_low = result["answer"].lower()
            ok = any(kw.lower() in ans_low for kw in case["keywords"])
            answer_correct += ok
            answer_total += 1
            ans = "PASS" if ok else "FAIL"
        else:
            ans = "-"

        flag = "OK" if route_ok else "XX"
        print(f"{pred_route:10} {flag:3} {retr:30} {ans:5} {case['q'][:45]}")

    n = len(EVAL_SET)
    print("\n================ SCORECARD ================")
    print(f"Router accuracy       : {router_correct}/{n} = {router_correct/n:.0%}")
    if answer_total:
        print(f"Answer keyword match  : {answer_correct}/{answer_total} = {answer_correct/answer_total:.0%}")
    if precisions:
        print(f"Mean Precision@{TOP_K}     : {sum(precisions)/len(precisions):.2f}")
        print(f"Mean Recall@{TOP_K}        : {sum(recalls)/len(recalls):.2f}")
        print(f"Hit Rate@{TOP_K}           : {sum(hits)/len(hits):.0%}")
    print("===========================================")


if __name__ == "__main__":
    main()