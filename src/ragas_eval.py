
import time
from langsmith import Client, traceable
try:
    from langsmith import get_current_run_tree
except ImportError:
    from langsmith.run_helpers import get_current_run_tree


# ---- Shim the unused Vertex AI import that ragas eagerly loads (we only use OpenAI) ----
import sys, types
_vx = "langchain_community.chat_models.vertexai"
try:
    __import__(_vx)
except Exception:
    _m = types.ModuleType(_vx)
    _m.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules[_vx] = _m
# ---------------------------------------------------------------------------------------


import os
from dotenv import load_dotenv
import nest_asyncio
nest_asyncio.apply()   # ragas runs async under the hood; this keeps it happy on Windows

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from doc_qa import answer_from_docs

# ragas imports (written to tolerate small naming differences across versions)
from ragas import evaluate, EvaluationDataset
try:
    from ragas import SingleTurnSample
except ImportError:
    from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas.metrics import Faithfulness
try:
    from ragas.metrics import ResponseRelevancy as AnswerRel
except ImportError:
    from ragas.metrics import AnswerRelevancy as AnswerRel
try:
    from ragas.metrics import LLMContextPrecisionWithReference as CtxPrec
except ImportError:
    from ragas.metrics import ContextPrecision as CtxPrec
try:
    from ragas.metrics import LLMContextRecall as CtxRecall
except ImportError:
    from ragas.metrics import ContextRecall as CtxRecall

load_dotenv()

# Small golden subset of document/RAG questions, each with a reference (ground-truth) answer
EVAL_SET = [
    {
        "question": "How long does standard delivery take and what does it cost?",
        "reference": "Standard delivery takes 3 to 5 business days and costs £3.99.",
    },
    {
        "question": "How many reward points do I earn per £1, and do gift cards earn points?",
        "reference": "You earn 10 reward points for every £1 spent on eligible purchases. Gift cards do not earn reward points.",
    },
    {
        "question": "How much does gift wrapping cost?",
        "reference": "Online gift wrapping is a flat fee of £2.99 per item; in-store gift wrapping is free during November and December.",
    },
]


def build_samples():
    samples = []
    for row in EVAL_SET:
        q = row["question"]
        answer_text, docs = answer_from_docs(q)          # run YOUR pipeline
        contexts = [d.page_content for d in docs]         # the chunks it retrieved
        print(f"   built sample: {q[:55]}...  ({len(contexts)} contexts)")
        samples.append(SingleTurnSample(
            user_input=q,
            response=answer_text,
            retrieved_contexts=contexts,
            reference=row["reference"],
        ))
    return samples

@traceable(name="ragas_eval")
def _traced_run(question, answer, reference):
    """A small traced run so each question's RAGAS scores have something to attach to."""
    try:
        rt = get_current_run_tree()
        return str(rt.id) if rt is not None else None
    except Exception:
        return None


def log_ragas_to_langsmith(df):
    client = Client()
    score_keys = [c for c in df.columns if c in (
        "faithfulness", "answer_relevancy",
        "llm_context_precision_with_reference", "context_recall")]
    run_ids = []
    for _, row in df.iterrows():
        rid = _traced_run(row["user_input"], row.get("response", ""), row.get("reference", ""))
        run_ids.append((rid, row))
    time.sleep(3)  # give the traces a moment to reach LangSmith before attaching feedback
    logged = 0
    for rid, row in run_ids:
        if not rid:
            continue
        for k in score_keys:
            try:
                client.create_feedback(run_id=rid, key=f"ragas_{k}", score=float(row[k]))
            except Exception as e:
                print("   feedback error:", e)
        logged += 1
    print(f"\nLogged RAGAS scores for {logged} question(s) to LangSmith "
          f"project '{os.getenv('LANGSMITH_PROJECT', 'default')}'.")






def main():
    print("Generating answers + retrieved contexts from your pipeline...\n")
    samples = build_samples()
    dataset = EvaluationDataset(samples=samples)

    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    evaluator_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    metrics = [Faithfulness(), AnswerRel(), CtxPrec(), CtxRecall()]

    print("\nScoring with RAGAS (faithfulness, answer relevancy, context precision, context recall)...\n")
    result = evaluate(dataset=dataset, metrics=metrics, llm=evaluator_llm, embeddings=evaluator_emb)

    print("\n===== RAGAS RESULTS (averaged) =====")
    print(result)
    df = None
    try:
        df = result.to_pandas()
        skip = {"user_input", "response", "retrieved_contexts", "reference"}
        score_cols = [c for c in df.columns if c not in skip]
        print("\nPer-question scores:\n")
        print(df[["user_input"] + score_cols].to_string(index=False))
    except Exception as e:
        print("(could not render per-question table:", e, ")")

    if df is not None:
        print("\nLogging RAGAS scores to LangSmith...")
        log_ragas_to_langsmith(df)


if __name__ == "__main__":
    main()
