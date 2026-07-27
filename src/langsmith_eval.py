import re
from dotenv import load_dotenv
from langsmith import Client, evaluate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from assistant import answer
from evaluate import EVAL_SET   # reuse the golden set we already built

load_dotenv()

DATASET_NAME = "qa-assistant-golden"

JUDGE_PROMPT = ChatPromptTemplate.from_template(
    """Grade the AI answer on ONE dimension from 1 (poor) to 5 (excellent).
Dimension: {dimension}
Meaning: {definition}
Question: {question}
{extra}
Answer: {answer}
Respond with ONLY the number (1-5)."""
)


def judge(dimension, definition, question, answer_text, extra=""):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    raw = llm.invoke(JUDGE_PROMPT.format(dimension=dimension, definition=definition,
                                         question=question, answer=answer_text, extra=extra)).content
    m = re.search(r"[1-5]", raw)
    return (int(m.group()) / 5.0) if m else 0.0


# ---- 1) Upload the golden dataset (only once) ----
client = Client()
if not client.has_dataset(dataset_name=DATASET_NAME):
    ds = client.create_dataset(DATASET_NAME, description="Golden Q&A set for the retail assistant")
    client.create_examples(
        dataset_id=ds.id,
        inputs=[{"question": c["q"]} for c in EVAL_SET],
        outputs=[{"reference": c["reference"], "route": c["route"]} for c in EVAL_SET],
    )
    print(f"Created dataset '{DATASET_NAME}'.")
else:
    print(f"Dataset '{DATASET_NAME}' already exists.")


# ---- 2) Target: run the assistant ----
def target(inputs):
    result = answer(inputs["question"])
    ctx = " ".join(s["text"] for s in result.get("doc_sources", []))
    if result.get("sql_result"):
        ctx += " " + str(result["sql_result"])
    return {"answer": result["answer"], "route": result["route"], "context": ctx}


# ---- 3) Evaluators ----
def correctness(run, example):
    return {"key": "correctness",
            "score": judge("Correctness", "matches the reference facts",
                           example.inputs["question"], run.outputs["answer"],
                           f"Reference: {example.outputs['reference']}")}

def faithfulness(run, example):
    return {"key": "faithfulness",
            "score": judge("Faithfulness", "every claim supported by the context (no hallucination)",
                           example.inputs["question"], run.outputs["answer"],
                           f"Context: {run.outputs.get('context') or '(none)'}")}

def answer_relevance(run, example):
    return {"key": "answer_relevance",
            "score": judge("Answer relevance", "directly addresses the question",
                           example.inputs["question"], run.outputs["answer"])}

def routing_correct(run, example):
    return {"key": "routing_correct",
            "score": 1.0 if run.outputs["route"] == example.outputs["route"] else 0.0}


# ---- 4) Run the experiment ----
if __name__ == "__main__":
    evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[correctness, faithfulness, answer_relevance, routing_correct],
        experiment_prefix="qa-eval",
    )
    print("Done — open LangSmith → Datasets & Experiments to see the scores.")