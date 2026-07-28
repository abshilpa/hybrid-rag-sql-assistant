import os
import json
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

FEEDBACK_FILE = "feedback_log.jsonl"


def _now():
    return datetime.now(timezone.utc).isoformat()


def log_feedback(question, answer, route, rating, run_id=None,
                 sources=None, role="customer", comment=""):
    """Record a thumbs up/down for one answer.
    rating: "up" or "down"  ->  score 1.0 / 0.0
    Writes to a local JSONL file AND (best-effort) to LangSmith.
    Returns a short status string for the UI.
    """
    score = 1.0 if rating == "up" else 0.0

    entry = {
        "timestamp": _now(),
        "question": question,
        "answer": answer,
        "route": route,
        "role": role,
        "rating": rating,
        "score": score,
        "run_id": run_id,
        "sources": sources or [],
        "comment": comment,
    }

    # 1. Always log locally — this is our review queue for the golden set
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # 2. Also push to LangSmith so it shows on the trace (best-effort)
    ls_status = "local only"
    if run_id:
        try:
            from langsmith import Client
            Client().create_feedback(
                run_id=run_id,
                key="user_rating",
                score=score,
                comment=comment or f"User pressed {'up' if rating == 'up' else 'down'}",
            )
            ls_status = "logged to LangSmith"
        except Exception as e:
            ls_status = f"local only (LangSmith skipped: {e})"

    return f"Feedback saved ({rating}) - {ls_status}"


def load_feedback():
    """Read all feedback entries back (for review / building the golden set)."""
    if not os.path.exists(FEEDBACK_FILE):
        return []
    rows = []
    with open(FEEDBACK_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summary():
    rows = load_feedback()
    up = sum(1 for r in rows if r.get("rating") == "up")
    down = sum(1 for r in rows if r.get("rating") == "down")
    return {"total": len(rows), "up": up, "down": down,
            "low_rated": [r for r in rows if r.get("rating") == "down"]}


if __name__ == "__main__":
    # quick self-test (no real run_id needed)
    print(log_feedback(
        question="Do you offer a student loan?",
        answer="This topic isn't covered in the current policies...",
        route="documents",
        rating="down",
        role="customer",
    ))
    print(log_feedback(
        question="How much does standard delivery cost?",
        answer="Standard delivery costs £3.99.",
        route="documents",
        rating="up",
        role="customer",
    ))
    s = summary()
    print(f"\nFeedback summary: {s['total']} total | up {s['up']} | down {s['down']}")
    print("Low-rated questions to review for the golden set:")
    for r in s["low_rated"]:
        print(f"   - {r['question']}  (route={r['route']})")