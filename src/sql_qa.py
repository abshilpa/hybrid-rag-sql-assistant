import sys
import re
from datetime import date
from dotenv import load_dotenv
from langsmith import traceable
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase

load_dotenv()

DB_PATH = "data/db/jd_retail.db"
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

SQL_PROMPT = ChatPromptTemplate.from_template(
    """You are a SQLite expert. Using the schema below, write ONE valid SQLite
SELECT query that answers the question.

Today's date is {today}. In SQLite, date('now') also returns today's date.
- If the question asks about CURRENT / ACTIVE / "now" promotions or offers,
  only include rows where start_date <= date('now') AND end_date >= date('now').
- Dates are stored as TEXT in 'YYYY-MM-DD' format.

Rules:
- Output ONLY the SQL query. No explanation, no markdown fences.
- Use SELECT only. Never write INSERT/UPDATE/DELETE/DROP/ALTER/CREATE.
- Only use tables and columns that exist in the schema.

Schema:
{schema}

Question: {question}

SQL query:"""
)

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """Write a clear, natural-language answer to the question, based only on the
SQL result provided.

Question: {question}
SQL query: {query}
SQL result: {result}

Answer:"""
)

# --- PII protection -------------------------------------------------------
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")


def mask_pii(text):
    """Redact PII (customer emails) BEFORE it reaches the LLM or the UI.
    In production this would use a dedicated PII engine (e.g. Microsoft
    Presidio) to also detect names, phone numbers and addresses."""
    if not text:
        return text
    return EMAIL_RE.sub("[EMAIL REDACTED]", str(text))
# --------------------------------------------------------------------------


def clean_sql(text):
    text = text.strip()
    text = re.sub(r"^```sql", "", text, flags=re.IGNORECASE).strip()
    text = text.replace("```", "").strip()
    return text


def is_safe(query):
    q = query.strip().lower()
    if not q.startswith("select"):
        return False
    forbidden = r"\b(insert|update|delete|drop|alter|create|replace|truncate|grant|revoke)\b"
    if re.search(forbidden, q):
        return False
    if ";" in q.rstrip(";"):
        return False
    return True


@traceable(name="sql_pipeline")
def answer_from_sql(question):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    schema = db.get_table_info()

    sql = clean_sql(llm.invoke(
        SQL_PROMPT.format(schema=schema, question=question, today=date.today().isoformat())
    ).content)

    if not is_safe(sql):
        return "Query blocked for safety (only read-only SELECT queries are allowed).", sql, None

    raw_result = db.run(sql)
    result = mask_pii(raw_result)          #  PII redacted here for security, before it reaches the LLM or the UI

    answer = llm.invoke(
        ANSWER_PROMPT.format(question=question, query=sql, result=result)
    ).content
    return answer, sql, result


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How many products are there?"
    print(f"\nQ: {question}\n")
    answer, sql, result = answer_from_sql(question)
    print("Generated SQL:", sql)
    print("Raw result :", result)
    print("\nA:", answer)