import sys
import re
import sqlite3
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
- If the question asks about CURRENT / ACTIVE / "now" promotions, only include rows where
  start_date <= date('now') AND end_date >= date('now').
- Dates are stored as TEXT in 'YYYY-MM-DD' format.

{value_hints}When filtering a categorical column, use the EXACT values listed above, and map
the user's wording to the closest valid value (e.g. "in process" -> "Processing",
"transit" -> "In Transit"). If the intent matches several values, include them all.

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

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")


def _get_known_customer_names():
    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute("SELECT DISTINCT customer_name FROM Orders;")
        names = [r[0] for r in cur.fetchall() if r[0]]
        con.close()
        return names
    except Exception:
        return []


KNOWN_NAMES = _get_known_customer_names()


def _get_value_hints():
    """Distinct values of low-cardinality categorical columns, so the LLM uses exact values."""
    hints = {}
    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        for table, col in [("Orders", "order_status"), ("Stores", "click_collect"),
                           ("Products", "category"), ("Products", "brand")]:
            try:
                cur.execute(f"SELECT DISTINCT {col} FROM {table};")
                vals = [str(r[0]) for r in cur.fetchall() if r[0] is not None]
                if 0 < len(vals) <= 30:
                    hints[f"{table}.{col}"] = vals
            except Exception:
                pass
        con.close()
    except Exception:
        pass
    return hints


VALUE_HINTS = _get_value_hints()


def _format_hints():
    if not VALUE_HINTS:
        return ""
    lines = [f"- {k} is one of: {', '.join(repr(v) for v in vs)}" for k, vs in VALUE_HINTS.items()]
    return "Known valid column values (use these EXACT values in WHERE clauses):\n" + "\n".join(lines) + "\n\n"


def mask_pii(text, mask=True):
    if not text or not mask:
        return text
    text = EMAIL_RE.sub("[EMAIL REDACTED]", str(text))
    for name in KNOWN_NAMES:
        if name:
            text = text.replace(name, "[NAME REDACTED]")
    return text


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

def _selects_pii(sql):
    """True if the SELECT clause explicitly reads customer names or emails."""
    m = re.search(r"select\s+(.*?)\s+from", sql, flags=re.IGNORECASE | re.DOTALL)
    clause = (m.group(1) if m else sql).lower()
    return "customer_name" in clause or "customer_email" in clause




@traceable(name="sql_pipeline")
def answer_from_sql(question, mask=True):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    schema = db.get_table_info()

    sql = clean_sql(llm.invoke(
        SQL_PROMPT.format(schema=schema, question=question,
                          today=date.today().isoformat(), value_hints=_format_hints())
    ).content)

    if not is_safe(sql):
        return "Query blocked for safety (only read-only SELECT queries are allowed).", sql, None

    # PII governance: non-admin roles may not read customer names or emails.
    # Refuse politely instead of running the query and masking the results.
    if mask and _selects_pii(sql):
        msg = ("I'm sorry, but I can't share customer names or contact details, as that "
               "information is confidential. I'd be glad to help with product details, "
               "prices, stock, orders, promotions, or store information instead.")
        return msg, None, None

    raw_result = db.run(sql)
    result = mask_pii(raw_result, mask=mask)
    ...

    answer = llm.invoke(
        ANSWER_PROMPT.format(question=question, query=sql, result=result)
    ).content
    return answer, sql, result


if __name__ == "__main__":
    admin = "--admin" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--admin"]
    question = " ".join(args) or "how many orders are in process and transit?"
    print(f"\nQ: {question}   (role: {'admin' if admin else 'customer'})\n")
    answer, sql, result = answer_from_sql(question, mask=not admin)
    print("Generated SQL:", sql)
    print("Raw result :", result)
    print("\nA:", answer)