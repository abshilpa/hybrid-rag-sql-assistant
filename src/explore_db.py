import sqlite3
import os

# Path to sqllite database (relative to the project root)
DB_PATH = os.path.join("data", "db", "jd_retail.db")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Get all real tables (skip SQLite's internal ones)
cursor.execute(
    "SELECT name FROM sqlite_master "
    "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
)
tables = [row[0] for row in cursor.fetchall()]

print(f"\nDatabase: {DB_PATH}")
print(f"Found {len(tables)} tables: {tables}\n")

for table in tables:
    print(f"================ {table} ================")
    cursor.execute(f"PRAGMA table_info('{table}');")
    for col in cursor.fetchall():
        # col = (id, name, type, notnull, default, primary_key)
        print(f"   - {col[1]} ({col[2]})")
    cursor.execute(f"SELECT COUNT(*) FROM '{table}';")
    count = cursor.fetchone()[0]
    print(f"   Rows: {count}\n")

conn.close()
print("Done.")