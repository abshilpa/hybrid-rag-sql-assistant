import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from assistant import qa_graph   #  compiled LangGraph app

# Get the Mermaid source straight from the current graph
mermaid = qa_graph.get_graph().draw_mermaid()

# Strip LangGraph's config frontmatter so GitHub renders it cleanly
lines = mermaid.splitlines()
if lines and lines[0].strip() == "---":
    try:
        end = lines.index("---", 1)
        lines = lines[end + 1:]
    except ValueError:
        pass
mermaid_clean = "\n".join(lines).strip()

# 1) Save a PNG (needs internet)
try:
    png = qa_graph.get_graph().draw_mermaid_png()
    with open("architecture_graph.png", "wb") as f:
        f.write(png)
    print("✅ Wrote architecture_graph.png")
except Exception as e:
    print("⚠️  Could not render PNG (needs internet):", e)

# 2) Write architecture.md — GitHub renders the ```mermaid block as a live diagram
md = (
    "# Architecture\n\n"
    "This diagram is **auto-generated** from the LangGraph app in `src/assistant.py`.\n"
    "Whenever the graph changes, regenerate it with:\n\n"
    "```bash\n"
    "python src/gen_diagram.py\n"
    "```\n\n"
    "```mermaid\n"
    f"{mermaid_clean}\n"
    "```\n"
)
with open("architecture.md", "w", encoding="utf-8") as f:
    f.write(md)
print("✅ Wrote architecture.md")
