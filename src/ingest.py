import os
import re
import glob
import hashlib
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DOCS_DIR = os.path.join("data", "documents")
CHROMA_DIR = "chroma_store"
COLLECTION_NAME = "documents"
SUPPORTED = (".pdf", ".docx", ".txt", ".md")


def get_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


def load_file(path):
    """Choose the correct loader based on the file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(path).load()
    elif ext == ".docx":
        return Docx2txtLoader(path).load()
    elif ext in (".txt", ".md"):
        return TextLoader(path, encoding="utf-8").load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _chunk_hash(source, content):
    """Stable fingerprint of a chunk (its text within a given document)."""
    h = hashlib.md5()
    h.update(source.encode("utf-8"))
    h.update(b"::")
    h.update(content.encode("utf-8"))
    return h.hexdigest()


# --- Structure-aware chunking -------------------------------------------------
# Split a document at its natural section/entry boundaries (each policy section,
# each product, each catalogue entry) instead of blind fixed-size cuts. Any
# section that is still too long falls back to recursive splitting.

_MAX_SECTION_CHARS = 1000
_MIN_SECTION_CHARS = 60
_fallback_splitter = RecursiveCharacterTextSplitter(
    chunk_size=_MAX_SECTION_CHARS, chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def _is_heading(line):
    """A short title-like line that starts a new section (e.g. '2. Gift Wrapping',
    'The Last Horizon', 'Nike Air Max 270') — not a bullet, field, or sentence."""
    s = line.strip()
    return (
        bool(s)
        and len(s) <= 60
        and s[0] not in "*-•"
        and ":" not in s
        and not s.endswith((".", ",", ";", "!", "?"))
        and len(s.split()) <= 8
        and bool(re.search(r"[A-Za-z]", s))
    )


def structure_split(text):
    lines = text.split("\n")
    flags = [_is_heading(l) for l in lines]

    # Suppress runs of >= 3 consecutive heading-like lines — those are list items
    # (e.g. bullet lists that lost their markers), not real section headings.
    i = 0
    while i < len(lines):
        if flags[i]:
            j = i
            while j < len(lines) and (flags[j] or not lines[j].strip()):
                j += 1
            run = [k for k in range(i, j) if flags[k]]
            if len(run) >= 3:
                for k in run:
                    flags[k] = False
            i = j
        else:
            i += 1

    # Group each heading with the body that follows it, up to the next heading.
    sections, cur = [], []
    for idx, l in enumerate(lines):
        if flags[idx] and any(x.strip() for x in cur):
            sections.append("\n".join(cur).strip())
            cur = [l]
        else:
            cur.append(l)
    if cur:
        sections.append("\n".join(cur).strip())
    sections = [s for s in sections if s.strip()]

    # Merge tiny lone-heading sections into the next one.
    merged = []
    for s in sections:
        if merged and len(merged[-1]) < _MIN_SECTION_CHARS:
            merged[-1] = merged[-1] + "\n" + s
        else:
            merged.append(s)

    # Any section still too long is recursively split.
    out = []
    for s in merged:
        out.extend([s] if len(s) <= _MAX_SECTION_CHARS else _fallback_splitter.split_text(s))
    return out


def ingest_files(file_paths, replace=True):
    vs = get_vectorstore()
    added_total = 0
    changed_any = False

    for path in file_paths:
        name = os.path.basename(path)

        # 1) Load, then split each page/section structurally
        docs = load_file(path)
        chunks = []
        for d in docs:
            for piece in structure_split(d.page_content):
                c = Document(page_content=piece, metadata=dict(d.metadata))
                c.metadata["source"] = name
                c.metadata.setdefault("page", d.metadata.get("page", 0))
                c.metadata["chunk_hash"] = _chunk_hash(name, piece)
                chunks.append(c)

        # 2) What's already stored for this document?
        existing = vs.get(where={"source": name})
        existing_ids = set(existing.get("ids", []))

        # 3) Diff by hash: embed only new/changed chunks, drop only removed ones
        new_id_set, seen, to_add = set(), set(), []
        for c in chunks:
            cid = c.metadata["chunk_hash"]
            new_id_set.add(cid)
            if cid in existing_ids or cid in seen:
                continue
            seen.add(cid)
            to_add.append(c)

        to_delete = [i for i in existing_ids if i not in new_id_set]

        if to_delete:
            vs.delete(ids=to_delete)
        if to_add:
            vs.add_documents(to_add, ids=[c.metadata["chunk_hash"] for c in to_add])

        if to_add or to_delete:
            changed_any = True

        unchanged = len(new_id_set & existing_ids)
        added_total += len(to_add)
        print(f"   {name}: {len(chunks)} chunk(s) -> "
              f"{len(to_add)} new/changed, {unchanged} unchanged, {len(to_delete)} removed")

    print(f"\nDone. {added_total} chunk(s) embedded this run (unchanged chunks were skipped).")

    if changed_any:
        try:
            from cache import clear_cache
            clear_cache()
            print("   Content changed -> answer cache cleared.")
        except Exception:
            pass
    else:
        print("   No content changed -> answer cache kept.")

    return added_total


if __name__ == "__main__":
    files = []
    for ext in SUPPORTED:
        files.extend(glob.glob(os.path.join(DOCS_DIR, f"*{ext}")))
    print(f"Found {len(files)} file(s) in {DOCS_DIR}:")
    for p in files:
        print("   -", os.path.basename(p))
    print()
    ingest_files(files)