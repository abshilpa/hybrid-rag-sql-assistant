import os
import glob
import hashlib
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
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


def ingest_files(file_paths, replace=True):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    vs = get_vectorstore()

    added_total = 0
    changed_any = False

    for path in file_paths:
        name = os.path.basename(path)

        # 1) Build the new chunks; each chunk's content-hash is its stable ID
        docs = load_file(path)
        chunks = splitter.split_documents(docs)
        for c in chunks:
            c.metadata["source"] = name
            c.metadata.setdefault("page", 0)          # docx/txt have no pages
            c.metadata["chunk_hash"] = _chunk_hash(name, c.page_content)

        # 2) What's already stored for this document?
        existing = vs.get(where={"source": name})
        existing_ids = set(existing.get("ids", []))

        # 3) Diff by hash: embed only new/changed chunks, drop only removed ones
        new_id_set, seen, to_add = set(), set(), []
        for c in chunks:
            cid = c.metadata["chunk_hash"]
            new_id_set.add(cid)
            if cid in existing_ids or cid in seen:
                continue                              # already embedded -> skip
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

    # Freshness: only clear the answer cache if content actually changed
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