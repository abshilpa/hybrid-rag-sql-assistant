import os
import glob
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


def ingest_files(file_paths, replace=True):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    vs = get_vectorstore()

    total = 0
    for path in file_paths:
        name = os.path.basename(path)

        if replace:
            existing = vs.get(where={"source": name})
            ids = existing.get("ids", [])
            if ids:
                vs.delete(ids=ids)
                print(f"   Removed {len(ids)} old chunks for {name}")

        docs = load_file(path)
        chunks = splitter.split_documents(docs)
        for c in chunks:
            c.metadata["source"] = name
            c.metadata.setdefault("page", 0)   # docx/txt have no pages
        vs.add_documents(chunks)
        total += len(chunks)
        print(f"   {name}: {len(docs)} section(s) -> {len(chunks)} chunks")

    print(f"\nDone. {total} chunks added this run.")
    return total


if __name__ == "__main__":
    files = []
    for ext in SUPPORTED:
        files.extend(glob.glob(os.path.join(DOCS_DIR, f"*{ext}")))
    print(f"Found {len(files)} file(s) in {DOCS_DIR}:")
    for p in files:
        print("   -", os.path.basename(p))
    print()
    ingest_files(files)