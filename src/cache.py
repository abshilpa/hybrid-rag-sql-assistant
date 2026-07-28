import os
import json
import math
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

CACHE_FILE = "response_cache.json"
SIM_THRESHOLD = 0.96
_embedder = OpenAIEmbeddings(model="text-embedding-3-small")


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _load():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save(cache):
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception:
        pass


def get_cached(question, role="customer"):
    """Return (result, similarity) if a semantically-similar question is cached for this role."""
    try:
        emb = _embedder.embed_query(question)
    except Exception:
        return None, 0.0
    best, best_sim = None, 0.0
    for entry in _load():
        if entry.get("role") != role:
            continue
        sim = _cosine(emb, entry.get("embedding", []))
        if sim > best_sim:
            best_sim, best = sim, entry
    if best and best_sim >= SIM_THRESHOLD:
        return best["result"], best_sim
    return None, best_sim


def add_to_cache(question, result, role="customer"):
    try:
        emb = _embedder.embed_query(question)
    except Exception:
        return
    cache = _load()
    cache.append({"question": question, "embedding": emb, "result": result, "role": role})
    _save(cache)


def clear_cache():
    _save([])