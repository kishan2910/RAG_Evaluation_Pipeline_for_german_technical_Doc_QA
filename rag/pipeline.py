"""
The RAG query pipeline: embed → retrieve → generate.

Two entry points:

    contexts = await retrieve(cfg, session, "question text")
    result   = await rag_query(cfg, session, "question text")

`rag_query` returns a dict with the answer, retrieved contexts, and per-stage
timing — easy to integrate into server.py or a benchmark harness.
"""

import time

import aiohttp

from .client import chat, embed
from .config import RagConfig
from .ingest import open_collection


# Tweak this prompt for your domain; defaults to German since the bundled
# evaluation set is German technical content.
RAG_SYSTEM_PROMPT = (
    "Du bist ein präziser technischer Assistent. Beantworte die Frage "
    "ausschließlich anhand des bereitgestellten Kontexts. Wenn die Antwort "
    "nicht eindeutig aus dem Kontext hervorgeht, antworte: "
    "'Keine Information im Dokument gefunden.' "
    "Antworte kurz, faktentreu und ohne Spekulation."
)


def build_messages(question: str, contexts: list[dict]) -> list[dict]:
    """Assemble the chat messages list from retrieved chunks + the question."""
    blocks = []
    for c in contexts:
        meta = c.get("metadata") or {}
        page = meta.get("page", "?")
        src  = meta.get("source", "?")
        blocks.append(f"[Quelle: {src}, Seite {page}]\n{c['text']}")
    context_str = "\n\n---\n\n".join(blocks) if blocks else "(kein Kontext)"
    user = (
        f"Kontext:\n{context_str}\n\n"
        f"Frage: {question}\n\n"
        f"Antwort:"
    )
    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


async def retrieve(
    cfg: RagConfig,
    session: aiohttp.ClientSession,
    query: str,
    *,
    top_k: int | None = None,
) -> list[dict]:
    """Embed the query and return the top-k chunks from Chroma.

    Each result has: id, text, metadata, distance.
    """
    k = top_k if top_k is not None else cfg.top_k
    [query_emb] = await embed(session, cfg.embedding, [query], timeout=cfg.request_timeout)

    collection = open_collection(cfg.chroma_dir, cfg.collection)
    raw = collection.query(query_embeddings=[query_emb], n_results=k)

    ids       = (raw.get("ids")       or [[]])[0]
    docs      = (raw.get("documents") or [[]])[0]
    metas     = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]

    return [
        {"id": ids[i], "text": docs[i], "metadata": metas[i], "distance": distances[i]}
        for i in range(len(ids))
    ]


async def rag_query(
    cfg: RagConfig,
    session: aiohttp.ClientSession,
    question: str,
    *,
    top_k: int | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dict:
    """Full pipeline: retrieve + generate. Returns:

        {
          "question": str,
          "answer":   str,
          "contexts": [chunk dicts],
          "usage":    {prompt_tokens, completion_tokens, total_tokens} | {},
          "timings":  {retrieve: float, generate: float, total: float},
        }
    """
    t0 = time.perf_counter()
    contexts = await retrieve(cfg, session, question, top_k=top_k)
    t_retrieve = time.perf_counter() - t0

    messages = build_messages(question, contexts)
    t1 = time.perf_counter()
    out = await chat(
        session, cfg.chat, messages,
        temperature=cfg.chat_temperature if temperature is None else temperature,
        max_tokens=cfg.chat_max_tokens if max_tokens is None else max_tokens,
        timeout=cfg.request_timeout,
    )
    t_generate = time.perf_counter() - t1

    return {
        "question": question,
        "answer":   out["text"],
        "contexts": contexts,
        "usage": {
            "prompt_tokens":     out.get("prompt_tokens"),
            "completion_tokens": out.get("completion_tokens"),
            "total_tokens":      out.get("total_tokens"),
        },
        "timings": {
            "retrieve": t_retrieve,
            "generate": t_generate,
            "total":    t_retrieve + t_generate,
        },
    }
