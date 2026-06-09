from __future__ import annotations

import time

from openai import AsyncOpenAI

from mlflow_tracking import start_span
from openai_compat import chat, embed
from settings import RagEvalConfig
from vectorstore import open_collection

RAG_SYSTEM_PROMPT = (
    "Du bist ein praziser technischer Assistent. Beantworte die Frage "
    "ausschliesslich anhand des bereitgestellten Kontexts. Wenn die Antwort "
    "nicht eindeutig aus dem Kontext hervorgeht, antworte: "
    "'Keine Information im Dokument gefunden.' "
    "Antworte kurz, faktentreu und ohne Spekulation."
)


def build_messages(question: str, contexts: list[dict]) -> list[dict[str, str]]:
    blocks: list[str] = []
    for ctx in contexts:
        metadata = ctx.get("metadata") or {}
        blocks.append(
            f"[Quelle: {metadata.get('source', '?')}, Seite {metadata.get('page', '?')}]\n{ctx['text']}"
        )

    context_text = "\n\n---\n\n".join(blocks) if blocks else "(kein Kontext)"
    user_prompt = (
        f"Kontext:\n{context_text}\n\n"
        f"Frage: {question}\n\n"
        "Antwort:"
    )
    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


async def retrieve(
    cfg: RagEvalConfig,
    embedding_client: AsyncOpenAI,
    query: str,
    *,
    top_k: int | None = None,
    collection=None,
) -> list[dict]:
    k = top_k if top_k is not None else cfg.top_k
    with start_span(
        "rag.retrieve",
        span_type="RETRIEVER",
        attributes={"top_k": k, "collection": cfg.collection},
    ) as span:
        span.set_inputs({"query": query, "top_k": k})
        query_embedding = await embed(embedding_client, cfg.embedding, [query])
        if collection is None:
            collection = open_collection(cfg.chroma_dir, cfg.collection)
        raw = collection.query(query_embeddings=query_embedding, n_results=k)

        ids = (raw.get("ids") or [[]])[0]
        docs = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        results: list[dict] = []
        for i in range(len(ids)):
            results.append(
                {
                    "id": ids[i],
                    "text": docs[i],
                    "metadata": metadatas[i],
                    "distance": distances[i],
                }
            )
        span.set_outputs({
            "result_count": len(results),
            "results": results[:5],
        })
        return results


async def rag_answer(
    cfg: RagEvalConfig,
    embedding_client: AsyncOpenAI,
    chat_client: AsyncOpenAI,
    question: str,
    *,
    top_k: int | None = None,
    collection=None,
) -> dict:
    with start_span(
        "rag.answer",
        span_type="CHAIN",
        attributes={"chat_model": cfg.chat.model},
    ) as span:
        span.set_inputs({"question": question, "top_k": top_k if top_k is not None else cfg.top_k})
        start = time.perf_counter()
        contexts = await retrieve(cfg, embedding_client, question, top_k=top_k, collection=collection)
        retrieve_time = time.perf_counter() - start

        messages = build_messages(question, contexts)
        chat_start = time.perf_counter()
        response = await chat(
            chat_client,
            cfg.chat,
            messages,
            temperature=cfg.chat_temperature,
            max_tokens=cfg.chat_max_tokens,
            extra={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        )
        generate_time = time.perf_counter() - chat_start

        result = {
            "question": question,
            "answer": response["text"],
            "contexts": contexts,
            "usage": {
                "prompt_tokens": response.get("prompt_tokens"),
                "completion_tokens": response.get("completion_tokens"),
                "total_tokens": response.get("total_tokens"),
            },
            "timings": {
                "retrieve": retrieve_time,
                "generate": generate_time,
                "total": retrieve_time + generate_time,
            },
        }
        span.set_outputs({
            "answer_preview": result["answer"][:500],
            "context_count": len(contexts),
            "timings": result["timings"],
            "usage": result["usage"],
        })
        return result
