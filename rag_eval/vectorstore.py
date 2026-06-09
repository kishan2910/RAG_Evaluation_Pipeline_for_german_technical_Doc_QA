from __future__ import annotations

from pathlib import Path
from typing import Callable

from openai import AsyncOpenAI

from mlflow_tracking import start_span
from openai_compat import embed
from settings import RagEvalConfig

ProgressCb = Callable[[dict], None] | None


def extract_pdf(path: str | Path) -> list[dict]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[dict] = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    return pages


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            last_space = text.rfind(" ", start + chunk_size // 2, end)
            if last_space != -1:
                end = last_space
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = end - overlap
    return chunks


def build_chunks(pdf_path: str | Path, chunk_size: int, overlap: int) -> list[dict]:
    path = Path(pdf_path)
    pages = extract_pdf(path)
    out: list[dict] = []
    for page_info in pages:
        text_chunks = chunk_text(page_info["text"], chunk_size, overlap)
        for idx, chunk in enumerate(text_chunks):
            out.append(
                {
                    "id": f"{path.name}::p{page_info['page']}::c{idx}",
                    "text": chunk,
                    "metadata": {
                        "source": path.name,
                        "page": page_info["page"],
                        "chunk_index": idx,
                    },
                }
            )
    return out


def open_collection(chroma_dir: str | Path, collection_name: str, reset: bool = False):
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    return client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})


async def ingest_documents(
    cfg: RagEvalConfig,
    embedding_client: AsyncOpenAI,
    *,
    reset: bool = False,
    progress: ProgressCb = None,
) -> dict:
    with start_span(
        "vectorstore.ingest_documents",
        span_type="CHAIN",
        attributes={
            "pdf_dir": str(cfg.pdf_dir),
            "chroma_dir": str(cfg.chroma_dir),
            "collection": cfg.collection,
            "reset": reset,
        },
    ) as span:
        pdf_paths = sorted(Path(cfg.pdf_dir).glob("*.pdf"))
        if not pdf_paths:
            raise FileNotFoundError(f"no PDFs found in {cfg.pdf_dir}")

        all_chunks: list[dict] = []
        for pdf_path in pdf_paths:
            chunks = build_chunks(pdf_path, cfg.chunk_size, cfg.chunk_overlap)
            all_chunks.extend(chunks)
            if progress:
                progress({"type": "pdf", "path": pdf_path.name, "chunks": len(chunks)})

        collection = open_collection(cfg.chroma_dir, cfg.collection, reset=reset)
        all_ids = [c["id"] for c in all_chunks]
        existing = set(collection.get(ids=all_ids).get("ids", []))
        new_chunks = [c for c in all_chunks if c["id"] not in existing]

        if progress:
            progress(
                {
                    "type": "plan",
                    "total": len(all_chunks),
                    "new": len(new_chunks),
                    "existing": len(existing),
                }
            )

        if not new_chunks:
            if progress:
                progress({"type": "done", "added": 0, "total": len(all_chunks)})
            stats = {
                "pdf_count": len(pdf_paths),
                "total_chunks": len(all_chunks),
                "new_chunks": 0,
                "existing_chunks": len(existing),
                "added_chunks": 0,
                "batches": 0,
            }
            span.set_outputs(stats)
            return stats

        batch_size = cfg.embed_batch_size
        added = 0
        batch_count = 0
        for start in range(0, len(new_chunks), batch_size):
            batch = new_chunks[start : start + batch_size]
            texts = [chunk["text"] for chunk in batch]
            with start_span(
                "vectorstore.embed_batch",
                span_type="EMBEDDING",
                attributes={
                    "batch_index": start // batch_size + 1,
                    "batch_size": len(batch),
                },
            ) as batch_span:
                batch_span.set_inputs({"chunk_ids": [chunk["id"] for chunk in batch]})
                embeddings = await embed(embedding_client, cfg.embedding, texts)
                collection.add(
                    ids=[chunk["id"] for chunk in batch],
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=[chunk["metadata"] for chunk in batch],
                )
                batch_span.set_outputs({"embedded_chunks": len(batch)})
            added += len(batch)
            batch_count += 1
            if progress:
                progress(
                    {
                        "type": "batch",
                        "batch": start // batch_size + 1,
                        "size": len(batch),
                        "added_so_far": added,
                        "total_new": len(new_chunks),
                    }
                )

        if progress:
            progress({"type": "done", "added": added, "total": len(all_chunks)})
        stats = {
            "pdf_count": len(pdf_paths),
            "total_chunks": len(all_chunks),
            "new_chunks": len(new_chunks),
            "existing_chunks": len(existing),
            "added_chunks": added,
            "batches": batch_count,
        }
        span.set_outputs(stats)
        return stats
