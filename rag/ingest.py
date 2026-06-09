"""
Document ingestion: PDF → chunks → embeddings → Chroma.

Functions are independently usable:

    pages = extract_pdf(path)                       # [{page, text}, ...]
    chunks = chunk_text(text, 900, 150)             # [str, ...]
    docs = build_chunks(path, 900, 150)             # [{id, text, metadata}, ...]
    collection = open_collection(dir, name)          # chromadb collection
    added = await ingest_documents(cfg, session)    # full pipeline, idempotent
"""

from pathlib import Path
from typing import Callable, Optional

import aiohttp

from .client import embed
from .config import RagConfig


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_pdf(path: str | Path) -> list[dict]:
    """Extract per-page text from a PDF. Returns [{page, text}, ...] (1-indexed)."""
    from pypdf import PdfReader  # local import: heavy dep
    reader = PdfReader(str(path))
    pages: list[dict] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    return pages


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Sliding window over text, breaking on whitespace where possible.

    Pure function — given the same args, deterministic output.
    """
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
            # Prefer breaking on whitespace within the second half of the window.
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
    """Full per-PDF pipeline: extract pages → chunk each → attach metadata.

    Returns [{id, text, metadata}] suitable for chromadb.add().
    Chunk ids are stable across runs (idempotent re-ingest).
    """
    path = Path(pdf_path)
    pages = extract_pdf(path)
    out: list[dict] = []
    for page_info in pages:
        for i, chunk in enumerate(chunk_text(page_info["text"], chunk_size, overlap)):
            out.append({
                "id": f"{path.name}::p{page_info['page']}::c{i}",
                "text": chunk,
                "metadata": {
                    "source": path.name,
                    "page": page_info["page"],
                    "chunk_index": i,
                },
            })
    return out


# ---------------------------------------------------------------------------
# Chroma collection helpers
# ---------------------------------------------------------------------------

def open_collection(chroma_dir: str | Path, collection_name: str, *, reset: bool = False):
    """Open (or create) a persistent Chroma collection using cosine distance."""
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
    return client.get_or_create_collection(
        collection_name,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Full ingest pipeline
# ---------------------------------------------------------------------------

ProgressCb = Optional[Callable[[dict], None]]


async def ingest_documents(
    cfg: RagConfig,
    session: aiohttp.ClientSession,
    *,
    reset: bool = False,
    progress: ProgressCb = None,
) -> int:
    """Walk PDF_DIR, build chunks, embed in batches, upsert into Chroma.

    Idempotent: skips chunks whose id is already in the collection (unless
    reset=True). Returns the number of newly-added chunks.
    """
    pdf_paths = sorted(Path(cfg.pdf_dir).glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"no PDFs found in {cfg.pdf_dir}")

    # 1. Build chunks for every PDF (pure, no I/O to the embedder yet).
    all_chunks: list[dict] = []
    for path in pdf_paths:
        chunks = build_chunks(path, cfg.chunk_size, cfg.chunk_overlap)
        all_chunks.extend(chunks)
        if progress:
            progress({"type": "pdf", "path": path.name, "chunks": len(chunks)})

    if not all_chunks:
        if progress:
            progress({"type": "done", "added": 0, "total": 0})
        return 0

    # 2. Open Chroma (optionally reset) and figure out which ids are new.
    collection = open_collection(cfg.chroma_dir, cfg.collection, reset=reset)
    all_ids = [c["id"] for c in all_chunks]
    existing = set(collection.get(ids=all_ids).get("ids", []))
    new_chunks = [c for c in all_chunks if c["id"] not in existing]
    if progress:
        progress({"type": "plan", "total": len(all_chunks), "new": len(new_chunks),
                  "existing": len(existing)})
    if not new_chunks:
        if progress:
            progress({"type": "done", "added": 0, "total": len(all_chunks)})
        return 0

    # 3. Embed in batches and upsert.
    batch_size = cfg.embed_batch_size
    added = 0
    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        embeddings = await embed(
            session, cfg.embedding, texts,
            timeout=cfg.request_timeout,
        )
        collection.add(
            ids=[c["id"] for c in batch],
            documents=texts,
            embeddings=embeddings,
            metadatas=[c["metadata"] for c in batch],
        )
        added += len(batch)
        if progress:
            progress({
                "type": "batch",
                "batch": i // batch_size + 1,
                "size": len(batch),
                "added_so_far": added,
                "total_new": len(new_chunks),
            })

    if progress:
        progress({"type": "done", "added": added, "total": len(all_chunks)})
    return added
