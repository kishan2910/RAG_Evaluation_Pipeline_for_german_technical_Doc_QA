from __future__ import annotations

import argparse
import asyncio
import time

import mlflow

from mlflow_tracking import log_metrics, start_run
from openai_compat import make_async_client
from settings import DEFAULT_ENV_FILE, load_config
from vectorstore import ingest_documents


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build/refresh Chroma vectorstore from PDFs.")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE), help="Path to .env file")
    parser.add_argument("--reset", action="store_true", help="Reset collection before ingest")
    return parser


async def _run(args: argparse.Namespace) -> None:
    cfg = load_config(args.env_file)
    run = start_run(cfg, "ingest", params={"reset": args.reset})

    def on_progress(event: dict) -> None:
        kind = event.get("type")
        if kind == "pdf":
            print(f"[pdf] {event['path']}: {event['chunks']} chunks")
        elif kind == "plan":
            print(f"[plan] total={event['total']} new={event['new']} existing={event['existing']}")
        elif kind == "batch":
            print(f"[batch] {event['batch']} added={event['added_so_far']}/{event['total_new']}")
        elif kind == "done":
            print(f"[done] added={event['added']} total={event['total']}")

    started = time.perf_counter()
    embedding_client = make_async_client(cfg.embedding, timeout=cfg.request_timeout)
    status = "FINISHED"
    try:
        stats = await ingest_documents(cfg, embedding_client, reset=args.reset, progress=on_progress)
        elapsed = time.perf_counter() - started
        log_metrics(
            {
                "ingest_seconds": elapsed,
                "pdf_count": stats["pdf_count"],
                "total_chunks": stats["total_chunks"],
                "new_chunks": stats["new_chunks"],
                "existing_chunks": stats["existing_chunks"],
                "added_chunks": stats["added_chunks"],
                "batches": stats["batches"],
            }
        )
        print(
            f"Vectorstore ready at {cfg.chroma_dir} "
            f"(collection={cfg.collection}, added={stats['added_chunks']})"
        )
    except Exception:
        status = "FAILED"
        raise
    finally:
        mlflow.end_run(status=status)


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
