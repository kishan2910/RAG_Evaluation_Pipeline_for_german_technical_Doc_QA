from __future__ import annotations

import argparse
import asyncio

import mlflow

from mlflow_tracking import log_metrics, start_run
from openai_compat import make_async_client
from rag_query import rag_answer
from settings import DEFAULT_ENV_FILE, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask questions against the RAG vectorstore.")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE), help="Path to .env file")
    parser.add_argument("--question", "-q", help="Single question to ask")
    parser.add_argument("--top-k", type=int, help="Override retrieval top_k")
    parser.add_argument("--show-contexts", action="store_true", help="Print retrieved chunks")
    parser.add_argument("--interactive", action="store_true", help="Start interactive chat loop")
    return parser


def _print_answer(result: dict, show_contexts: bool) -> None:
    print(f"\nQ: {result['question']}\n")
    print(f"A: {result['answer']}\n")
    print(
        "timing: "
        f"retrieve={result['timings']['retrieve']:.2f}s "
        f"generate={result['timings']['generate']:.2f}s"
    )

    if show_contexts:
        print("\nRetrieved contexts:")
        for idx, ctx in enumerate(result["contexts"], start=1):
            meta = ctx.get("metadata") or {}
            text = ctx.get("text", "")
            preview = text if len(text) <= 220 else text[:220] + "..."
            print(
                f"[{idx}] {meta.get('source', '?')} p{meta.get('page', '?')} "
                f"distance={ctx.get('distance')}"
            )
            print(f"    {preview}")


async def _ask_once(args: argparse.Namespace) -> None:
    cfg = load_config(args.env_file)
    run = start_run(cfg, "ask", params={"top_k_override": args.top_k})
    embedding_client = make_async_client(cfg.embedding, timeout=cfg.request_timeout)
    chat_client = make_async_client(cfg.chat, timeout=cfg.request_timeout)
    status = "FINISHED"
    try:
        result = await rag_answer(cfg, embedding_client, chat_client, args.question, top_k=args.top_k)
        log_metrics(
            {
                "retrieve_seconds": result["timings"]["retrieve"],
                "generate_seconds": result["timings"]["generate"],
                "total_seconds": result["timings"]["total"],
                "retrieved_contexts": len(result["contexts"]),
                "prompt_tokens": result["usage"].get("prompt_tokens"),
                "completion_tokens": result["usage"].get("completion_tokens"),
                "total_tokens": result["usage"].get("total_tokens"),
            }
        )
    except Exception:
        status = "FAILED"
        raise
    finally:
        mlflow.end_run(status=status)
    _print_answer(result, args.show_contexts)


async def _ask_interactive(args: argparse.Namespace) -> None:
    cfg = load_config(args.env_file)
    run = start_run(cfg, "chat_interactive", params={"top_k_override": args.top_k})
    embedding_client = make_async_client(cfg.embedding, timeout=cfg.request_timeout)
    chat_client = make_async_client(cfg.chat, timeout=cfg.request_timeout)
    turns = 0
    status = "FINISHED"
    try:
        while True:
            question = input("\nquestion (or 'exit'): ").strip()
            if not question or question.lower() in {"exit", "quit"}:
                break
            turns += 1
            result = await rag_answer(cfg, embedding_client, chat_client, question, top_k=args.top_k)
            log_metrics(
                {
                    "retrieve_seconds": result["timings"]["retrieve"],
                    "generate_seconds": result["timings"]["generate"],
                    "total_seconds": result["timings"]["total"],
                    "retrieved_contexts": len(result["contexts"]),
                    "prompt_tokens": result["usage"].get("prompt_tokens"),
                    "completion_tokens": result["usage"].get("completion_tokens"),
                    "total_tokens": result["usage"].get("total_tokens"),
                },
                step=turns,
            )
            _print_answer(result, args.show_contexts)
    except Exception:
        status = "FAILED"
        raise
    finally:
        log_metrics({"turns": turns})
        mlflow.end_run(status=status)


def main() -> None:
    args = build_parser().parse_args()
    if args.interactive:
        asyncio.run(_ask_interactive(args))
        return
    if not args.question:
        raise SystemExit("Provide --question or use --interactive")
    asyncio.run(_ask_once(args))


if __name__ == "__main__":
    main()
