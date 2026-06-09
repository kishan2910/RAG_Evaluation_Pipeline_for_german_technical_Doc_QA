from __future__ import annotations

import argparse
import asyncio
import itertools
import re
import time
from pathlib import Path

import mlflow

from deepeval_runner import evaluate_dataset, save_results, summarize
from mlflow_tracking import log_artifact_if_exists, log_metrics, start_run, start_span
from openai_compat import make_async_client, make_sync_client
from rag_query import rag_answer
from settings import (
    DEFAULT_ENV_FILE,
    ModelConfig,
    apply_overrides,
    load_chat_model_configs,
    load_embedding_model_configs,
    load_judge_model_configs,
    load_config,
)
from vectorstore import ingest_documents


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="End-to-end RAG + DeepEval pipeline")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE), help="Path to .env file")

    sub = parser.add_subparsers(dest="cmd", required=True)

    ingest = sub.add_parser("ingest", help="Chunk PDFs, create embeddings, and build Chroma")
    ingest.add_argument("--reset", action="store_true", help="Reset collection before ingest")

    ask = sub.add_parser("ask", help="Retrieve context and chat with the documents")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int)
    ask.add_argument("--show-contexts", action="store_true")

    for name, help_text in [
        ("eval", "Evaluate all benchmark questions with DeepEval"),
        ("run-all", "Ingest first, then run full DeepEval benchmark"),
    ]:
        p = sub.add_parser(name, help=help_text)
        if name == "run-all":
            p.add_argument("--reset", action="store_true")
        p.add_argument("--limit", type=int, help="Evaluate only first N questions")
        p.add_argument("--top-k", type=int, help="Override retrieval top_k")
        p.add_argument("--output", default="deepeval_results.json", help="Output JSON path")
        p.add_argument(
            "--embedding-model",
            help="Pin embedding model (single run). If omitted, runs all EMBEDDING_MODEL* from .env",
        )
        p.add_argument(
            "--embedding-base-url",
            help="Base URL for --embedding-model (defaults to EMBEDDING_BASE_URL from .env)",
        )
        p.add_argument(
            "--chat-model",
            help="Pin chat model (single run). If omitted, runs all CHAT_MODEL* from .env",
        )
        p.add_argument(
            "--chat-base-url",
            help="Base URL for --chat-model (defaults to CHAT_BASE_URL from .env)",
        )
        p.add_argument(
            "--judge-model",
            help="Pin judge model (single run). If omitted, runs all JUDGE_MODEL* from .env",
        )
        p.add_argument(
            "--judge-base-url",
            help="Base URL for --judge-model (defaults to JUDGE_BASE_URL from .env)",
        )
        p.add_argument("--run-name", help="Custom MLflow run name (auto-generated if omitted)")

    return parser


def _output_path_for_run(base_output: str, run_name: str, total: int) -> str:
    """Single combo → use base_output as-is. Multiple combos → write into a sub-directory."""
    if total == 1:
        return base_output
    base = Path(base_output)
    out_dir = base.parent / base.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"{run_name}.json")


def _make_run_name(chat: str, embed: str, judge: str) -> str:
    def s(n: str) -> str:
        return re.sub(r"[^a-zA-Z0-9._-]", "-", n)[:28]
    return f"eval__emb-{s(embed)}__chat-{s(chat)}__judge-{s(judge)}"


def _print_ingest_progress(event: dict) -> None:
    kind = event.get("type")
    if kind == "pdf":
        print(f"[pdf] {event['path']}: {event['chunks']} chunks")
    elif kind == "plan":
        print(f"[plan] total={event['total']} new={event['new']} existing={event['existing']}")
    elif kind == "batch":
        print(f"[batch] {event['batch']} added={event['added_so_far']}/{event['total_new']}")
    elif kind == "done":
        print(f"[done] added={event['added']} total={event['total']}")


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


async def _run_eval(
    cfg,
    args: argparse.Namespace,
    embedding_client,
    chat_client,
    judge_client,
    judge_async_client,
    *,
    output_path: str,
) -> None:
    def on_row(row, step: int) -> None:
        if row.error:
            print(f"  [{step}] row {row.id} — ERROR: {row.error}")
        else:
            scores_str = "  ".join(
                f"{name[:12]}={score:.2f}" for name, score in row.scores.items() if score is not None
            )
            print(f"  [{step}] row {row.id} done — {scores_str}")

        metrics: dict = {
            "row_retrieval_seconds": row.retrieval_time,
            "row_generation_seconds": row.generation_time,
            "row_failed": 1 if row.error else 0,
        }
        for name, score in row.scores.items():
            metrics["row_score_" + re.sub(r"[^a-z0-9_\-./]", "", name.lower().replace(" ", "_"))] = score
        log_metrics(metrics, step=step)

    rows = await evaluate_dataset(
        cfg,
        embedding_client,
        chat_client,
        judge_client,
        judge_async_client,
        limit=args.limit,
        top_k=args.top_k,
        progress=on_row,
    )
    summary = summarize(rows)
    model_config = {
        "embedding_model": cfg.embedding.model,
        "embedding_base_url": cfg.embedding.base_url,
        "chat_model": cfg.chat.model,
        "chat_base_url": cfg.chat.base_url,
        "judge_model": cfg.judge.model,
        "judge_base_url": cfg.judge.base_url,
    }
    save_results(rows, summary, output_path, config=model_config)
    log_artifact_if_exists(output_path)
    summary_metrics: dict = {
        "total_questions": summary["total"],
        "failed_questions": summary["failed"],
    }
    for name, avg in summary["avg_scores"].items():
        summary_metrics["avg_score_" + re.sub(r"[^a-z0-9_\-./]", "", name.lower().replace(" ", "_"))] = avg
    log_metrics(summary_metrics)
    print("DeepEval finished")
    print(f"  total: {summary['total']}")
    print(f"  failed: {summary['failed']}")
    for name, avg in summary["avg_scores"].items():
        print(f"  avg_{name}: {avg:.3f}" if avg is not None else f"  avg_{name}: None")
    print(f"  output: {output_path}")


def _resolve_embedding_models(args: argparse.Namespace, base_cfg) -> list[ModelConfig]:
    """Return embedding model(s) to evaluate — one from CLI or all from .env."""
    if getattr(args, "embedding_model", None):
        base_url = getattr(args, "embedding_base_url", None) or base_cfg.embedding.base_url
        return [ModelConfig(base_url=base_url, api_key=base_cfg.embedding.api_key, model=args.embedding_model)]
    return load_embedding_model_configs(args.env_file)


def _resolve_chat_models(args: argparse.Namespace, base_cfg) -> list[ModelConfig]:
    """Return chat model(s) to evaluate — one from CLI or all from .env."""
    if getattr(args, "chat_model", None):
        base_url = getattr(args, "chat_base_url", None) or base_cfg.chat.base_url
        return [ModelConfig(base_url=base_url, api_key=base_cfg.chat.api_key, model=args.chat_model)]
    return load_chat_model_configs(args.env_file)


def _resolve_judge_models(args: argparse.Namespace, base_cfg) -> list[ModelConfig]:
    """Return judge model(s) to evaluate — one from CLI or all from .env."""
    if getattr(args, "judge_model", None):
        base_url = getattr(args, "judge_base_url", None) or base_cfg.judge.base_url
        return [ModelConfig(base_url=base_url, api_key=base_cfg.judge.api_key, model=args.judge_model)]
    return load_judge_model_configs(args.env_file)


async def _run_eval_for_model(
    args: argparse.Namespace,
    cfg,
    ingest_stats: dict | None = None,
    *,
    total_combinations: int = 1,
) -> None:
    """Run one MLflow eval run for the given cfg (already has the right models set)."""
    run_name = getattr(args, "run_name", None) or _make_run_name(
        cfg.chat.model, cfg.embedding.model, cfg.judge.model
    )
    output_path = _output_path_for_run(args.output, run_name, total_combinations)
    print(f"\n=== Starting run: {run_name} ===")
    print(f"    embed={cfg.embedding.model}  chat={cfg.chat.model}  judge={cfg.judge.model}")
    print(f"    output → {output_path}")

    start_run(cfg, run_name, params=vars(args))
    started = time.perf_counter()
    status = "FINISHED"
    try:
        embedding_client = make_async_client(cfg.embedding, timeout=cfg.request_timeout)
        chat_client = make_async_client(cfg.chat, timeout=cfg.request_timeout)
        judge_client = make_sync_client(cfg.judge, timeout=cfg.request_timeout)
        judge_async_client = make_async_client(cfg.judge, timeout=cfg.request_timeout)

        if ingest_stats:
            log_metrics({
                "pdf_count": ingest_stats["pdf_count"],
                "total_chunks": ingest_stats["total_chunks"],
                "new_chunks": ingest_stats["new_chunks"],
                "existing_chunks": ingest_stats["existing_chunks"],
                "added_chunks": ingest_stats["added_chunks"],
                "batches": ingest_stats["batches"],
            })

        # No wrapping span here — each row in evaluate_dataset creates its own
        # root-level trace. A parent span here would make all rows children of
        # one trace instead of appearing as 10 separate traces in MLflow.
        await _run_eval(cfg, args, embedding_client, chat_client, judge_client, judge_async_client, output_path=output_path)
    except Exception:
        status = "FAILED"
        raise
    finally:
        log_metrics({"run_seconds": time.perf_counter() - started})
        mlflow.end_run(status=status)


async def _run(args: argparse.Namespace) -> None:
    base_cfg = load_config(args.env_file)

    # --- ingest and ask don't need multi-model logic ---
    if args.cmd == "ingest":
        start_run(base_cfg, "ingest", params=vars(args))
        started = time.perf_counter()
        status = "FINISHED"
        try:
            embedding_client = make_async_client(base_cfg.embedding, timeout=base_cfg.request_timeout)
            with start_span("pipeline.ingest", span_type="CHAIN") as span:
                stats = await ingest_documents(base_cfg, embedding_client, reset=args.reset, progress=_print_ingest_progress)
                log_metrics({
                    "pdf_count": stats["pdf_count"],
                    "total_chunks": stats["total_chunks"],
                    "new_chunks": stats["new_chunks"],
                    "existing_chunks": stats["existing_chunks"],
                    "added_chunks": stats["added_chunks"],
                    "batches": stats["batches"],
                })
                span.set_outputs(stats)
        except Exception:
            status = "FAILED"
            raise
        finally:
            log_metrics({"run_seconds": time.perf_counter() - started})
            mlflow.end_run(status=status)
        return

    if args.cmd == "ask":
        start_run(base_cfg, "ask", params=vars(args))
        started = time.perf_counter()
        status = "FINISHED"
        try:
            embedding_client = make_async_client(base_cfg.embedding, timeout=base_cfg.request_timeout)
            chat_client = make_async_client(base_cfg.chat, timeout=base_cfg.request_timeout)
            with start_span("pipeline.ask", span_type="CHAIN") as span:
                result = await rag_answer(base_cfg, embedding_client, chat_client, args.question, top_k=args.top_k)
                log_metrics({
                    "retrieve_seconds": result["timings"]["retrieve"],
                    "generate_seconds": result["timings"]["generate"],
                    "total_seconds": result["timings"]["total"],
                    "retrieved_contexts": len(result["contexts"]),
                    "prompt_tokens": result["usage"].get("prompt_tokens"),
                    "completion_tokens": result["usage"].get("completion_tokens"),
                    "total_tokens": result["usage"].get("total_tokens"),
                })
                span.set_outputs({"answer_preview": result["answer"][:500], "timings": result["timings"]})
                _print_answer(result, args.show_contexts)
        except Exception:
            status = "FAILED"
            raise
        finally:
            log_metrics({"run_seconds": time.perf_counter() - started})
            mlflow.end_run(status=status)
        return

    # --- eval and run-all: iterate over all combinations ---
    embedding_models = _resolve_embedding_models(args, base_cfg)
    chat_models = _resolve_chat_models(args, base_cfg)
    judge_models = _resolve_judge_models(args, base_cfg)

    if not embedding_models:
        raise RuntimeError("No embedding models found. Check EMBEDDING_MODEL in .env or pass --embedding-model.")
    if not chat_models:
        raise RuntimeError("No chat models found. Check CHAT_MODEL in .env or pass --chat-model.")
    if not judge_models:
        raise RuntimeError("No judge models found. Check JUDGE_MODEL in .env or pass --judge-model.")

    combinations = list(itertools.product(embedding_models, chat_models, judge_models))
    print(
        f"Running {len(combinations)} combination(s): "
        f"{len(embedding_models)} embed × {len(chat_models)} chat × {len(judge_models)} judge"
    )

    # For run-all: ingest ONCE per unique embedding model, not once per combination
    ingest_stats_by_model: dict[str, dict] = {}
    if args.cmd == "run-all":
        seen_embedding_models: set[str] = set()
        for emb_cfg in embedding_models:
            if emb_cfg.model in seen_embedding_models:
                continue
            seen_embedding_models.add(emb_cfg.model)
            cfg_for_ingest = apply_overrides(base_cfg, embedding=emb_cfg)
            embedding_client = make_async_client(cfg_for_ingest.embedding, timeout=base_cfg.request_timeout)
            print(f"\n=== Ingesting for embedding model: {emb_cfg.model} (collection: {cfg_for_ingest.collection}) ===")
            stats = await ingest_documents(cfg_for_ingest, embedding_client, reset=args.reset, progress=_print_ingest_progress)
            ingest_stats_by_model[emb_cfg.model] = stats

    for emb_cfg, chat_cfg, judge_cfg in combinations:
        cfg = apply_overrides(base_cfg, embedding=emb_cfg, chat=chat_cfg, judge=judge_cfg)
        ingest_stats = ingest_stats_by_model.get(emb_cfg.model)
        await _run_eval_for_model(args, cfg, ingest_stats=ingest_stats, total_combinations=len(combinations))


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
