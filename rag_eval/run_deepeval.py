from __future__ import annotations

import argparse
import asyncio
import re
import time

import mlflow

from deepeval_runner import evaluate_dataset, save_results, summarize
from mlflow_tracking import log_artifact_if_exists, log_metrics, start_run
from openai_compat import make_async_client, make_sync_client
from settings import DEFAULT_ENV_FILE, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DeepEval over all questions in the benchmark CSV.")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE), help="Path to .env file")
    parser.add_argument("--limit", type=int, help="Evaluate only first N rows")
    parser.add_argument("--top-k", type=int, help="Override retrieval top_k")
    parser.add_argument("--output", default="deepeval_results.json", help="Output JSON path")
    return parser


async def _run(args: argparse.Namespace) -> None:
    cfg = load_config(args.env_file)
    run = start_run(
        cfg,
        "eval",
        params={"limit": args.limit, "top_k_override": args.top_k, "output": args.output},
    )

    def on_row(row, step: int) -> None:
        metrics: dict = {
            "row_retrieval_seconds": row.retrieval_time,
            "row_generation_seconds": row.generation_time,
            "row_failed": 1 if row.error else 0,
        }
        for name, score in row.scores.items():
            metrics["row_score_" + re.sub(r"[^a-z0-9_\-./]", "", name.lower().replace(" ", "_"))] = score
        log_metrics(metrics, step=step)

    started = time.perf_counter()
    embedding_client = make_async_client(cfg.embedding, timeout=cfg.request_timeout)
    chat_client = make_async_client(cfg.chat, timeout=cfg.request_timeout)
    judge_client = make_sync_client(cfg.judge, timeout=cfg.request_timeout)
    judge_async_client = make_async_client(cfg.judge, timeout=cfg.request_timeout)
    status = "FINISHED"
    try:
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
        save_results(rows, summary, args.output)
        log_artifact_if_exists(args.output)

        elapsed = time.perf_counter() - started
        summary_metrics: dict = {
            "eval_seconds": elapsed,
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
        print(f"  output: {args.output}")
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
