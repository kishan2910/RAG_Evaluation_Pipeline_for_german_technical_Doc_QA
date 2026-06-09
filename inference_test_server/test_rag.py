#!/usr/bin/env python3
"""
Standalone RAG benchmark CLI.

Three subcommands:

    ingest   Build / refresh the Chroma collection from PDFs in PDF_DIR.
    ask      Run a single RAG query end-to-end (sanity check).
    eval     Run the full Q&A benchmark and write a JSON report.

Config comes from rag/.env (see rag/.env.example). The functions in ./rag/ are
also importable directly from server.py or test_llm.py.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import aiohttp

from rag.config import load_config, RagConfig
from rag.evaluation import (
    evaluate,
    load_questions,
    save_results,
    summarize,
)
from rag.ingest import ingest_documents
from rag.pipeline import rag_query


# ---------------------------------------------------------------------------
# Pretty printing (tries rich, falls back to plain)
# ---------------------------------------------------------------------------

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
    HAS_RICH = True
except ImportError:
    console = None
    HAS_RICH = False


def info(msg: str) -> None:
    if HAS_RICH and console:
        console.print(msg)
    else:
        print(msg)


def warn(msg: str) -> None:
    if HAS_RICH and console:
        console.print(f"[yellow]{msg}[/yellow]")
    else:
        print(msg, file=sys.stderr)


# ---------------------------------------------------------------------------
# Sanity helpers
# ---------------------------------------------------------------------------

def _require_endpoint(name: str, mc) -> None:
    if not mc.is_configured():
        warn(f"{name} endpoint not configured (need {name.upper()}_BASE_URL + {name.upper()}_MODEL in rag/.env)")
        sys.exit(2)


def _print_summary(summary: dict) -> None:
    if not summary or not summary.get("n_total"):
        info("No results.")
        return

    overall = summary["overall"]
    info(f"\nQuestions evaluated: {overall['n']}  (failed: {overall['n_failed']})")
    js = overall.get("judge_score")
    if js:
        info(f"  Judge score      : mean {js['mean']:.2f}  p50 {js['p50']:.1f}  min {js['min']:.1f}  max {js['max']:.1f}  (n={js['n']})")
    sim = overall.get("semantic_similarity")
    if sim:
        info(f"  Cosine similarity: mean {sim['mean']:.3f}  p50 {sim['p50']:.3f}  min {sim['min']:.3f}  max {sim['max']:.3f}  (n={sim['n']})")
    rt = overall.get("retrieve_time")
    gt = overall.get("generate_time")
    if rt and gt:
        info(f"  Time per query   : retrieve {rt['mean']:.2f}s  generate {gt['mean']:.2f}s  (mean)")

    bd = summary.get("by_difficulty") or {}
    if not bd:
        return

    if HAS_RICH:
        table = Table(title="By difficulty", show_lines=False)
        table.add_column("Schwierigkeit", style="cyan")
        table.add_column("n", justify="right")
        table.add_column("Judge mean", justify="right")
        table.add_column("Sim mean", justify="right")
        table.add_column("Failed", justify="right")
        for diff, st in bd.items():
            j = st.get("judge_score")
            s = st.get("semantic_similarity")
            table.add_row(
                str(diff),
                str(st["n"]),
                f"{j['mean']:.2f}" if j else "—",
                f"{s['mean']:.3f}" if s else "—",
                str(st["n_failed"]),
            )
        console.print(table)  # type: ignore[union-attr]
    else:
        print("\nBy difficulty:")
        for diff, st in bd.items():
            j = st.get("judge_score")
            s = st.get("semantic_similarity")
            print(f"  {diff:<10}  n={st['n']:<3} judge={'%.2f' % j['mean'] if j else '—':<5} sim={'%.3f' % s['mean'] if s else '—':<6} failed={st['n_failed']}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

async def cmd_ingest(args: argparse.Namespace, cfg: RagConfig) -> None:
    _require_endpoint("embedding", cfg.embedding)

    pdfs = sorted(Path(cfg.pdf_dir).glob("*.pdf"))
    info(f"PDFs in {cfg.pdf_dir}: {len(pdfs)}")
    for p in pdfs:
        info(f"  · {p.name}")

    def on_progress(ev: dict) -> None:
        t = ev["type"]
        if t == "pdf":
            info(f"  extracted {ev['chunks']} chunks from {ev['path']}")
        elif t == "plan":
            info(f"  {ev['new']} new chunks to embed ({ev['existing']} already in store)")
        elif t == "batch":
            info(f"  embedded batch {ev['batch']} ({ev['added_so_far']}/{ev['total_new']})")
        elif t == "done":
            info(f"  added {ev['added']} chunks (total in plan: {ev['total']})")

    timeout = aiohttp.ClientTimeout(total=cfg.request_timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        added = await ingest_documents(cfg, session, reset=args.reset, progress=on_progress)
    info(f"\nDone. {added} chunks added to '{cfg.collection}' at {cfg.chroma_dir}.")


async def cmd_ask(args: argparse.Namespace, cfg: RagConfig) -> None:
    _require_endpoint("embedding", cfg.embedding)
    _require_endpoint("chat", cfg.chat)

    timeout = aiohttp.ClientTimeout(total=cfg.request_timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        out = await rag_query(cfg, session, args.question, top_k=args.top_k)

    info(f"\nQ: {out['question']}\n")
    info(f"A: {out['answer']}\n")
    info(f"(retrieved {len(out['contexts'])} chunks · "
         f"retrieve {out['timings']['retrieve']:.2f}s · "
         f"generate {out['timings']['generate']:.2f}s)")
    if args.show_contexts:
        info("\nContexts:")
        for i, c in enumerate(out["contexts"], 1):
            m = c.get("metadata") or {}
            preview = (c["text"][:200] + "…") if len(c["text"]) > 200 else c["text"]
            info(f"  [{i}] {m.get('source')} p{m.get('page')}  d={c.get('distance'):.4f}\n      {preview}")


async def cmd_eval(args: argparse.Namespace, cfg: RagConfig) -> None:
    _require_endpoint("embedding", cfg.embedding)
    _require_endpoint("chat", cfg.chat)
    if not args.no_judge:
        _require_endpoint("judge", cfg.judge)

    questions = load_questions(cfg.eval_csv)
    if args.limit:
        questions = questions[: args.limit]
    info(f"Evaluating {len(questions)} questions from {cfg.eval_csv}")
    info(f"  chat:  {cfg.chat.model} @ {cfg.chat.base_url}")
    info(f"  embed: {cfg.embedding.model} @ {cfg.embedding.base_url}")
    if not args.no_judge:
        info(f"  judge: {cfg.judge.model} @ {cfg.judge.base_url}")

    done = 0

    def on_progress(ev: dict) -> None:
        nonlocal done
        if ev.get("type") == "row":
            done += 1
            mark = "✗" if ev.get("error") else "✓"
            judge = ev.get("judge_score")
            sim   = ev.get("similarity")
            extras = []
            if judge is not None:
                extras.append(f"judge={judge:.1f}")
            if sim is not None:
                extras.append(f"sim={sim:.3f}")
            extra_str = ("  " + " ".join(extras)) if extras else ""
            info(f"  [{done:>4}/{len(questions)}] {mark} id={ev['id']}{extra_str}")

    timeout = aiohttp.ClientTimeout(total=cfg.request_timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = await evaluate(
            cfg, session, questions,
            do_judge=not args.no_judge,
            do_similarity=not args.no_similarity,
            concurrency=args.concurrency,
            progress=on_progress,
        )

    summary = summarize(results)
    _print_summary(summary)

    if args.output:
        save_results(results, summary, args.output)
        info(f"\nWrote {len(results)} results + summary to {args.output}")
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        info(f"Wrote summary to {args.summary_json}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RAG benchmark for OpenAI-compatible endpoints.")
    p.add_argument("--env-file", default="rag/.env", help="path to .env (default: rag/.env)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="(re)build the Chroma collection from PDFs")
    pi.add_argument("--reset", action="store_true", help="wipe the collection before adding")

    pa = sub.add_parser("ask", help="run one RAG query")
    pa.add_argument("question")
    pa.add_argument("--top-k", type=int, help="override TOP_K for this query")
    pa.add_argument("--show-contexts", action="store_true")

    pe = sub.add_parser("eval", help="run the full Q&A benchmark")
    pe.add_argument("--limit", type=int, help="only evaluate the first N questions")
    pe.add_argument("--concurrency", type=int, default=1, help="parallel requests (default: 1)")
    pe.add_argument("--no-judge", action="store_true", help="skip LLM-as-judge scoring")
    pe.add_argument("--no-similarity", action="store_true", help="skip cosine similarity scoring")
    pe.add_argument("--output", default="rag_results.json", help="JSON output path (results + summary)")
    pe.add_argument("--summary-json", help="optional separate JSON file for the summary block only")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg  = load_config(args.env_file)
    coro = {"ingest": cmd_ingest, "ask": cmd_ask, "eval": cmd_eval}[args.cmd](args, cfg)
    asyncio.run(coro)


if __name__ == "__main__":
    main()
