"""
Evaluation runner: takes the Q&A CSV and scores RAG output.

Metrics:
  - semantic_similarity: cosine of (predicted, expected) sentence embeddings
  - judge_score: 0–5 from an LLM-as-judge (German rubric)
  - timing: retrieve / generate per question

Public API:

    questions = load_questions(cfg.eval_csv)
    results = await evaluate(cfg, session, questions, ...)
    summary = summarize(results)
"""

import asyncio
import csv
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import aiohttp

from .client import chat, embed
from .config import RagConfig
from .pipeline import rag_query


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    id: str
    question: str
    expected: str
    predicted: str
    semantic_similarity: Optional[float] = None
    judge_score: Optional[float] = None
    judge_reason: Optional[str] = None
    difficulty: Optional[str] = None
    expected_section: Optional[str] = None
    typ: Optional[str] = None                 # "Im Dokument" etc.
    bewertungstyp: Optional[str] = None       # "Semantisch", "Exakt", ...
    status: Optional[str] = None              # "Konform" / "Nicht konform"
    retrieved_chunks: list[dict] = field(default_factory=list)
    retrieve_time: float = 0.0
    generate_time: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = (
    "Du bist ein strikter Bewertungsrichter. Vergleiche die vorhergesagte "
    "Antwort mit der erwarteten Antwort und vergib eine Punktzahl von 0 bis 5:\n"
    "0 = komplett falsch oder keine Antwort\n"
    "1 = größtenteils falsch, aber thematisch verwandt\n"
    "2 = teilweise korrekt, Schlüsselinformation fehlt\n"
    "3 = größtenteils korrekt, kleinere Fehler\n"
    "4 = korrekt, kleinere Unterschiede in der Formulierung\n"
    "5 = im Wesentlichen identisch in Bedeutung und Inhalt\n\n"
    'Antworte ausschließlich mit JSON, z.B. {"score": 4, "reason": "..."}'
)

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_judge_output(raw: str) -> tuple[float, str]:
    """Lenient JSON extraction from the judge's chat output."""
    text = raw.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)
    # Try direct parse first
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fall back to finding the first {...} blob
        m = _JSON_RE.search(text)
        if not m:
            return float("nan"), f"unparseable: {raw[:120]}"
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            return float("nan"), f"parse error: {e}"
    try:
        score = float(data.get("score"))
    except (TypeError, ValueError):
        return float("nan"), f"missing/invalid score in: {raw[:120]}"
    reason = str(data.get("reason", ""))
    return score, reason


async def judge_answer(
    session: aiohttp.ClientSession,
    cfg: RagConfig,
    question: str,
    expected: str,
    predicted: str,
) -> tuple[float, str]:
    """Ask the judge LLM to score a (question, expected, predicted) triple."""
    user = (
        f"Frage:\n{question}\n\n"
        f"Erwartete Antwort:\n{expected}\n\n"
        f"Vorhergesagte Antwort:\n{predicted}"
    )
    out = await chat(
        session, cfg.judge,
        [{"role": "system", "content": JUDGE_SYSTEM},
         {"role": "user",   "content": user}],
        temperature=0.0,
        max_tokens=256,
        timeout=cfg.request_timeout,
    )
    return _parse_judge_output(out["text"])


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return float("nan")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na  += x * x
        nb  += y * y
    if na <= 0 or nb <= 0:
        return float("nan")
    return dot / (math.sqrt(na) * math.sqrt(nb))


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_questions(path: str | Path) -> list[dict]:
    """Load the Q&A CSV. Handles the UTF-8 BOM the source file ships with."""
    rows: list[dict] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            # Normalise keys (CSV uses spaces / German names; expose ASCII aliases too)
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

ProgressCb = Optional[Callable[[dict], None]]


async def _score_one(
    session: aiohttp.ClientSession,
    cfg: RagConfig,
    row: dict,
    *,
    do_judge: bool,
    do_similarity: bool,
) -> EvalResult:
    qid      = (row.get("ID") or "").strip()
    question = (row.get("Frage") or "").strip()
    expected = (row.get("Erwartete Antwort") or "").strip()

    result = EvalResult(
        id=qid, question=question, expected=expected, predicted="",
        difficulty=row.get("Schwierigkeitsgrad"),
        expected_section=row.get("Kapitel / Abschnitt"),
        typ=row.get("Typ"),
        bewertungstyp=row.get("Bewertungstyp"),
        status=row.get("Status"),
    )

    if not question or not expected:
        result.error = "missing question or expected answer"
        return result

    try:
        out = await rag_query(cfg, session, question)
        result.predicted        = out["answer"]
        result.retrieve_time    = out["timings"]["retrieve"]
        result.generate_time    = out["timings"]["generate"]
        result.retrieved_chunks = [
            {"id": c["id"], "page": c["metadata"].get("page"),
             "source": c["metadata"].get("source"), "distance": c.get("distance")}
            for c in out["contexts"]
        ]
    except Exception as e:
        result.error = f"rag_query failed: {e}"
        return result

    # Metrics — best-effort, individual failures shouldn't blow the row away.
    if do_similarity:
        try:
            embs = await embed(
                session, cfg.embedding,
                [result.predicted, expected],
                timeout=cfg.request_timeout,
            )
            if len(embs) == 2:
                result.semantic_similarity = cosine(embs[0], embs[1])
        except Exception as e:
            result.error = (result.error + " | " if result.error else "") + f"similarity failed: {e}"

    if do_judge:
        try:
            score, reason = await judge_answer(session, cfg, question, expected, result.predicted)
            result.judge_score = score
            result.judge_reason = reason
        except Exception as e:
            result.error = (result.error + " | " if result.error else "") + f"judge failed: {e}"

    return result


async def evaluate(
    cfg: RagConfig,
    session: aiohttp.ClientSession,
    questions: list[dict],
    *,
    do_judge: bool = True,
    do_similarity: bool = True,
    limit: int | None = None,
    concurrency: int = 1,
    progress: ProgressCb = None,
) -> list[EvalResult]:
    """Run the eval loop. Bounded concurrency via a semaphore.

    For local backends with one model loaded, leave concurrency=1.
    For multi-tenant servers, bump it up.
    """
    rows = questions[:limit] if limit else questions
    semaphore = asyncio.Semaphore(max(1, concurrency))
    results: list[Optional[EvalResult]] = [None] * len(rows)

    async def worker(idx: int, row: dict):
        async with semaphore:
            r = await _score_one(session, cfg, row,
                                 do_judge=do_judge, do_similarity=do_similarity)
            results[idx] = r
            if progress:
                progress({
                    "type": "row",
                    "index": idx,
                    "total": len(rows),
                    "id": r.id,
                    "judge_score": r.judge_score,
                    "similarity": r.semantic_similarity,
                    "error": r.error,
                })

    await asyncio.gather(*(worker(i, row) for i, row in enumerate(rows)))
    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _safe_stats(values: list[float]) -> Optional[dict]:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None
    vals_sorted = sorted(vals)
    return {
        "n": len(vals),
        "mean": sum(vals) / len(vals),
        "min":  vals_sorted[0],
        "p50":  vals_sorted[len(vals) // 2],
        "max":  vals_sorted[-1],
    }


def summarize(results: list[EvalResult]) -> dict:
    """Aggregate eval results overall and per difficulty."""
    if not results:
        return {"n_total": 0}

    def slice_stats(subset: list[EvalResult]) -> dict:
        return {
            "n":              len(subset),
            "n_failed":       sum(1 for r in subset if r.error),
            "judge_score":    _safe_stats([r.judge_score for r in subset if r.judge_score is not None]),
            "semantic_similarity": _safe_stats([r.semantic_similarity for r in subset if r.semantic_similarity is not None]),
            "retrieve_time":  _safe_stats([r.retrieve_time for r in subset if r.error is None]),
            "generate_time":  _safe_stats([r.generate_time for r in subset if r.error is None]),
        }

    by_difficulty: dict[str, dict] = {}
    for diff in sorted({(r.difficulty or "unknown") for r in results}):
        subset = [r for r in results if (r.difficulty or "unknown") == diff]
        by_difficulty[diff] = slice_stats(subset)

    return {
        "n_total": len(results),
        "overall": slice_stats(results),
        "by_difficulty": by_difficulty,
    }


def save_results(results: list[EvalResult], summary: dict, path: str | Path) -> None:
    """Write detailed results + summary to a single JSON file."""
    out = {
        "summary": summary,
        "results": [asdict(r) for r in results],
    }
    Path(path).write_text(json.dumps(out, indent=2, ensure_ascii=False))
