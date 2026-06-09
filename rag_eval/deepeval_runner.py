from __future__ import annotations

import asyncio
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from deepeval.metrics import (
    FaithfulnessMetric,
    GEval,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from openai import AsyncOpenAI, OpenAI

from mlflow_tracking import start_span
from rag_query import rag_answer
from settings import ModelConfig, RagEvalConfig
from vectorstore import open_collection


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class EvalRow:
    id: str
    question: str
    expected: str
    predicted: str
    scores: dict[str, float | None]
    reasons: dict[str, str | None]
    difficulty: str | None
    section: str | None
    status: str | None
    retrieval_time: float
    generation_time: float
    error: str | None


class OpenAICompatibleJudge(DeepEvalBaseLLM):
    def __init__(self, cfg: ModelConfig, client: OpenAI, async_client: AsyncOpenAI):
        self.cfg = cfg
        self.client = client
        self.async_client = async_client

    def get_model_name(self) -> str:
        return self.cfg.model

    def load_model(self):
        return self

    def _call_chat(self, prompt: str) -> str:
        with start_span(
            "deepeval.judge_call",
            span_type="LLM",
            attributes={"model": self.cfg.model, "base_url": self.cfg.base_url},
        ) as span:
            span.set_inputs({"prompt": prompt})
            response = self.client.chat.completions.create(
                model=self.cfg.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1024,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            choices = response.choices or []
            if not choices:
                span.set_outputs({"content": ""})
                return ""
            content = choices[0].message.content or ""
            span.set_outputs({"content": content[:500]})
            return content

    async def _a_call_chat(self, prompt: str) -> str:
        with start_span(
            "deepeval.judge_call",
            span_type="LLM",
            attributes={"model": self.cfg.model, "base_url": self.cfg.base_url},
        ) as span:
            span.set_inputs({"prompt": prompt})
            response = await self.async_client.chat.completions.create(
                model=self.cfg.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1024,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            choices = response.choices or []
            if not choices:
                span.set_outputs({"content": ""})
                return ""
            content = choices[0].message.content or ""
            span.set_outputs({"content": content[:500]})
            return content

    def _normalize_structured_output(self, content: str) -> str:
        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        match = _JSON_BLOCK_RE.search(text)
        if match:
            text = match.group(0)

        if not text:
            raise ValueError("LLM returned an empty response; cannot parse structured output")
        data = json.loads(text)
        return json.dumps(data)

    def generate(self, prompt: str, schema: Any | None = None) -> Any:
        content = self._call_chat(prompt)
        if schema is None:
            return content
        return schema.model_validate_json(self._normalize_structured_output(content))

    async def a_generate(self, prompt: str, schema: Any | None = None) -> Any:
        content = await self._a_call_chat(prompt)
        if schema is None:
            return content
        return schema.model_validate_json(self._normalize_structured_output(content))


def load_eval_rows(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    return rows


def _make_metrics(judge: OpenAICompatibleJudge) -> list:
    return [
        GEval(
            name="RAG Correctness",
            criteria=(
                "Bewerte, wie gut die vorhergesagte Antwort zur erwarteten Antwort passt. "
                "Achte auf fachliche Korrektheit, Vollstandigkeit und ob Kernaussagen fehlen."
            ),
            evaluation_steps=[
                "Prufe, ob die Kerninformation aus der erwarteten Antwort enthalten ist.",
                "Ziehe Punkte ab, wenn zentrale Fakten fehlen oder verfalscht sind.",
                "Ignoriere rein stilistische Unterschiede ohne Bedeutungsanderung.",
            ],
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            model=judge,
            async_mode=True,
        ),
        FaithfulnessMetric(model=judge, async_mode=True),
    ]


async def _measure_all(
    metrics: list, case: LLMTestCase
) -> tuple[dict[str, float | None], dict[str, str | None]]:
    async def _one(metric) -> tuple[str, float | None, str | None]:
        metric_name = metric.__name__
        try:
            await metric.a_measure(case, _show_indicator=False)
            score = float(metric.score) if metric.score is not None else None
            return metric_name, score, metric.reason
        except Exception as exc:
            return metric_name, None, f"[metric error] {exc}"

    results = await asyncio.gather(*[_one(m) for m in metrics])
    scores = {name: score for name, score, _ in results}
    reasons = {name: reason for name, _, reason in results}
    return scores, reasons


async def evaluate_dataset(
    cfg: RagEvalConfig,
    embedding_client: AsyncOpenAI,
    chat_client: AsyncOpenAI,
    judge_client: OpenAI,
    judge_async_client: AsyncOpenAI,
    *,
    limit: int | None = None,
    top_k: int | None = None,
    progress: Callable[[EvalRow, int], None] | None = None,
) -> list[EvalRow]:
    rows = load_eval_rows(cfg.eval_csv)
    if limit is not None:
        rows = rows[:limit]

    chroma_collection = open_collection(cfg.chroma_dir, cfg.collection)
    judge_model = OpenAICompatibleJudge(cfg.judge, judge_client, judge_async_client)
    semaphore = asyncio.Semaphore(cfg.eval_concurrency)
    out: list[EvalRow | None] = [None] * len(rows)

    total_rows = len(rows)

    async def _eval_one(index: int, row: dict) -> None:
        async with semaphore:
            question = (row.get("Frage") or "").strip()
            expected = (row.get("Erwartete Antwort") or "").strip()
            row_id = (row.get("ID") or str(index)).strip()

            print(f"  [{index}/{total_rows}] starting row {row_id} — {question[:60]}...")

            with start_span(
                f"row_{row_id}",
                span_type="EVALUATOR",
                attributes={
                    "row_id": row_id,
                    "difficulty": row.get("Schwierigkeitsgrad"),
                    "section": row.get("Kapitel / Abschnitt"),
                    "question_preview": question[:80],
                },
            ) as row_span:
                row_span.set_inputs({"id": row_id, "question": question, "expected": expected})

                if not question or not expected:
                    eval_row = EvalRow(
                        id=row_id,
                        question=question,
                        expected=expected,
                        predicted="",
                        scores={},
                        reasons={},
                        difficulty=row.get("Schwierigkeitsgrad"),
                        section=row.get("Kapitel / Abschnitt"),
                        status=row.get("Status"),
                        retrieval_time=0.0,
                        generation_time=0.0,
                        error="missing question or expected answer",
                    )
                    row_span.set_outputs({"error": eval_row.error})
                    out[index - 1] = eval_row
                    if progress:
                        progress(eval_row, index)
                    return

                retrieval_time = 0.0
                generation_time = 0.0
                predicted = ""
                error: str | None = None
                scores: dict[str, float | None] = {}
                reasons: dict[str, str | None] = {}

                try:
                    rag_result = await rag_answer(cfg, embedding_client, chat_client, question, top_k=top_k, collection=chroma_collection)
                    predicted = rag_result["answer"]
                    retrieval_time = float(rag_result["timings"]["retrieve"])
                    generation_time = float(rag_result["timings"]["generate"])
                except Exception as exc:
                    error = f"[RAG] {exc}"
                    print(f"  row {row_id}: RAG error — {exc}")

                if error is None:
                    retrieval_context = [ctx["text"] for ctx in rag_result["contexts"]]
                    case = LLMTestCase(
                        input=question,
                        actual_output=predicted,
                        expected_output=expected,
                        retrieval_context=retrieval_context,
                    )
                    metrics = _make_metrics(judge_model)
                    try:
                        scores, reasons = await _measure_all(metrics, case)
                    except Exception as exc:
                        error = f"[judge] {exc}"
                        print(f"  row {row_id}: judge error — {exc}")

                eval_row = EvalRow(
                    id=row_id,
                    question=question,
                    expected=expected,
                    predicted=predicted,
                    scores=scores,
                    reasons=reasons,
                    difficulty=row.get("Schwierigkeitsgrad"),
                    section=row.get("Kapitel / Abschnitt"),
                    status=row.get("Status"),
                    retrieval_time=retrieval_time,
                    generation_time=generation_time,
                    error=error,
                )

                if error:
                    row_span.set_status("ERROR")
                    row_span.set_attribute("exception.message", error)
                    row_span.set_outputs({"error": error})
                else:
                    row_span.set_outputs({
                        "predicted": predicted[:500],
                        "scores": scores,
                    })

                out[index - 1] = eval_row
                if progress:
                    progress(eval_row, index)

    tasks = [_eval_one(i, r) for i, r in enumerate(rows, start=1)]
    await asyncio.gather(*tasks)

    return [r for r in out if r is not None]


def summarize(rows: list[EvalRow]) -> dict[str, Any]:
    total = len(rows)
    failed = [r for r in rows if r.error]

    metric_names: set[str] = set()
    for row in rows:
        metric_names.update(row.scores.keys())

    avg_scores: dict[str, float | None] = {}
    for name in metric_names:
        vals = [r.scores[name] for r in rows if r.scores.get(name) is not None]
        avg_scores[name] = sum(vals) / len(vals) if vals else None

    by_difficulty: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row.difficulty or "unknown"
        by_difficulty.setdefault(key, {"count": 0, "failed": 0, "_score_lists": {}})
        by_difficulty[key]["count"] += 1
        if row.error:
            by_difficulty[key]["failed"] += 1
        for name, score in row.scores.items():
            by_difficulty[key]["_score_lists"].setdefault(name, [])
            if score is not None:
                by_difficulty[key]["_score_lists"][name].append(score)

    for diff in by_difficulty.values():
        score_lists = diff.pop("_score_lists")
        diff["avg_scores"] = {
            name: (sum(v) / len(v) if v else None)
            for name, v in score_lists.items()
        }

    return {
        "total": total,
        "failed": len(failed),
        "avg_scores": avg_scores,
        "by_difficulty": by_difficulty,
    }


def save_results(
    rows: list[EvalRow],
    summary: dict[str, Any],
    output_path: str | Path,
    *,
    config: dict | None = None,
) -> None:
    payload = {
        "config": config or {},
        "summary": summary,
        "results": [asdict(row) for row in rows],
    }
    Path(output_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
