from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient

from settings import RagEvalConfig


def configure_mlflow(cfg: RagEvalConfig) -> None:
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)


def _ensure_experiment(cfg: RagEvalConfig) -> str:
    configure_mlflow(cfg)
    cfg.mlflow_artifact_root.mkdir(parents=True, exist_ok=True)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(cfg.mlflow_experiment)
    if experiment is None:
        return client.create_experiment(
            cfg.mlflow_experiment,
            artifact_location=cfg.mlflow_artifact_root.as_uri(),
        )
    return experiment.experiment_id


def start_run(cfg: RagEvalConfig, run_name: str, params: dict[str, Any] | None = None):
    experiment_id = _ensure_experiment(cfg)
    run = mlflow.start_run(run_name=run_name, experiment_id=experiment_id)

    default_params = {
        "collection": cfg.collection,
        "pdf_dir": str(cfg.pdf_dir),
        "eval_csv": str(cfg.eval_csv),
        "chunk_size": cfg.chunk_size,
        "chunk_overlap": cfg.chunk_overlap,
        "top_k": cfg.top_k,
        "embed_batch_size": cfg.embed_batch_size,
        "chat_model": cfg.chat.model,
        "embedding_model": cfg.embedding.model,
        "judge_model": cfg.judge.model,
    }
    mlflow.log_params(default_params)
    if params:
        cli_params = {
            f"cli_{key}": str(value)
            for key, value in params.items()
            if value is not None
        }
        if cli_params:
            mlflow.log_params(cli_params)
    mlflow.set_tags({
        "chat_model": cfg.chat.model,
        "embedding_model": cfg.embedding.model,
        "judge_model": cfg.judge.model,
        "same_judge": str(cfg.judge.model == cfg.chat.model),
    })
    return run


@contextmanager
def start_span(name: str, *, span_type: str = "UNKNOWN", attributes: dict[str, Any] | None = None):
    with mlflow.start_span(name=name, span_type=span_type, attributes=attributes) as span:
        yield span


def log_metrics(metrics: dict[str, float | int | None], *, step: int | None = None) -> None:
    for key, value in metrics.items():
        if value is None:
            continue
        if step is None:
            mlflow.log_metric(key, float(value))
        else:
            mlflow.log_metric(key, float(value), step=step)


def log_artifact_if_exists(path: str | Path) -> None:
    p = Path(path)
    if p.exists():
        mlflow.log_artifact(str(p))
