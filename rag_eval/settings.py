from __future__ import annotations

import os
import re as _re
from dataclasses import replace as dataclass_replace
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args, **_kwargs):
        return False


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DEFAULT_ENV_FILE = BASE_DIR / ".env"
DEFAULT_MLFLOW_DB = BASE_DIR / "mlflow.db"
DEFAULT_MLFLOW_ARTIFACT_ROOT = BASE_DIR / "mlartifacts"


@dataclass
class ModelConfig:
    base_url: str
    api_key: str
    model: str

    def is_configured(self) -> bool:
        return bool(self.base_url and self.model)


@dataclass
class RagEvalConfig:
    embedding: ModelConfig
    chat: ModelConfig
    judge: ModelConfig
    pdf_dir: Path
    eval_csv: Path
    chroma_dir: Path
    collection: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    embed_batch_size: int
    chat_max_tokens: int
    chat_temperature: float
    request_timeout: float
    eval_concurrency: int
    mlflow_tracking_uri: str
    mlflow_experiment: str
    mlflow_artifact_root: Path


def _model_from_env(prefix: str, fallback: ModelConfig | None = None) -> ModelConfig:
    base_url = os.environ.get(f"{prefix}_BASE_URL", "").rstrip("/")
    api_key = os.environ.get(f"{prefix}_API_KEY", "none") or "none"
    model = os.environ.get(f"{prefix}_MODEL", "")
    if not base_url and fallback is not None:
        return fallback
    return ModelConfig(base_url=base_url, api_key=api_key, model=model)


def _resolve_path(value: str | Path, *, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _resolve_tracking_uri(value: str) -> str:
    if value.startswith("sqlite:///"):
        db_path = value.removeprefix("sqlite:///")
        resolved = _resolve_path(db_path, base_dir=BASE_DIR)
        return f"sqlite:///{resolved}"
    return value


def load_chat_model_configs(env_file: str | Path = DEFAULT_ENV_FILE) -> list[ModelConfig]:
    """Return all chat model configs defined in .env (CHAT_MODEL, CHAT_MODEL_2, CHAT_MODEL_3, ...)."""
    load_dotenv(str(env_file), override=False)
    configs: list[ModelConfig] = []

    # First config uses the base names (no suffix)
    base_url = os.environ.get("CHAT_BASE_URL", "").rstrip("/")
    api_key = os.environ.get("CHAT_API_KEY", "none") or "none"
    model = os.environ.get("CHAT_MODEL", "")
    if base_url and model:
        configs.append(ModelConfig(base_url=base_url, api_key=api_key, model=model))

    # Additional configs use numeric suffix _2, _3, ...
    suffix = 2
    while True:
        base_url = os.environ.get(f"CHAT_BASE_URL_{suffix}", "").rstrip("/")
        api_key = os.environ.get(f"CHAT_API_KEY_{suffix}", "none") or "none"
        model = os.environ.get(f"CHAT_MODEL_{suffix}", "")
        if not model:
            break
        if not base_url:
            base_url = os.environ.get("CHAT_BASE_URL", "").rstrip("/")
        configs.append(ModelConfig(base_url=base_url, api_key=api_key, model=model))
        suffix += 1

    return configs


def load_embedding_model_configs(env_file: str | Path = DEFAULT_ENV_FILE) -> list[ModelConfig]:
    """Return all embedding model configs defined in .env (EMBEDDING_MODEL, EMBEDDING_MODEL_2, ...)."""
    load_dotenv(str(env_file), override=False)
    configs: list[ModelConfig] = []

    base_url = os.environ.get("EMBEDDING_BASE_URL", "").rstrip("/")
    api_key = os.environ.get("EMBEDDING_API_KEY", "none") or "none"
    model = os.environ.get("EMBEDDING_MODEL", "")
    if base_url and model:
        configs.append(ModelConfig(base_url=base_url, api_key=api_key, model=model))

    suffix = 2
    while True:
        model = os.environ.get(f"EMBEDDING_MODEL_{suffix}", "")
        if not model:
            break
        base_url = os.environ.get(f"EMBEDDING_BASE_URL_{suffix}", "").rstrip("/")
        api_key = os.environ.get(f"EMBEDDING_API_KEY_{suffix}", "none") or "none"
        if not base_url:
            base_url = os.environ.get("EMBEDDING_BASE_URL", "").rstrip("/")
        configs.append(ModelConfig(base_url=base_url, api_key=api_key, model=model))
        suffix += 1

    return configs


def load_judge_model_configs(env_file: str | Path = DEFAULT_ENV_FILE) -> list[ModelConfig]:
    """Return all judge model configs defined in .env (JUDGE_MODEL, JUDGE_MODEL_2, ...).

    If no JUDGE_MODEL is set at all, falls back to the first chat model config
    (preserves existing 'judge defaults to chat' behavior).
    """
    load_dotenv(str(env_file), override=False)
    configs: list[ModelConfig] = []

    base_url = os.environ.get("JUDGE_BASE_URL", "").rstrip("/")
    api_key = os.environ.get("JUDGE_API_KEY", "none") or "none"
    model = os.environ.get("JUDGE_MODEL", "")
    if base_url and model:
        configs.append(ModelConfig(base_url=base_url, api_key=api_key, model=model))

    suffix = 2
    while True:
        model = os.environ.get(f"JUDGE_MODEL_{suffix}", "")
        if not model:
            break
        base_url = os.environ.get(f"JUDGE_BASE_URL_{suffix}", "").rstrip("/")
        api_key = os.environ.get(f"JUDGE_API_KEY_{suffix}", "none") or "none"
        if not base_url:
            base_url = os.environ.get("JUDGE_BASE_URL", "").rstrip("/")
        configs.append(ModelConfig(base_url=base_url, api_key=api_key, model=model))
        suffix += 1

    if not configs:
        # Fall back to first chat model
        chat_configs = load_chat_model_configs(env_file)
        if chat_configs:
            configs = [chat_configs[0]]

    return configs


def apply_overrides(
    cfg: RagEvalConfig,
    embedding: ModelConfig | None = None,
    chat: ModelConfig | None = None,
    judge: ModelConfig | None = None,
) -> RagEvalConfig:
    """Return a copy of cfg with any combination of models replaced.

    When the embedding model changes, cfg.collection is updated to a
    model-specific name so each embedding model uses its own ChromaDB collection.
    """
    new_cfg = cfg
    if embedding is not None and embedding != cfg.embedding:
        slug = _re.sub(r"[^a-zA-Z0-9]", "-", embedding.model)[:32]
        base = _re.sub(r"__emb-.*$", "", cfg.collection)
        new_cfg = dataclass_replace(new_cfg, embedding=embedding, collection=f"{base}__emb-{slug}")
    if chat is not None:
        new_cfg = dataclass_replace(new_cfg, chat=chat)
    if judge is not None:
        new_cfg = dataclass_replace(new_cfg, judge=judge)
    elif chat is not None and new_cfg.judge == cfg.chat:
        # judge was mirroring chat — keep it mirroring the new chat
        new_cfg = dataclass_replace(new_cfg, judge=chat)
    return new_cfg


def apply_chat_override(cfg: RagEvalConfig, chat_model: ModelConfig) -> RagEvalConfig:
    """Backward-compat shim — use apply_overrides() for new code."""
    return apply_overrides(cfg, chat=chat_model)


def load_config(env_file: str | Path = DEFAULT_ENV_FILE) -> RagEvalConfig:
    load_dotenv(str(env_file))

    embedding = _model_from_env("EMBEDDING")
    chat = _model_from_env("CHAT")
    judge = _model_from_env("JUDGE", fallback=chat)

    return RagEvalConfig(
        embedding=embedding,
        chat=chat,
        judge=judge,
        pdf_dir=_resolve_path(os.environ.get("PDF_DIR", "../data"), base_dir=BASE_DIR),
        eval_csv=_resolve_path(os.environ.get("EVAL_CSV", "../data/VDE_FNN_Benchmark.csv"), base_dir=BASE_DIR),
        chroma_dir=_resolve_path(os.environ.get("CHROMA_DIR", "./chroma_db"), base_dir=BASE_DIR),
        collection=os.environ.get("COLLECTION", "vde_fnn"),
        chunk_size=int(os.environ.get("CHUNK_SIZE", "900")),
        chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", "150")),
        top_k=int(os.environ.get("TOP_K", "5")),
        embed_batch_size=int(os.environ.get("EMBED_BATCH_SIZE", "32")),
        chat_max_tokens=int(os.environ.get("CHAT_MAX_TOKENS", "512")),
        chat_temperature=float(os.environ.get("CHAT_TEMPERATURE", "0.0")),
        request_timeout=float(os.environ.get("REQUEST_TIMEOUT", "120")),
        eval_concurrency=int(os.environ.get("EVAL_CONCURRENCY", "5")),
        mlflow_tracking_uri=_resolve_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{DEFAULT_MLFLOW_DB}")),
        mlflow_experiment=os.environ.get("MLFLOW_EXPERIMENT", "rag_eval"),
        mlflow_artifact_root=_resolve_path(os.environ.get("MLFLOW_ARTIFACT_ROOT", str(DEFAULT_MLFLOW_ARTIFACT_ROOT)), base_dir=BASE_DIR),
    )
