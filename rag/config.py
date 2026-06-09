"""
Configuration loader. Reads .env into typed dataclasses.

`ModelConfig` describes a single OpenAI-compatible endpoint (chat or embedding).
`RagConfig` bundles three of them (embedding / chat / judge) plus the rest of
the pipeline settings (paths, chunking, top_k).
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv  # type: ignore[assignment]
except ImportError:  # let import-time work even before deps installed
    def load_dotenv(*_args, **_kwargs):  # type: ignore[no-redef]
        return False


@dataclass
class ModelConfig:
    """OpenAI-compatible endpoint config."""
    base_url: str
    api_key: str
    model: str

    def is_configured(self) -> bool:
        return bool(self.base_url and self.model)


@dataclass
class RagConfig:
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


def _model_from_env(prefix: str, fallback: Optional[ModelConfig] = None) -> ModelConfig:
    base_url = os.environ.get(f"{prefix}_BASE_URL", "").rstrip("/")
    api_key  = os.environ.get(f"{prefix}_API_KEY", "none") or "none"
    model    = os.environ.get(f"{prefix}_MODEL", "")
    if not base_url and fallback is not None:
        return fallback
    return ModelConfig(base_url=base_url, api_key=api_key, model=model)


def load_config(env_file: str = "rag/.env") -> RagConfig:
    """Load .env from the given path. Returns a fully-populated RagConfig."""
    load_dotenv(env_file)

    embedding = _model_from_env("EMBEDDING")
    chat      = _model_from_env("CHAT")
    # Judge defaults to the chat endpoint unless overridden.
    judge     = _model_from_env("JUDGE", fallback=chat)

    return RagConfig(
        embedding=embedding,
        chat=chat,
        judge=judge,
        pdf_dir=Path(os.environ.get("PDF_DIR", "./data")),
        eval_csv=Path(os.environ.get("EVAL_CSV", "./data/VDE_FNN_Benchmark.csv")),
        chroma_dir=Path(os.environ.get("CHROMA_DIR", "./rag/chroma_db")),
        collection=os.environ.get("COLLECTION", "vde_fnn"),
        chunk_size=int(os.environ.get("CHUNK_SIZE", "900")),
        chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", "150")),
        top_k=int(os.environ.get("TOP_K", "5")),
        embed_batch_size=int(os.environ.get("EMBED_BATCH_SIZE", "32")),
        chat_max_tokens=int(os.environ.get("CHAT_MAX_TOKENS", "512")),
        chat_temperature=float(os.environ.get("CHAT_TEMPERATURE", "0.0")),
        request_timeout=float(os.environ.get("REQUEST_TIMEOUT", "120")),
    )
