"""
RAG benchmark module.

Public surface (importable from server.py / test_llm.py):

    from rag.config import load_config, RagConfig, ModelConfig
    from rag.client import embed, chat
    from rag.ingest import extract_pdf, chunk_text, ingest_documents
    from rag.pipeline import retrieve, rag_query
    from rag.evaluation import load_questions, evaluate, summarize

All functions accept an explicit aiohttp.ClientSession so callers can pool
connections; nothing reads globals or env vars at call time (only load_config
does, and only once).
"""

from .config import ModelConfig, RagConfig, load_config

__all__ = ["ModelConfig", "RagConfig", "load_config"]
