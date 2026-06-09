"""
Minimal async OpenAI-compatible client for embeddings + chat.

Both functions take an explicit `aiohttp.ClientSession` so callers can pool
connections across many calls. They take a `ModelConfig` describing the
endpoint. Nothing here reads env vars or globals — pure functions.

Designed to be drop-in usable from server.py / test_llm.py later.
"""

from typing import Any

import aiohttp

from .config import ModelConfig


def _headers(cfg: ModelConfig) -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg.api_key}",
    }


async def embed(
    session: aiohttp.ClientSession,
    cfg: ModelConfig,
    texts: list[str],
    *,
    timeout: float = 120.0,
) -> list[list[float]]:
    """POST /v1/embeddings, return one vector per input text."""
    if not texts:
        return []
    url = f"{cfg.base_url}/embeddings"
    payload = {"model": cfg.model, "input": texts}
    async with session.post(
        url, json=payload, headers=_headers(cfg),
        timeout=aiohttp.ClientTimeout(total=timeout),
    ) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"embed HTTP {resp.status}: {body[:300]}")
        body = await resp.json(content_type=None)
    data = body.get("data") or []
    return [item["embedding"] for item in data]


async def chat(
    session: aiohttp.ClientSession,
    cfg: ModelConfig,
    messages: list[dict],
    *,
    temperature: float = 0.0,
    max_tokens: int = 512,
    timeout: float = 120.0,
    extra: dict[str, Any] | None = None,
) -> dict:
    """POST /v1/chat/completions (non-streaming). Returns a dict with:
        text:            assistant content
        prompt_tokens:   usage.prompt_tokens or None
        completion_tokens: usage.completion_tokens or None
        total_tokens:    usage.total_tokens or None
        raw:             the full response body (for debugging)
    """
    url = f"{cfg.base_url}/chat/completions"
    payload: dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra:
        payload.update(extra)
    async with session.post(
        url, json=payload, headers=_headers(cfg),
        timeout=aiohttp.ClientTimeout(total=timeout),
    ) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"chat HTTP {resp.status}: {body[:300]}")
        body = await resp.json(content_type=None)
    choices = body.get("choices") or []
    text = ""
    if choices:
        msg = choices[0].get("message") or {}
        text = msg.get("content") or ""
    usage = body.get("usage") or {}
    return {
        "text": text,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "raw": body,
    }
