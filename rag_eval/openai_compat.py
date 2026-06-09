from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI, OpenAI

from mlflow_tracking import start_span
from settings import ModelConfig


def make_async_client(cfg: ModelConfig, *, timeout: float) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=timeout)


def make_sync_client(cfg: ModelConfig, *, timeout: float) -> OpenAI:
    return OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=timeout)


async def embed(
    client: AsyncOpenAI,
    cfg: ModelConfig,
    texts: list[str],
) -> list[list[float]]:
    if not texts:
        return []
    with start_span(
        "openai.embed",
        span_type="EMBEDDING",
        attributes={
            "model": cfg.model,
            "base_url": cfg.base_url,
            "input_count": len(texts),
        },
    ) as span:
        span.set_inputs({"texts": texts[:5], "input_count": len(texts)})
        response = await client.embeddings.create(model=cfg.model, input=texts)
        vectors = [item.embedding for item in response.data]
        span.set_outputs({
            "embedding_count": len(vectors),
            "embedding_dimensions": len(vectors[0]) if vectors else 0,
        })
        return vectors


async def chat(
    client: AsyncOpenAI,
    cfg: ModelConfig,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra:
        payload.update(extra)

    with start_span(
        "openai.chat",
        span_type="LLM",
        attributes={
            "model": cfg.model,
            "base_url": cfg.base_url,
            "message_count": len(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    ) as span:
        span.set_inputs({"messages": messages, "parameters": payload})
        response = await client.chat.completions.create(**payload)

        text = ""
        choices = response.choices or []
        if choices:
            text = choices[0].message.content or ""

        usage = response.usage
        result = {
            "text": text,
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "total_tokens": usage.total_tokens if usage else None,
            "raw": response.model_dump(mode="json"),
        }
        span.set_outputs({
            "text_preview": text[:500],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["total_tokens"],
        })
        return result
