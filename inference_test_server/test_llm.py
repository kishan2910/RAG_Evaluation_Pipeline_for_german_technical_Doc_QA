#!/usr/bin/env python3
"""
LLM Endpoint Benchmark Tool
Tests OpenAI-compatible LLM endpoints for latency, throughput, and concurrency.

Usage:
  CLI:   python test_llm.py --endpoint NAME --model MODEL [opts]
  Web:   python server.py    (interactive frontend on http://localhost:8765)
"""

import asyncio
import csv
import json
import os
import sys
import time
from urllib.parse import urlparse
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

import aiohttp
import yaml

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import SpinnerColumn, TextColumn  # noqa: F401
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

console = Console() if HAS_RICH else None


# ---------------------------------------------------------------------------
# Defaults — used when the UI / CLI doesn't override
# ---------------------------------------------------------------------------

DEFAULT_TESTS = {
    "prompts": [
        "What is the capital of France?",
        "Explain the difference between a list and a tuple in Python in one sentence.",
        "Write a haiku about the ocean.",
        "What is 17 * 43?",
    ],
    "system_prompt": "You are a helpful assistant. Be concise.",
    "max_tokens": 256,
    "temperature": 0.7,
    "iterations": 3,
    "concurrency_levels": [1, 5, 10],
    "timeout": 60,
    "streaming": True,
    "stream_include_usage": True,
    "warmup": True,
    "warmup_timeout": 180,  # local models may take a while to load
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    endpoint_name: str
    model: str
    prompt: str
    concurrency: int
    iteration: int
    success: bool
    error: Optional[str] = None
    ttft: Optional[float] = None          # time to first token
    total_time: Optional[float] = None    # wall-clock request time
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    tokens_per_second: Optional[float] = None
    response_text: Optional[str] = None
    is_warmup: bool = False


@dataclass
class EndpointSummary:
    endpoint_name: str
    model: str
    concurrency: int
    total_requests: int
    successful: int
    failed: int
    avg_ttft: Optional[float]
    p50_ttft: Optional[float]
    p95_ttft: Optional[float]
    p99_ttft: Optional[float]
    avg_total_time: Optional[float]
    p50_total_time: Optional[float]
    p95_total_time: Optional[float]
    p99_total_time: Optional[float]
    avg_tokens_per_second: Optional[float]
    throughput_rps: Optional[float]


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load endpoints config. The simplified schema only requires url + name."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    for ep in raw.get("endpoints", []):
        # Expand env vars in api_key
        key = ep.get("api_key", "")
        if isinstance(key, str) and key.startswith("${") and key.endswith("}"):
            var = key[2:-1]
            ep["api_key"] = os.environ.get(var, "")
            if not ep["api_key"]:
                print(f"WARNING: env var {var} not set for endpoint '{ep.get('name')}'")
        ep.setdefault("api_key", "none")
        ep.setdefault("enabled", True)

    return raw


def normalize_base_url(url: str) -> str:
    """Strip trailing slash; append /v1 if no API path is present."""
    base = url.rstrip("/")
    parsed = urlparse(base)
    if not parsed.path or parsed.path == "/":
        base = f"{base}/v1"
    return base


def auth_headers(endpoint: dict) -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {endpoint.get('api_key', 'none')}",
    }


def is_local_endpoint(endpoint: dict) -> bool:
    """Heuristic: endpoint is local if hostname is localhost / loopback / .local."""
    host = (urlparse(endpoint["url"]).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1", "0.0.0.0") or host.endswith(".local")


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

async def fetch_models(session: aiohttp.ClientSession, endpoint: dict, timeout: float = 15) -> list[str]:
    """Call /v1/models and return the list of model IDs. Raises on failure."""
    url = f"{normalize_base_url(endpoint['url'])}/models"
    async with session.get(url, headers=auth_headers(endpoint), timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"HTTP {resp.status}: {body[:200]}")
        body = await resp.json()
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list):
        # Some servers return a flat list
        if isinstance(body, list):
            data = body
        else:
            raise RuntimeError(f"Unexpected /models response shape: {str(body)[:200]}")
    return [m["id"] if isinstance(m, dict) else str(m) for m in data]


# ---------------------------------------------------------------------------
# Core request logic
# ---------------------------------------------------------------------------

async def single_request(
    session: aiohttp.ClientSession,
    endpoint: dict,
    model: str,
    prompt: str,
    config_tests: dict,
    concurrency: int,
    iteration: int,
    *,
    is_warmup: bool = False,
    max_tokens_override: Optional[int] = None,
    timeout_override: Optional[float] = None,
) -> RequestResult:
    base_url = normalize_base_url(endpoint["url"])
    url = f"{base_url}/chat/completions"
    headers = auth_headers(endpoint)

    messages = []
    sys_prompt = config_tests.get("system_prompt")
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": prompt})

    streaming = config_tests.get("streaming", True)
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens_override if max_tokens_override is not None else config_tests.get("max_tokens", 256),
        "temperature": config_tests.get("temperature", 0.7),
        "stream": streaming,
    }
    if streaming and config_tests.get("stream_include_usage", True):
        payload["stream_options"] = {"include_usage": True}

    result = RequestResult(
        endpoint_name=endpoint["name"],
        model=model,
        prompt=prompt,
        concurrency=concurrency,
        iteration=iteration,
        success=False,
        is_warmup=is_warmup,
    )

    timeout_s = timeout_override if timeout_override is not None else config_tests.get("timeout", 60)
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    t_start = time.perf_counter()

    try:
        async with session.post(url, json=payload, headers=headers, timeout=timeout) as resp:
            if resp.status != 200:
                body = await resp.text()
                result.error = f"HTTP {resp.status}: {body[:200]}"
                result.total_time = time.perf_counter() - t_start
                return result

            content_type = resp.headers.get("Content-Type", "").lower()
            is_sse = "event-stream" in content_type or "text/event-stream" in content_type

            if streaming and is_sse:
                # Server-Sent Events: each event is `data: <json>` (with optional
                # space after the colon per the SSE spec). Tolerate `data:` too.
                text_chunks = []
                ttft_recorded = False

                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                    if not line or line.startswith(":"):  # blank or SSE comment
                        continue
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].lstrip()  # strip optional space after colon
                    if not data_str or data_str == "[DONE]":
                        if data_str == "[DONE]":
                            break
                        continue

                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if chunk.get("choices"):
                        delta = chunk["choices"][0].get("delta") or {}
                        content = delta.get("content") or ""
                        if content:
                            if not ttft_recorded:
                                result.ttft = time.perf_counter() - t_start
                                ttft_recorded = True
                            text_chunks.append(content)
                    usage = chunk.get("usage")
                    if usage:
                        result.prompt_tokens = usage.get("prompt_tokens")
                        result.completion_tokens = usage.get("completion_tokens")
                        result.total_tokens = usage.get("total_tokens")

                result.response_text = "".join(text_chunks)
            else:
                # Either streaming was off, OR the server ignored stream=true and
                # returned a normal JSON body. Either way, parse as JSON.
                body = await resp.json(content_type=None)
                choices = body.get("choices") or []
                if choices:
                    msg = choices[0].get("message") or {}
                    result.response_text = msg.get("content") or ""
                else:
                    result.response_text = ""
                usage = body.get("usage") or {}
                result.prompt_tokens = usage.get("prompt_tokens")
                result.completion_tokens = usage.get("completion_tokens")
                result.total_tokens = usage.get("total_tokens")
                # If we asked for streaming but didn't get SSE back, TTFT
                # isn't measurable. Successful request, just no TTFT — the null
                # in the results tells the user.

        result.total_time = time.perf_counter() - t_start
        result.success = True

        if result.completion_tokens and result.total_time:
            result.tokens_per_second = result.completion_tokens / result.total_time

        if result.completion_tokens is None and result.response_text:
            approx = len(result.response_text.split()) * 4 // 3
            result.completion_tokens = approx
            if result.total_time:
                result.tokens_per_second = approx / result.total_time

    except asyncio.TimeoutError:
        result.error = "Timeout"
        result.total_time = time.perf_counter() - t_start
    except Exception as e:
        result.error = str(e)
        result.total_time = time.perf_counter() - t_start

    return result


async def warmup_model(
    session: aiohttp.ClientSession,
    endpoint: dict,
    model: str,
    config_tests: dict,
) -> RequestResult:
    """Send one tiny request so local backends (ollama, llama.cpp) load the model
    into memory before we start timing. Returns a RequestResult marked is_warmup=True."""
    warmup_cfg = {**config_tests, "streaming": False, "stream_include_usage": False}
    return await single_request(
        session,
        endpoint,
        model,
        prompt="Hi",
        config_tests=warmup_cfg,
        concurrency=0,
        iteration=0,
        is_warmup=True,
        max_tokens_override=1,
        timeout_override=config_tests.get("warmup_timeout", DEFAULT_TESTS["warmup_timeout"]),
    )


async def run_concurrent_batch(
    endpoint: dict,
    model: str,
    prompts: list[str],
    config_tests: dict,
    concurrency: int,
    iteration: int,
) -> list[RequestResult]:
    connector = aiohttp.TCPConnector(limit=concurrency + 5)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            single_request(
                session,
                endpoint,
                model,
                prompts[i % len(prompts)],
                config_tests,
                concurrency,
                iteration,
            )
            for i in range(concurrency)
        ]
        return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    sorted_v = sorted(values)
    idx = (len(sorted_v) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_v) - 1)
    frac = idx - lo
    return sorted_v[lo] * (1 - frac) + sorted_v[hi] * frac


def summarize(results: list[RequestResult], endpoint_name: str, model: str, concurrency: int, wall_time: float) -> EndpointSummary:
    # Exclude warmup requests from summaries
    real = [r for r in results if not r.is_warmup]
    successful = [r for r in real if r.success]
    failed = [r for r in real if not r.success]

    ttft_vals = [r.ttft for r in successful if r.ttft is not None]
    total_vals = [r.total_time for r in successful if r.total_time is not None]
    tps_vals = [r.tokens_per_second for r in successful if r.tokens_per_second is not None]

    def safe_avg(vals):
        return sum(vals) / len(vals) if vals else None

    return EndpointSummary(
        endpoint_name=endpoint_name,
        model=model,
        concurrency=concurrency,
        total_requests=len(real),
        successful=len(successful),
        failed=len(failed),
        avg_ttft=safe_avg(ttft_vals),
        p50_ttft=percentile(ttft_vals, 50) if ttft_vals else None,
        p95_ttft=percentile(ttft_vals, 95) if ttft_vals else None,
        p99_ttft=percentile(ttft_vals, 99) if ttft_vals else None,
        avg_total_time=safe_avg(total_vals),
        p50_total_time=percentile(total_vals, 50) if total_vals else None,
        p95_total_time=percentile(total_vals, 95) if total_vals else None,
        p99_total_time=percentile(total_vals, 99) if total_vals else None,
        avg_tokens_per_second=safe_avg(tps_vals),
        throughput_rps=len(successful) / wall_time if wall_time > 0 else None,
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def fmt(val: Optional[float], unit: str = "s", decimals: int = 3) -> str:
    if val is None:
        return "N/A"
    if unit == "ms":
        return f"{val * 1000:.1f} ms"
    if unit == "tok/s":
        return f"{val:.1f} tok/s"
    if unit == "rps":
        return f"{val:.2f} rps"
    return f"{val:.{decimals}f}{unit}"


def print_summary_table(summaries: list[EndpointSummary]):
    if HAS_RICH:
        table = Table(title="LLM Endpoint Benchmark Results", show_lines=True)
        table.add_column("Endpoint", style="cyan", no_wrap=True)
        table.add_column("Model", style="magenta")
        table.add_column("Conc.", justify="right")
        table.add_column("OK/Total", justify="right")
        table.add_column("TTFT avg", justify="right")
        table.add_column("TTFT p95", justify="right")
        table.add_column("Latency avg", justify="right")
        table.add_column("Latency p95", justify="right")
        table.add_column("Tok/s", justify="right")
        table.add_column("RPS", justify="right")

        for s in summaries:
            ok_str = f"{s.successful}/{s.total_requests}"
            color = "green" if s.failed == 0 else "yellow" if s.successful > 0 else "red"
            table.add_row(
                s.endpoint_name, s.model, str(s.concurrency),
                f"[{color}]{ok_str}[/{color}]",
                fmt(s.avg_ttft, "ms"), fmt(s.p95_ttft, "ms"),
                fmt(s.avg_total_time, "ms"), fmt(s.p95_total_time, "ms"),
                fmt(s.avg_tokens_per_second, "tok/s"), fmt(s.throughput_rps, "rps"),
            )
        console.print(table)
    else:
        header = f"{'Endpoint':<20} {'Model':<20} {'Conc':>5} {'OK/Tot':>8} {'TTFT avg':>10} {'TTFT p95':>10} {'Lat avg':>10} {'Lat p95':>10} {'Tok/s':>10} {'RPS':>8}"
        print("\n" + "=" * len(header))
        print("LLM Endpoint Benchmark Results")
        print("=" * len(header))
        print(header)
        print("-" * len(header))
        for s in summaries:
            print(
                f"{s.endpoint_name:<20} {s.model:<20} {s.concurrency:>5} "
                f"{s.successful}/{s.total_requests:>6} "
                f"{fmt(s.avg_ttft, 'ms'):>10} {fmt(s.p95_ttft, 'ms'):>10} "
                f"{fmt(s.avg_total_time, 'ms'):>10} {fmt(s.p95_total_time, 'ms'):>10} "
                f"{fmt(s.avg_tokens_per_second, 'tok/s'):>10} {fmt(s.throughput_rps, 'rps'):>8}"
            )
        print("=" * len(header) + "\n")


def save_json(all_results: list[RequestResult], summaries: list[EndpointSummary], path: str):
    out = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "summaries": [asdict(s) for s in summaries],
        "raw_results": [asdict(r) for r in all_results],
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Detailed results saved to {path}")


def save_csv(summaries: list[EndpointSummary], path: str):
    if not summaries:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(summaries[0]).keys()))
        writer.writeheader()
        for s in summaries:
            writer.writerow(asdict(s))
    print(f"CSV summary saved to {path}")


# ---------------------------------------------------------------------------
# Benchmark runner — used by both CLI and web server
# ---------------------------------------------------------------------------

# Progress callback signature: (event_dict) -> None
ProgressCb = Optional[Callable[[dict], None]]


async def run_selection(
    endpoint: dict,
    model: str,
    tests: dict,
    progress: ProgressCb = None,
) -> tuple[list[RequestResult], list[EndpointSummary]]:
    """Run the full benchmark (warmup + all concurrency levels + iterations) for
    one (endpoint, model) pair."""
    prompts = tests.get("prompts") or DEFAULT_TESTS["prompts"]
    concurrency_levels = tests.get("concurrency_levels") or DEFAULT_TESTS["concurrency_levels"]
    iterations = tests.get("iterations") or DEFAULT_TESTS["iterations"]
    do_warmup = tests.get("warmup", DEFAULT_TESTS["warmup"])

    all_results: list[RequestResult] = []
    summaries: list[EndpointSummary] = []
    label_prefix = f"{endpoint['name']} / {model}"

    # Warmup
    if do_warmup:
        if progress:
            progress({"type": "warmup_start", "endpoint": endpoint["name"], "model": model})
        async with aiohttp.ClientSession() as session:
            t0 = time.perf_counter()
            wr = await warmup_model(session, endpoint, model, tests)
            warmup_time = time.perf_counter() - t0
        all_results.append(wr)
        if progress:
            progress({
                "type": "warmup_done",
                "endpoint": endpoint["name"], "model": model,
                "ok": wr.success, "error": wr.error, "elapsed": warmup_time,
            })
        if not wr.success:
            # Don't try to run timed iterations if the model can't even load.
            if progress:
                progress({"type": "skip", "endpoint": endpoint["name"], "model": model, "reason": wr.error})
            return all_results, summaries

    # Timed iterations
    for concurrency in concurrency_levels:
        batch_results: list[RequestResult] = []
        for it in range(1, iterations + 1):
            if progress:
                progress({
                    "type": "batch_start",
                    "endpoint": endpoint["name"], "model": model,
                    "concurrency": concurrency, "iteration": it, "total_iterations": iterations,
                })
            t0 = time.perf_counter()
            results = await run_concurrent_batch(endpoint, model, prompts, tests, concurrency, it)
            wall = time.perf_counter() - t0
            batch_results.extend(results)
            all_results.extend(results)
            ok = sum(1 for r in results if r.success)
            if progress:
                progress({
                    "type": "batch_done",
                    "endpoint": endpoint["name"], "model": model,
                    "concurrency": concurrency, "iteration": it,
                    "ok": ok, "total": len(results), "wall": wall,
                })
            if it < iterations:
                await asyncio.sleep(0.5)

        total_wall = sum(r.total_time for r in batch_results if r.total_time) or 1.0
        s = summarize(batch_results, endpoint["name"], model, concurrency, total_wall)
        summaries.append(s)
        if progress:
            progress({"type": "summary", "summary": asdict(s)})

    return all_results, summaries


async def run_benchmarks(
    selections: list[dict],   # [{"endpoint": dict, "model": str}, ...]
    tests: dict,
    progress: ProgressCb = None,
) -> tuple[list[RequestResult], list[EndpointSummary]]:
    all_results: list[RequestResult] = []
    summaries: list[EndpointSummary] = []
    for sel in selections:
        ep = sel["endpoint"]
        model = sel["model"]
        if progress:
            progress({"type": "selection_start", "endpoint": ep["name"], "model": model})
        rr, ss = await run_selection(ep, model, tests, progress=progress)
        all_results.extend(rr)
        summaries.extend(ss)
        if progress:
            progress({"type": "selection_done", "endpoint": ep["name"], "model": model})
    return all_results, summaries


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _print_progress(ev: dict):
    t = ev["type"]
    if t == "warmup_start":
        msg = f"  warmup {ev['endpoint']}/{ev['model']} ..."
    elif t == "warmup_done":
        if ev["ok"]:
            msg = f"  warmup ok ({ev['elapsed']:.1f}s)"
        else:
            msg = f"  warmup FAILED: {ev.get('error')}"
    elif t == "batch_start":
        msg = f"  conc={ev['concurrency']} iter={ev['iteration']}/{ev['total_iterations']} ..."
    elif t == "batch_done":
        msg = f"    done {ev['ok']}/{ev['total']} in {ev['wall']:.2f}s"
    elif t == "selection_start":
        msg = f"\n--- {ev['endpoint']} / {ev['model']} ---"
    else:
        return
    if HAS_RICH:
        console.print(msg)
    else:
        print(msg)


async def _list_models_cli(endpoints: list[dict]):
    async with aiohttp.ClientSession() as session:
        for ep in endpoints:
            print(f"\n{ep['name']} ({ep['url']}):")
            try:
                models = await fetch_models(session, ep)
                for m in models:
                    print(f"  - {m}")
            except Exception as e:
                print(f"  ERROR: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark OpenAI-compatible LLM endpoints")
    parser.add_argument("--config", "-c", default="config.yaml")
    parser.add_argument("--endpoint", "-e", help="Endpoint name (substring match)")
    parser.add_argument("--model", "-m", help="Model id to benchmark (required unless --list-models)")
    parser.add_argument("--list-models", action="store_true", help="List models exposed by /v1/models for each endpoint and exit")
    parser.add_argument("--concurrency", "-n", type=int, help="Single concurrency level (overrides defaults)")
    parser.add_argument("--iterations", "-i", type=int)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--save-json", default="results.json")
    parser.add_argument("--save-csv", default="results.csv")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(str(config_path))
    endpoints = [ep for ep in config.get("endpoints", []) if ep.get("enabled", True)]

    if args.endpoint:
        endpoints = [ep for ep in endpoints if args.endpoint.lower() in ep["name"].lower()]

    if not endpoints:
        print("No matching endpoints in config.")
        sys.exit(1)

    if args.list_models:
        asyncio.run(_list_models_cli(endpoints))
        return

    if not args.model:
        print("Error: --model is required (or use --list-models to see what's available, or run server.py for the UI)")
        sys.exit(1)

    tests = dict(DEFAULT_TESTS)
    if args.concurrency is not None:
        tests["concurrency_levels"] = [args.concurrency]
    if args.iterations is not None:
        tests["iterations"] = args.iterations
    if args.no_stream:
        tests["streaming"] = False
    if args.no_warmup:
        tests["warmup"] = False

    selections = [{"endpoint": ep, "model": args.model} for ep in endpoints]

    if HAS_RICH:
        console.print(f"[bold]LLM Inference Benchmark[/bold] — config: {config_path.resolve()}")
        console.print(f"Endpoints: {[ep['name'] for ep in endpoints]}")
        console.print(f"Model: {args.model}")
        console.print(f"Concurrency: {tests['concurrency_levels']}  Iterations: {tests['iterations']}  Streaming: {tests['streaming']}  Warmup: {tests['warmup']}")
    else:
        print(f"LLM Inference Benchmark — endpoints={[ep['name'] for ep in endpoints]} model={args.model}")

    all_results, summaries = asyncio.run(run_benchmarks(selections, tests, progress=_print_progress))

    print_summary_table(summaries)

    if args.save_json:
        save_json(all_results, summaries, args.save_json)
    if args.save_csv and summaries:
        save_csv(summaries, args.save_csv)


if __name__ == "__main__":
    main()
