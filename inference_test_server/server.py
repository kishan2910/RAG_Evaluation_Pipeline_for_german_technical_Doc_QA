#!/usr/bin/env python3
"""
Web frontend for the LLM benchmark tool.

Run:    python server.py [--config config.yaml] [--port 8765]
Open:   http://localhost:8765
"""

import argparse
import asyncio
import secrets
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
from aiohttp import web

from test_llm import (
    DEFAULT_TESTS,
    fetch_models,
    is_local_endpoint,
    load_config,
    run_benchmarks,
    save_csv,
    save_json,
)

ROOT = Path(__file__).parent
INDEX_HTML = ROOT / "index.html"

# In-memory job store. Each job_id -> dict with status / events / results.
JOBS: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

async def index(request: web.Request) -> web.StreamResponse:
    if not INDEX_HTML.exists():
        return web.Response(status=500, text="index.html not found next to server.py")
    return web.FileResponse(INDEX_HTML)


async def get_endpoints(request: web.Request) -> web.Response:
    cfg = request.app["config"]
    eps = [
        {
            "name": ep["name"],
            "url": ep["url"],
            "is_local": is_local_endpoint(ep),
        }
        for ep in cfg.get("endpoints", [])
        if ep.get("enabled", True)
    ]
    return web.json_response({"endpoints": eps, "defaults": DEFAULT_TESTS})


def _write_endpoints_yaml(path: Path, endpoints: list[dict]) -> None:
    """Write a simplified endpoints config back to disk.

    Comments in the original file are not preserved; the simplified schema only
    carries name / url / api_key / enabled per endpoint.
    """
    lines = ["endpoints:"]
    for ep in endpoints:
        lines.append("")
        lines.append(f'  - name: "{ep["name"]}"')
        lines.append(f'    url: "{ep["url"]}"')
        api_key = ep.get("api_key")
        if api_key:
            lines.append(f'    api_key: "{api_key}"')
        if ep.get("enabled") is False:
            lines.append("    enabled: false")
    path.write_text("\n".join(lines) + "\n")


async def add_endpoint(request: web.Request) -> web.Response:
    """Body: {name, url, api_key?}. Persists to config.yaml and reloads."""
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON"}, status=400)

    name = (body.get("name") or "").strip()
    url = (body.get("url") or "").strip()
    api_key = (body.get("api_key") or "").strip() or "none"

    if not name:
        return web.json_response({"error": "name is required"}, status=400)
    if not url:
        return web.json_response({"error": "url is required"}, status=400)

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return web.json_response({"error": "url must be http(s)://host[:port][/path]"}, status=400)

    cfg = request.app["config"]
    endpoints = cfg.setdefault("endpoints", [])
    if any(ep.get("name") == name for ep in endpoints):
        return web.json_response({"error": f"endpoint name already exists: {name}"}, status=409)

    new_ep = {"name": name, "url": url, "api_key": api_key, "enabled": True}
    endpoints.append(new_ep)

    try:
        _write_endpoints_yaml(request.app["config_path"], endpoints)
    except Exception as e:
        # Roll back the in-memory add if we couldn't persist
        endpoints.pop()
        return web.json_response({"error": f"failed to write config: {e}"}, status=500)

    return web.json_response({
        "endpoint": {
            "name": new_ep["name"],
            "url": new_ep["url"],
            "is_local": is_local_endpoint(new_ep),
        }
    })


async def get_models(request: web.Request) -> web.Response:
    name = request.query.get("endpoint")
    cfg = request.app["config"]
    ep = next((e for e in cfg.get("endpoints", []) if e["name"] == name), None)
    if not ep:
        return web.json_response({"error": f"Unknown endpoint: {name}"}, status=404)
    try:
        async with aiohttp.ClientSession() as session:
            models = await fetch_models(session, ep)
        return web.json_response({"endpoint": name, "models": sorted(models)})
    except Exception as e:
        return web.json_response({"endpoint": name, "error": str(e)}, status=502)


async def post_benchmark(request: web.Request) -> web.Response:
    """Body: {selections: [{endpoint: name, model: str}], tests: {...optional overrides...}}"""
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON"}, status=400)

    cfg = request.app["config"]
    ep_by_name = {ep["name"]: ep for ep in cfg.get("endpoints", [])}

    selections_in = body.get("selections") or []
    if not selections_in:
        return web.json_response({"error": "no selections provided"}, status=400)

    selections = []
    for sel in selections_in:
        ep = ep_by_name.get(sel.get("endpoint"))
        if not ep:
            return web.json_response({"error": f"unknown endpoint: {sel.get('endpoint')}"}, status=400)
        if not sel.get("model"):
            return web.json_response({"error": "selection missing model"}, status=400)
        selections.append({"endpoint": ep, "model": sel["model"]})

    tests = dict(DEFAULT_TESTS)
    overrides = body.get("tests") or {}
    for k, v in overrides.items():
        if v is not None:
            tests[k] = v

    job_id = secrets.token_urlsafe(8)
    JOBS[job_id] = {
        "id": job_id,
        "status": "running",
        "started_at": datetime.utcnow().isoformat() + "Z",
        "finished_at": None,
        "selections": [{"endpoint": s["endpoint"]["name"], "model": s["model"]} for s in selections],
        "events": [],
        "summaries": [],
        "raw_count": 0,
        "error": None,
    }

    asyncio.create_task(_run_job(job_id, selections, tests))
    return web.json_response({"job_id": job_id})


async def get_benchmark(request: web.Request) -> web.Response:
    job_id = request.match_info["job_id"]
    job = JOBS.get(job_id)
    if not job:
        return web.json_response({"error": "unknown job"}, status=404)
    # Return job state without potentially huge raw_results blob
    since = int(request.query.get("since", "0"))
    events = job["events"][since:]
    return web.json_response({
        "id": job["id"],
        "status": job["status"],
        "started_at": job["started_at"],
        "finished_at": job["finished_at"],
        "selections": job["selections"],
        "summaries": job["summaries"],
        "raw_count": job["raw_count"],
        "events": events,
        "next_since": since + len(events),
        "error": job["error"],
    })


async def _run_job(job_id: str, selections: list[dict], tests: dict):
    job = JOBS[job_id]

    def on_progress(ev: dict):
        ev = {**ev, "ts": datetime.utcnow().isoformat() + "Z"}
        job["events"].append(ev)
        if ev.get("type") == "summary":
            job["summaries"].append(ev["summary"])

    try:
        all_results, summaries = await run_benchmarks(selections, tests, progress=on_progress)
        job["raw_count"] = len(all_results)
        job["status"] = "done"

        # Persist last run to disk so it survives a server restart.
        try:
            save_json(all_results, [s for s in summaries], "results.json")
            if summaries:
                save_csv(summaries, "results.csv")
        except Exception as e:
            on_progress({"type": "warning", "message": f"failed to persist results: {e}"})

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
    finally:
        job["finished_at"] = datetime.utcnow().isoformat() + "Z"


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def build_app(config_path: str) -> web.Application:
    app = web.Application()
    app["config_path"] = Path(config_path)
    app["config"] = load_config(config_path)

    app.router.add_get("/", index)
    app.router.add_get("/api/endpoints", get_endpoints)
    app.router.add_post("/api/endpoints", add_endpoint)
    app.router.add_get("/api/models", get_models)
    app.router.add_post("/api/benchmark", post_benchmark)
    app.router.add_get("/api/benchmark/{job_id}", get_benchmark)
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="config.yaml")
    parser.add_argument("--port", "-p", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    if not Path(args.config).exists():
        raise SystemExit(f"Config file not found: {args.config}")

    app = build_app(args.config)
    print(f"\nLLM Inference Benchmark UI")
    print(f"  config: {Path(args.config).resolve()}")
    print(f"  open:   http://{args.host}:{args.port}\n")
    web.run_app(app, host=args.host, port=args.port, print=lambda *a, **k: None)


if __name__ == "__main__":
    main()
