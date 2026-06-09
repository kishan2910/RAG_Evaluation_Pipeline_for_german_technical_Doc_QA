# LLM Inference Test

A benchmark + capacity-planning tool for OpenAI-compatible LLM endpoints
(ollama, llama.cpp, vLLM, SGLang, TGI, hosted APIs, …).

Two surfaces:

- **Web UI** (`server.py`) — point-and-click endpoint discovery, model
  selection via `/v1/models`, live benchmark runs, and a GPU concurrency /
  tokens-per-second predictor.
- **CLI** (`test_llm.py`) — scriptable benchmark runs.

Both surfaces share the same core (`single_request`, `run_benchmarks`,
warmup, summarization).

---

## Quickstart

```bash
uv sync                                  # install deps
uv run python server.py                  # open http://127.0.0.1:8765
```

The default config in `config.yaml` ships with three example endpoints; edit
to point at your own, or add new ones from the UI ("+ Add endpoint" in the
endpoints panel — writes back to `config.yaml`).

CLI:

```bash
uv run python test_llm.py --list-models                  # discover models per endpoint
uv run python test_llm.py -e mylab -m chat-small         # benchmark one model
uv run python test_llm.py -e mylab -m chat-small -n 5 -i 2 --no-warmup
```

---

## Web UI

Top-bar tabs route between two pages.

### Benchmark page

- **Endpoints & Models** — each endpoint from `config.yaml` is listed.
  Click **Load models** to fetch `GET /v1/models` and pick which to test.
- **Add endpoint** — opens an inline form (name / URL / api_key); on save it
  appends to `config.yaml` and refreshes the list.
- **Test parameters** — concurrency levels (comma list), iterations, max
  tokens, timeout, warmup toggle, streaming toggle.
- **Run benchmark** — kicks off a job. Status pill pulses while running.
- **Status** log — live progress: warmup start/done, batch progress,
  per-concurrency summaries.
- **Results** — table of every `(endpoint, model, concurrency)` summary
  with TTFT avg / p95, latency avg / p95, tok/s, RPS.

Results are also persisted to `results.json` and `results.csv` after every
job so they survive a server restart.

### GPU Predictor page

A pure client-side calculator for two questions:

1. How many concurrent users fit in VRAM at a given context length?
2. What single-request decode speed can I expect?

Inputs:

- **GPU**: preset (fills VRAM + memory bandwidth), VRAM total (GB), memory
  bandwidth (GB/s), MBU (Memory Bandwidth Utilization, %).
- **Model architecture**: preset (fills layers / KV heads / head_dim / params),
  num_layers, num_kv_heads, head_dim, total params, active params,
  weight quantization, KV cache dtype, overhead.
- **Request**: context length.

Outputs:

- **Max concurrent users** hero — `⌊ VRAM_available / KV_cache_per_request ⌋`.
- **Tokens / sec (decode est.)** hero — mid-context average + best/worst range.
- Breakdown stats: weights total, weights active, VRAM available, KV / token,
  KV / request.

#### Formulas

VRAM and concurrency:

```
weights_total       = total_params × bytes_per_param
weights_active      = active_params × bytes_per_param        # = total for dense
VRAM_available      = VRAM_total − weights_total − overhead
KV_cache_per_token  = 2 × num_layers × num_kv_heads × head_dim × bytes_per_element
KV_cache_per_request= KV_cache_per_token × context_length
max_concurrency     = ⌊ VRAM_available / KV_cache_per_request ⌋
```

Single-request decode speed (memory-bound, per Databricks / NVIDIA inference
optimization references):

```
TPOT              = (weights_active + KV_active) / (memory_bandwidth × MBU)
tokens_per_second = 1 / TPOT
```

`KV_active` grows from 0 at the start of generation to `KV_cache_per_request`
at the end. The hero number uses the mid-context average; the sub-line shows
the empty→full range.

For MoE models only the routed experts are read per decode step, so
`weights_active < weights_total`. Both quantities matter: total drives the
VRAM footprint, active drives speed.

Bundled presets cover GPUs from Ampere through Blackwell (incl. RTX Pro
Blackwell workstation) and models including Llama 3.1, Qwen 2.5 / 3 / 3.6,
Mistral / Mixtral, Gemma 2 / 3 / 4, DeepSeek-V2-Lite, Phi 3.

---

## CLI

```
usage: test_llm.py [-h] [--config CONFIG] [--endpoint ENDPOINT] [--model MODEL]
                   [--list-models] [--concurrency CONCURRENCY]
                   [--iterations ITERATIONS] [--no-stream] [--no-warmup]
                   [--save-json SAVE_JSON] [--save-csv SAVE_CSV]
```

Key flags:

| Flag | Effect |
|---|---|
| `--endpoint NAME` / `-e` | substring-match against endpoint names |
| `--model MODEL` / `-m` | model id to benchmark (required unless `--list-models`) |
| `--list-models` | print `/v1/models` for each matched endpoint and exit |
| `--concurrency N` / `-n` | replace the default `[1, 5, 10]` with a single level |
| `--iterations N` / `-i` | override default 3 iterations |
| `--no-stream` | disable streaming (TTFT becomes unmeasurable) |
| `--no-warmup` | skip warmup request |

CLI defaults live in `DEFAULT_TESTS` (in `test_llm.py`). The UI's
**Test parameters** form overrides them per-run.

---

## Configuration

`config.yaml` is intentionally minimal — only endpoints:

```yaml
endpoints:
  - name: "ollama"
    url: "http://0.0.0.0:11434/v1"
    api_key: "ollama"

  -...
```

Notes:

- `url` may include the `/v1` path or not — it's auto-appended.
- `api_key` accepts `${ENV_VAR}` for env expansion.
- Optional `enabled: false` hides an endpoint without deleting it.
- The UI's "+ Add endpoint" button writes new entries here.
  (Comments in the file are not preserved across writes — the writer rebuilds
  the file from the parsed structure.)

Defaults (prompts, system prompt, max_tokens, timeout, concurrency_levels,
warmup) live in `DEFAULT_TESTS` in `test_llm.py`. Edit there to change them
for both the CLI and the UI.

---

## Warmup

Local backends (ollama, llama.cpp) lazy-load models. Before any timed
iterations the runner sends one untimed `max_tokens=1` request per
`(endpoint, model)` pair using a longer timeout (180 s by default). The
warmup result is flagged `is_warmup=True` and excluded from summaries.

If warmup fails the runner skips that `(endpoint, model)` rather than
producing junk TTFT data.

Toggle from the UI or with `--no-warmup` on the CLI.

---

## Streaming parser

`single_request()` checks `Content-Type` on the response:

- `text/event-stream` — SSE parser. Tolerates `data:{...}` (no space after
  colon), `:` comment lines, blank keep-alives, CRLF/LF endings.
- Anything else — JSON body parser. Used both when streaming was disabled
  and when a server returns plain JSON despite `stream: true`. TTFT stays
  `null` in this case (intrinsically not measurable).

---

## File layout

```
config.yaml          # endpoint definitions
test_llm.py          # core + CLI
server.py            # aiohttp web server
index.html           # single-page frontend (Benchmark + Predictor)
results.json         # latest run, full results (overwritten per run)
results.csv          # latest run, summary table
pyproject.toml       # deps (aiohttp, pyyaml, rich)
```

### Backend API

`server.py` exposes:

| Method + path | Purpose |
|---|---|
| `GET /api/endpoints` | List endpoints from config (with `is_local` flag) + `DEFAULT_TESTS` |
| `POST /api/endpoints` | Add an endpoint; persists to `config.yaml` |
| `GET /api/models?endpoint=NAME` | Proxy `/v1/models` for that endpoint |
| `POST /api/benchmark` | Start a benchmark job, returns `{job_id}` |
| `GET /api/benchmark/{id}?since=N` | Poll job state — incremental events + summaries |

Jobs live in-process (`server.JOBS`). Restart loses history; persistent
results are written to `results.json` / `results.csv` after each run.

---

## Dependencies

- Python ≥ 3.11
- `aiohttp` (HTTP client + server)
- `pyyaml` (config loading + writing)
- `rich` (optional, prettier CLI output)

Managed via `uv` (`pyproject.toml` + `uv.lock`).

---

## License & attribution

Internal tool — no public license set. Inference formulas reference
Databricks ("LLM Inference Performance Engineering"), the NVIDIA inference
optimization blog, and the Mind the Memory Gap (arXiv 2503.08311) paper.
