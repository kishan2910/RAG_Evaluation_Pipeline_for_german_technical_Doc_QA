# RAG Evaluation Pipeline — German Technical Doc QA

Two standalone tools for capacity-planning your inference infrastructure and evaluating the RAG pipeline that runs on top of it.

| Tool | What it solves |
|---|---|
| [`inference_test_server/`](inference_test_server/) | How many concurrent users fit on my GPU? Which model handles my load target? |
| [`rag_eval/`](rag_eval/) | How accurate is my RAG pipeline across different model combinations? |

---

## Problem 1 — Inference Capacity Planning

> *"I have a GPU. Which model fits, and how many users can it handle simultaneously?"*

### GPU Predictor

The **GPU Predictor** tab answers two questions without hitting any endpoint:

1. **Max concurrent users** — how many requests can be served in parallel before VRAM runs out at a given context length.
2. **Decode speed** — estimated tokens/sec for a single request (mid-context average + empty→full range).

Pick a GPU preset (RTX Pro 5000 Blackwell, A100, H100, …) or enter VRAM and memory bandwidth manually. Pick a model preset (Llama 3.1, Qwen 3, Gemma 4, Mistral, DeepSeek, Phi 3, …) or enter architecture params directly. The calculator fills in the rest.

![GPU Predictor — 5 concurrent users, 28 tok/s on RTX Pro 5000 Blackwell with Llama 3.1 70B](images/Screenshot%202026-06-09%20142942.png)

Key formulas used:

```
VRAM_available      = VRAM_total − weights_total − overhead
KV_cache_per_request= 2 × num_layers × num_kv_heads × head_dim × bytes_per_element × context_length
max_concurrency     = ⌊ VRAM_available / KV_cache_per_request ⌋

TPOT                = (weights_active + KV_active) / (memory_bandwidth × MBU)
tokens_per_second   = 1 / TPOT
```

For MoE models `weights_active < weights_total` — only the routed experts are read per decode step, so MoE models are faster than their parameter count suggests.

### Live Benchmark

The **Benchmark** tab measures real latency against your running endpoints (ollama, llama.cpp, vLLM, SGLang, TGI, hosted APIs — anything OpenAI-compatible).

![Benchmark results — qwen3-coder-30b-a3b-instruct at concurrency 1/5/10 on hs-flensburg endpoint](images/Screenshot%202026-06-09%20142846.png)

Results show TTFT avg/p95, end-to-end latency avg/p95, tok/s, and RPS per `(endpoint, model, concurrency)` combination. Cross-reference these numbers with the GPU Predictor to validate predictions against real hardware.

### Quickstart

```bash
cd inference_test_server
uv sync
uv run python server.py          # open http://127.0.0.1:8765
```

CLI alternative:

```bash
uv run python test_llm.py --list-models
uv run python test_llm.py -e mylab -m qwen3-coder-30b-a3b-instruct -n 1,5,10
```

See [`inference_test_server/README.md`](inference_test_server/README.md) for full configuration, streaming parser details, and API reference.

---

## Problem 2 — RAG Evaluation with MLflow

> *"My pipeline retrieves and answers questions from German technical documents. Which embedding + chat + judge model combination produces the best results?"*

### What it does

The pipeline ingests PDFs into ChromaDB, runs a benchmark Q&A dataset through the full RAG stack, and scores each answer with DeepEval metrics (RAG Correctness, Faithfulness). Every run is tracked in MLflow with full span traces — inputs, retrieved contexts, generated answers, and scores are all stored and comparable.

### MLflow Observability

MLflow tracks every evaluation run as a named experiment. The Overview dashboard shows trace count, latency percentiles (p50/p90/p99), and error rates across runs.

![MLflow Overview — 4,374 traces, 14.23s avg latency, 4.0% error rate over 7 days](images/Screenshot%202026-06-09%20134441.png)

Individual traces show side-by-side comparisons of retrieved contexts, generated answers, and judge scores across model combinations:

![MLflow trace comparison — two model combos side by side with retrieved docs and predicted answers](images/Screenshot%202026-06-09%20134059.png)

![MLflow trace detail — full input/output spans with retrieved document scores](images/Screenshot%202026-06-09%20134223.png)

### Multi-model sweep

Define multiple embedding, chat, and judge models in `.env` using numeric suffixes. A single `eval` command runs **all combinations**:

```env
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_MODEL_2=bge-m3

CHAT_MODEL=qwen2.5:14b-instruct
CHAT_MODEL_2=gemma4-31b

JUDGE_MODEL=qwen3-coder-30b-a3b-instruct
```

2 embedding × 2 chat × 1 judge = **4 MLflow runs**, each saved to `deepeval_results/eval__emb-<embed>__chat-<chat>__judge-<judge>.json`.

### Quickstart

```bash
cd rag_eval
uv sync
cp .env.example .env          # fill in endpoint URLs and model names

# start MLflow UI (optional but recommended)
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# ingest PDFs, then run the full eval sweep
uv run python rag_eval_pipeline.py run-all
```

Open [http://localhost:5000](http://localhost:5000) to browse runs, compare metrics, and download artifacts.

See [`rag_eval/README.md`](rag_eval/README.md) for the full command reference, output JSON schema, and multi-model configuration guide.

---

## Repository layout

```
inference_test_server/   # capacity-planning tool (GPU Predictor + live benchmark)
rag_eval/                # RAG evaluation pipeline (DeepEval metrics + MLflow tracking)
rag/                     # shared RAG client / pipeline / ingest helpers
data/                    # benchmark Q&A dataset (techqa.json)
images/                  # screenshots referenced in this README
```
