# RAG Eval Pipeline

End-to-end evaluation pipeline for RAG systems. Ingests PDFs into ChromaDB, runs a benchmark Q&A dataset through the RAG pipeline, and scores answers with DeepEval metrics. All runs are tracked in MLflow with full span traces.

---

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure `.env`

```bash
cp .env.example .env
# edit .env with your endpoint URLs and model names
```

### 3. Start MLflow UI (optional but recommended)

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open [http://localhost:5000](http://localhost:5000) to browse runs, compare metrics, and download artifacts.

---

## Commands

All commands are run from the `rag_eval/` directory:

### `ingest` — chunk PDFs and build the vector store

```bash
uv run python rag_eval_pipeline.py ingest
uv run python rag_eval_pipeline.py ingest --reset   # drop existing collection first
```

### `ask` — single question through the RAG pipeline

```bash
uv run python rag_eval_pipeline.py ask "What is the title of the document?"
uv run python rag_eval_pipeline.py ask "..." --top-k 10 --show-contexts
```

### `eval` — run the benchmark (ingest must be done first)

```bash
uv run python rag_eval_pipeline.py eval
uv run python rag_eval_pipeline.py eval --limit 20            # first 20 questions only
uv run python rag_eval_pipeline.py eval --output results.json
```

### `run-all` — ingest then eval in one shot

```bash
uv run python rag_eval_pipeline.py run-all
uv run python rag_eval_pipeline.py run-all --reset --limit 50
```

---

## Multi-model experiments

The pipeline can sweep **all combinations** of embedding × chat × judge models in a single command.

### Define multiple models in `.env`

Use numeric suffixes (`_2`, `_3`, …) to register additional models. The base URL and API key can be shared or overridden per model:

```env
# Two embedding models
EMBEDDING_BASE_URL=http://localhost:11434/v1
EMBEDDING_MODEL=nomic-embed-text

EMBEDDING_BASE_URL_2=http://localhost:11434/v1
EMBEDDING_MODEL_2=bge-m3

# Two chat models (sharing the same base URL)
CHAT_BASE_URL=http://localhost:11434/v1
CHAT_MODEL=qwen2.5:14b-instruct

CHAT_MODEL_2=gemma4-31b            # CHAT_BASE_URL_2 falls back to CHAT_BASE_URL

# One judge model
JUDGE_BASE_URL=http://localhost:11434/v1
JUDGE_MODEL=qwen3-coder-30b-a3b-instruct
```

Running `eval` or `run-all` without model flags executes **all combinations** (2 embed × 2 chat × 1 judge = 4 runs above).

### Pin specific models via CLI

Fix any subset of models on the command line; the remaining dimensions use all values from `.env`:

```bash
# Fix chat model, sweep all embedding models from .env
uv run python rag_eval_pipeline.py eval --chat-model gemma4-31b

# Fix all three
uv run python rag_eval_pipeline.py eval \
  --embedding-model bge-m3 \
  --chat-model gemma4-31b \
  --judge-model qwen3-coder-30b-a3b-instruct

# Override base URL for a pinned model
uv run python rag_eval_pipeline.py eval \
  --chat-model my-model \
  --chat-base-url http://other-host:8000/v1
```

---

## Output JSON

Each run produces a JSON file that identifies exactly which models were used:

```json
{
  "config": {
    "embedding_model": "bge-m3",
    "embedding_base_url": "http://...",
    "chat_model": "gemma4-31b",
    "chat_base_url": "http://...",
    "judge_model": "qwen3-coder-30b-a3b-instruct",
    "judge_base_url": "http://..."
  },
  "summary": {
    "total": 500,
    "failed": 2,
    "avg_scores": {
      "RAG Correctness": 0.82,
      "Faithfulness": 0.91
    },
    "by_difficulty": {
      "easy": { "count": 200, "failed": 0, "avg_scores": { "RAG Correctness": 0.88 } }
    }
  },
  "results": [
    {
      "id": "1",
      "question": "...",
      "expected": "...",
      "predicted": "...",
      "scores": { "RAG Correctness": 0.9, "Faithfulness": 1.0 },
      "reasons": { "RAG Correctness": "All key facts present." },
      "difficulty": "easy",
      "section": "1.2",
      "retrieval_time": 0.12,
      "generation_time": 1.45,
      "error": null
    }
  ]
}
```

### File naming for multi-run sweeps

| Scenario              | Output location                                                                 |
| --------------------- | ------------------------------------------------------------------------------- |
| Single combination    | `--output` path as-is (default: `deepeval_results.json`)                    |
| Multiple combinations | `deepeval_results/eval__emb-<embed>__chat-<chat>__judge-<judge>.json` per run |

Each file is also uploaded as an MLflow artifact on the corresponding run.
