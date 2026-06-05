# RAG Evaluation Pipeline — German Technical Document QA

An end-to-end Retrieval-Augmented Generation (RAG) system built to answer questions over German technical standards (VDE FNN norms), with a full automated evaluation framework using DeepEval metrics and MLflow experiment tracking.

---

## Problem

VDE FNN publishes technical grid connection and operation rules that govern how electrical equipment interacts with the German power grid. These documents are essential for network operators, equipment manufacturers, installers, and compliance teams. However, the rules are distributed across numerous lengthy standards, application guides and technical documents, making it time-consuming to locate relevant requirements and interpret them correctly. Domain experts need fast, accurate and context-aware access to information from these documents without manually searching through hundreds of pages.

**Challenge:** Building a RAG system is straightforward — *evaluating whether it actually works* is not. Standard accuracy metrics don't apply to open-ended generative QA. This project solves both: it builds the RAG pipeline and provides a rigorous, multi-metric automated evaluation framework against a 500-question German-language benchmark.

---

## Solution Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION                                │
│  PDF Documents → Text Extraction → Chunking (900 chars)         │
│                → Embedding (Qwen3 Embedding) → ChromaDB         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RAG QUERY PIPELINE                         │
│  User Question → Embed Query → ChromaDB (top-k=5 retrieval)     │
│               → Build Prompt (DE system prompt + context)       │
│               → LLM Generation (Qwen 27B) → Answer              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   AUTOMATED EVALUATION                          │
│  500 benchmark Q&A pairs → RAG answer → DeepEval judge          │
│                                                                 │
│  Metrics (LLM-as-Judge via async concurrent evaluation):        │
│  ├── GEval               — RAG Correctness                      │
│  ├── Contextual Recall   — did retrieval find the right chunks? │
│  ├── Contextual Precision — were retrieved chunks relevant?     │
│  ├── Faithfulness        — is the answer grounded in context?   │
│  └── Answer Relevancy    — does the answer address the question?│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MLFLOW TRACKING                              │
│  500 individual traces (one per question) + experiment metrics  │
│  Visualise scores by difficulty level, section and metric       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Vector Store | ChromaDB (persistent) |
| LLM Backend | OpenAI-compatible API (Locally deployed LLMs via inference vllm inference server) |
| Embedding | Qwen3 Embedding 4B |
| Chat / Generation | Qwen3 27B Instruct |
| Evaluation Framework | DeepEval |
| Experiment Tracking | MLflow (traces + metrics + artifacts) |
| PDF Parsing | pypdf |
| Package Management | uv |

---

## Key Design Decisions

- **LLM-as-Judge evaluation**: All 5 metrics use a judge LLM (same OpenAI-compatible endpoint) to score each answer. This enables evaluation without human labellers.
- **Custom GEval rubric in German**: Standard metrics are language-agnostic; the GEval metric uses a German-language evaluation prompt with domain-specific criteria (factual correctness, completeness, key information presence).
- **Incremental ingestion**: ChromaDB chunk IDs are content-addressed (`filename::page::chunk_index`). Re-running ingest only embeds new or changed chunks.
- **Per-question MLflow traces**: Each of 500 questions generates its own MLflow trace with nested spans (retrieve → generate → 5×judge), making it easy to drill into any individual question in the UI.

---

## Project Structure

```
mbs-moe/
├── rag/                          # Prototype RAG pipeline (aiohttp-based)
│   ├── pipeline.py               # Core retrieve + generate flow
│   ├── evaluation.py             # Semantic similarity + LLM judge scorer
│   └── client.py                 # HTTP client for LLM endpoints
│
├── rag_eval/                     # Production evaluation pipeline (uv project)
│   ├── rag_query.py              # Async retrieve + generate with MLflow spans
│   ├── vectorstore.py            # ChromaDB ingestion with dedup logic
│   ├── deepeval_runner.py        # DeepEval metrics + concurrent evaluation
│   ├── mlflow_tracking.py        # MLflow run/span/metric helpers
│   ├── rag_eval_pipeline.py      # CLI entrypoint (ingest / ask / eval / run-all)
│   ├── settings.py               # Config from .env
│   ├── pyproject.toml            # uv project definition
│   └── .env                      # Endpoint config (not committed)
│
├── data/                         # PDF documents + benchmark CSV
│   └── VDE_FNN_Benchmark.csv     # 500 Q&A pairs with difficulty levels
│
└── inference_test_server/        # Endpoint connectivity tests
```

---

## Evaluation Metrics

| Metric | What it measures | Score range |
|---|---|---|
| **GEval (RAG Correctness)** | Factual match between predicted and expected answer, using a German-language rubric | 0.0 – 1.0 |
| **Contextual Recall** | What fraction of the expected answer is supported by retrieved chunks | 0.0 – 1.0 |
| **Contextual Precision** | What fraction of retrieved chunks are actually relevant to the question | 0.0 – 1.0 |
| **Faithfulness** | Whether the generated answer is grounded in the retrieved context (hallucination check) | 0.0 – 1.0 |
| **Answer Relevancy** | Whether the answer directly addresses the input question | 0.0 – 1.0 |

Results are aggregated globally and broken down by **difficulty level** (Einfach / Mittel / Schwer) and **document section**.

---

## Setup & Running

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Access to an OpenAI-compatible LLM inference server

### 1. Clone and configure

```bash
git clone https://github.com/<your-username>/mbs-moe.git
cd mbs-moe/rag_eval
cp .env.example .env
```

Edit `.env` with your LLM endpoint:

```env
EMBEDDING_BASE_URL=https://your-server/v1
EMBEDDING_API_KEY=your-key
EMBEDDING_MODEL=your-embedding-model

CHAT_BASE_URL=https://your-server/v1
CHAT_API_KEY=your-key
CHAT_MODEL=your-chat-model
```

### 2. Install dependencies

```bash
cd rag_eval
uv sync
```

### 3. Start MLflow tracking server (keep running in a separate terminal)

```bash
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) to view the experiment UI.

### 4. Ingest documents

```bash
uv run python rag_eval_pipeline.py ingest
```

Chunks all PDFs in `../data/`, embeds them, and stores in ChromaDB. Re-running skips already-embedded chunks automatically.

### 5. Ask a question (smoke test)

```bash
uv run python rag_eval_pipeline.py ask "Was ist der Zweck der VDE FNN Norm?" --show-contexts
```

### 6. Run evaluation

Quick test (first 10 questions):
```bash
uv run python rag_eval_pipeline.py eval --limit 10
```

Full benchmark (all 500 questions):
```bash
uv run python rag_eval_pipeline.py eval
```

Full pipeline (ingest + eval in one shot):
```bash
uv run python rag_eval_pipeline.py run-all
```

### 7. View results

- **MLflow UI**: [http://127.0.0.1:5000](http://127.0.0.1:5000) → Experiments → Traces (500 individual question traces)
- **JSON output**: `rag_eval/deepeval_results.json` — per-question scores + summary by difficulty

---

## Results Format

`deepeval_results.json` structure:
```json
{
  "summary": {
    "total": 500,
    "failed": 2,
    "avg_scores": {
      "RAG Correctness (GEval)": 0.73,
      "Contextual Recall": 0.81,
      "Contextual Precision": 0.62,
      "Faithfulness": 0.94,
      "Answer Relevancy": 0.78
    },
    "by_difficulty": {
      "Einfach": { "count": 180, "avg_scores": { ... } },
      "Mittel":  { "count": 210, "avg_scores": { ... } },
      "Schwer":  { "count": 110, "avg_scores": { ... } }
    }
  },
  "results": [
    {
      "id": "1",
      "question": "...",
      "expected": "...",
      "predicted": "...",
      "scores": { "Faithfulness": 1.0, "Answer Relevancy": 0.8, ... },
      "reasons": { "Faithfulness": "The answer is fully supported by ...", ... },
      "difficulty": "Einfach",
      "retrieval_time": 1.69,
      "generation_time": 1.92
    }
  ]
}
```

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_BASE_URL` | — | OpenAI-compatible embedding endpoint |
| `CHAT_BASE_URL` | — | OpenAI-compatible chat endpoint |
| `JUDGE_BASE_URL` | falls back to `CHAT_*` | Separate judge LLM (optional) |
| `TOP_K` | `5` | Number of chunks to retrieve per question |
| `CHUNK_SIZE` | `900` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between consecutive chunks |
| `EVAL_CONCURRENCY` | `5` | Parallel questions evaluated simultaneously |
| `REQUEST_TIMEOUT` | `300` | LLM request timeout in seconds |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | MLflow backend |
