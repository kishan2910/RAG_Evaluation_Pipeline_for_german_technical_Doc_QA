# Architecture & Production Design

## System Overview

The IT Support Helpdesk RAG System is designed as a production-grade solution that demonstrates enterprise software engineering practices alongside RAG/LLM technology.

### Core Components

#### 1. **RAG Pipeline** (`app/services/rag.py`)
Orchestrates the three-stage RAG process:
- **Retrieval**: Semantic search using vector embeddings
- **Generation**: Context-aware answer synthesis with LLM
- **Evaluation**: Quality metrics collection and analysis

#### 2. **Vector Database** (`app/services/vector_db.py`)
ChromaDB integration for semantic search:
- Manages document embeddings and storage
- Handles similarity-based retrieval with scores
- Provides collection management and metadata

#### 3. **LLM Service** (`app/services/llm.py`)
Ollama integration for local inference:
- Manages model lifecycle and switching
- Tracks generation latency
- Implements context-aware prompting
- Returns both answer and timing metrics

#### 4. **Evaluation Framework** (`app/evaluation/evaluator.py`)
Quality metrics and benchmarking:
- Retrieval quality metrics (similarity scores, coverage)
- Hallucination detection (keyword overlap analysis)
- Generation quality metrics (latency, coherence)
- Aggregate performance reporting

#### 5. **Monitoring** (`app/monitoring/metrics.py`)
System observability and metrics:
- Per-endpoint performance tracking
- Request latency aggregation
- Success rate calculation
- System uptime tracking

#### 6. **API Layer** (`app/main.py`)
FastAPI application with production patterns:
- Request ID generation for tracing
- Error handling with standardized responses
- Metrics collection middleware
- Health checks and status endpoints

## Data Flow

```
User Query
    ↓
API Request Handler (/query)
    ↓
Request ID Generation + Metrics Start
    ↓
RAG Pipeline.answer()
    ├─→ Retrieval Stage
    │   ├─→ VectorDB.search(query)
    │   │   ├─→ Embedding generation
    │   │   ├─→ Semantic similarity search
    │   │   └─→ Score calculation
    │   └─→ Results: [docs, scores, latency]
    │
    ├─→ Generation Stage
    │   ├─→ Context assembly from docs
    │   ├─→ LLM.generate_with_context()
    │   │   ├─→ System prompt setup
    │   │   ├─→ Ollama inference
    │   │   └─→ Response + latency
    │   └─→ Results: [answer, latency]
    │
    └─→ Evaluation Stage
        ├─→ Compute retrieval metrics
        ├─→ Detect hallucinations
        ├─→ Compute generation metrics
        └─→ Log results
    ↓
Response Building
    ├─→ Metrics Calculation
    │   ├─→ Avg retrieval score
    │   ├─→ Total latency
    │   └─→ Success flag
    ├─→ Response Formatting
    │   ├─→ QueryResponse schema
    │   ├─→ Metadata and timing
    │   └─→ Evaluation metrics
    └─→ Metrics Recording
        ├─→ Endpoint metrics
        ├─→ Latency aggregation
        └─→ Success tracking
    ↓
HTTP Response → Client
```

## API Response Schema

All `/query` responses include:

```json
{
  "request_id": "uuid-for-tracing",
  "query": "user-question",
  "retrieved_documents": ["doc1", "doc2", ...],
  "answer": "generated-answer",
  "retrieval_metrics": {
    "num_retrieved": 5,
    "retrieval_scores": [0.92, 0.87, ...],
    "avg_score": 0.88
  },
  "generation_metrics": {
    "model": "mistral",
    "temperature": 0.7,
    "generation_time_ms": 2345.67
  },
  "total_time_ms": 3456.78,
  "timestamp": "2024-01-15T10:30:00"
}
```

## Error Handling Strategy

### Layered Error Handling

1. **Service Layer** (Vector DB, LLM):
   - Raises typed exceptions (ValueError, RuntimeError)
   - Logs detailed error information
   - Provides recovery suggestions

2. **Pipeline Layer** (RAG):
   - Catches service exceptions
   - Wraps with context information
   - Tracks evaluation metrics even on partial failure

3. **API Layer** (FastAPI):
   - Catches all exceptions
   - Converts to HTTP status codes
   - Returns standardized error responses
   - Records failure metrics

### Standard Error Response

```json
{
  "request_id": "uuid-for-tracing",
  "error_code": "HUMAN_READABLE_CODE",
  "error_message": "User-friendly message",
  "timestamp": "2024-01-15T10:30:00"
}
```

## Evaluation Metrics

### Retrieval Quality

**Similarity Scores** (0-1):
- Cosine similarity between query and documents
- Collected for all retrieved documents
- Average score indicates overall quality

**Coverage**:
- Number of documents retrieved
- Helps assess whether sufficient context was found

### Generation Quality

**Hallucination Detection**:
- Keyword overlap analysis: compare answer words to context words
- Threshold-based classification (default: < 30% overlap = hallucination)
- Flags responses that may not be grounded in retrieved context

**Latency**:
- Generation time tracking
- Indicates model performance and system load

### System Performance

**Request Latency**:
- Total end-to-end time (retrieval + generation + overhead)
- Tracked per request and aggregated

**Success Rate**:
- Percentage of successful vs failed queries
- Indicates system reliability

## Monitoring & Observability

### Logging

**Levels**:
- DEBUG: Detailed information (embedding generation, search results)
- INFO: Major operations (initialization, successful queries)
- ERROR: Failures and exceptions

**Output**:
- Console: INFO level (user-facing)
- File: DEBUG level (diagnostics)
- Rotation: 10MB per file, 10 backup files

### Metrics Collection

**Per-Request**:
- Request ID (UUID)
- Endpoint path
- Response latency
- Success flag

**Aggregate**:
- Total requests processed
- Failed requests count
- Average response time
- Success rate
- Uptime

### Health Checks

**`GET /health`** Returns:
- Overall system status
- Ollama connection status
- ChromaDB connectivity
- Vector DB document count
- Current model name

**`GET /stats`** Returns:
- All `/health` data
- Plus: availability metrics, request counts, avg latency
- Evaluation framework statistics

## Performance Optimization

### Retrieval Optimization
- Sentence transformer model: all-MiniLM-L6-v2 (33M params)
- Cosine distance for similarity
- Configurable number of results (1-50)

### Generation Optimization
- Local inference (no API calls)
- Temperature tuning for quality/speed trade-off
- System prompt engineering for focused answers

### System Optimization
- Connection pooling to ChromaDB
- Embedding caching (via sentence-transformers)
- Request ID generation for distributed tracing

## Scalability Considerations

### Current Limitations
- Single process API (design for single machine)
- In-memory metrics (reset on restart)
- No distributed caching

### Scaling Strategies
- Horizontal: FastAPI workers + load balancer
- Vertical: GPU optimization, batch processing
- Caching: Add Redis for embedding cache
- Async: Background job processing for dataset updates

## Security Considerations

### API Security
- No authentication/authorization (add in production)
- CORS enabled for all origins (restrict in production)
- Input validation on query length
- No sensitive data in logs

### Data Security
- Credentials in .env (not committed)
- No PII in embeddings (use on vetted datasets)
- Consider encryption for ChromaDB at rest

### LLM Safety
- System prompts to discourage hallucinations
- Evaluation metrics to detect unreliable outputs
- User feedback collection for monitoring

## Testing Strategy

### Unit Tests
- Service layer (vector_db, llm, rag)
- Evaluation metrics calculations
- Config loading

### Integration Tests
- Full RAG pipeline with real data
- API endpoint testing
- Error handling verification

### Performance Tests
- Latency benchmarking
- Throughput testing
- Memory profiling

---

For deployment and configuration details, see `SETUP_GUIDE.md`.
