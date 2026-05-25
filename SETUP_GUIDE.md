# IT Support Enterprise Helpdesk RAG System - Setup & Usage Guide

## 📋 Project Structure

```
app/
├── __init__.py
├── core/
│   ├── __init__.py
│   └── config.py              # Configuration settings
├── services/
│   ├── __init__.py
│   ├── dataset_loader.py      # HuggingFace dataset loading
│   ├── vector_db.py           # ChromaDB vector database
│   ├── llm.py                 # Ollama LLM integration
│   └── rag.py                 # RAG pipeline
└── main.py                    # FastAPI application

scripts/
├── load_dataset.py            # Load dataset into ChromaDB
└── test_rag.py                # Test the RAG pipeline

docker-compose.yml             # Docker services (Ollama, ChromaDB, MLflow)
pyproject.toml                 # Project dependencies
.env                          # Environment variables (create this)
```

## 🚀 Quick Start

### 1. Start Docker Services

```bash
docker-compose up -d
```

This starts:
- **Ollama** (http://localhost:11434) - LLM engine
- **ChromaDB** (http://localhost:8000) - Vector database
- **MLflow** (http://localhost:5000) - Experiment tracking

Verify services:
```bash
# Check Ollama
curl http://localhost:11434/api/tags

# Check ChromaDB
curl http://localhost:8000/api/v1/heartbeat
```

### 2. Load Dependencies

```bash
uv sync
```

### 3. Load Dataset

```bash
python scripts/load_dataset.py
```

This will:
- Download `rojagtap/tech-qa` dataset from HuggingFace
- Create embeddings using sentence-transformers
- Store documents in ChromaDB

### 4. Run RAG Application

```bash
uvicorn app.main:app --host 0.0.0.0 --port 6000 --reload
```

Or use the test script:
```bash
python scripts/test_rag.py
```

## 🦙 Ollama Model Selection for 4GB GPU

### Recommended Models for 4GB GPU:

| Model | Size | VRAM | Speed | Quality | Recommendation |
|-------|------|------|-------|---------|-----------------|
| **Mistral** | 4.1GB | 3.8GB | ⚡⚡⚡ | ⭐⭐⭐⭐ | ✅ **BEST** |
| **Neural Chat** | 3.8GB | 3.5GB | ⚡⚡⚡ | ⭐⭐⭐⭐ | ✅ Good |
| **Phi** | 2.7GB | 2.5GB | ⚡⚡⚡⚡ | ⭐⭐⭐ | ✅ Fast |
| **Llama2** | 3.8GB | 3.6GB | ⚡⚡ | ⭐⭐⭐⭐ | ⚠️ Slower |
| **OpenHermes** | 7B | >4GB | ❌ | ⭐⭐⭐⭐⭐ | ❌ Too large |

### ✅ Recommended: **Mistral (4.1GB)**
- **Why?** Perfect balance of quality and performance for 4GB GPU
- Excellent for technical Q&A
- Fastest inference time
- Best for IT support scenarios

### 🔧 Alternative: **Phi (2.7GB)**
- If you want more GPU headroom
- Faster inference
- Slightly lower quality but still good for IT support

---

## 📥 Download & Deploy Model to Ollama

### Option 1: Using the API

```bash
# Start your app
uvicorn app.main:app --host 0.0.0.0 --port 6000

# Pull model via HTTP
curl -X POST http://localhost:6000/pull-model?model=mistral
```

### Option 2: Direct CLI

```bash
# Pull directly via Ollama CLI
ollama pull mistral

# Verify it's downloaded
ollama list
```

### Option 3: Python Script

```python
from app.services.llm import LLMService

llm = LLMService()
success = llm.pull_model("mistral")  # This will download and show progress
```

**First download will take 10-30 minutes depending on internet speed**

---

## 🔗 API Usage

### Check Health
```bash
curl http://localhost:6000/health
```

### Get System Stats
```bash
curl http://localhost:6000/stats
```

### Query the RAG System
```bash
curl -X POST http://localhost:6000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I fix a slow computer?",
    "n_results": 5
  }'
```

### Response Format
```json
{
  "query": "How do I fix a slow computer?",
  "retrieved_documents": [
    "Question: Why is my computer slow?\nAnswer: ...",
    "Question: How to optimize Windows?\nAnswer: ..."
  ],
  "answer": "To fix a slow computer, try these steps: ..."
}
```

---

## 🐍 Python Usage

### Load Dataset
```python
from app.services.dataset_loader import DatasetLoaderService

loader = DatasetLoaderService()
dataset = loader.load_tech_qa_dataset()
```

### Query with RAG
```python
from app.services.rag import RAGPipeline

rag = RAGPipeline(model="mistral")

result = rag.answer(
    "How do I reset my password?",
    n_results=5
)

print(f"Answer: {result['answer']}")
```

### Direct LLM Query
```python
from app.services.llm import LLMService

llm = LLMService()
response = llm.generate("What is a RAG system?")
print(response)
```

---

## ⚙️ Configuration (.env)

Create `.env` file:

```env
# Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=mistral

# ChromaDB
CHROMADB_HOST=localhost
CHROMADB_PORT=8000

# Embeddings
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Dataset
HUGGINGFACE_DATASET=rojagtap/tech-qa
```

---

## 📊 Workflow: End-to-End

```
1. Start Docker Services
   ↓
2. Download LLM Model (mistral)
   ↓
3. Load HuggingFace Dataset → ChromaDB
   ↓
4. User Query
   ↓
5. Retrieve Similar Docs from ChromaDB
   ↓
6. Generate Answer using Mistral + Context
   ↓
7. Return Answer to User
```

---

## 🧪 Testing

### Test Dataset Loading
```bash
python -c "
from app.services.dataset_loader import DatasetLoaderService
loader = DatasetLoaderService()
dataset = loader.load_tech_qa_dataset(split='train')
print(f'Dataset samples: {len(dataset)}')
print(f'First sample keys: {dataset[0].keys()}')
"
```

### Test Vector DB
```bash
python -c "
from app.services.vector_db import VectorDBService
db = VectorDBService()
info = db.get_collection_info()
print(f'Documents in DB: {info[\"document_count\"]}')
"
```

### Test LLM Connection
```bash
python -c "
from app.services.llm import LLMService
llm = LLMService()
print(f'Connected: {llm.check_ollama_connection()}')
models = llm.list_available_models()
print(f'Available models: {[m.get(\"name\") for m in models]}')
"
```

---

## 🐛 Troubleshooting

### Ollama not running
```bash
docker ps | grep ollama
# If not running:
docker-compose up -d ollama
```

### ChromaDB connection error
- Ensure ChromaDB is running: `docker-compose up -d chromadb`
- Check port 8000: `curl http://localhost:8000/api/v1/heartbeat`

### Model download stuck
- Check internet connection
- Try smaller model: `ollama pull phi` (2.7GB)
- Or pull manually: `ollama pull mistral`

### Out of memory (4GB GPU)
- Close other applications
- Use smaller model (phi instead of mistral)
- Reduce batch size in `load_dataset.py`

### Dataset loading slow
- It's normal for first load (~5-10 minutes)
- Subsequent loads use cache
- Check internet: `ping huggingface.co`

---

## 🚢 Deployment

### Docker Deployment
```bash
docker-compose up -d

# Run migrations/setup
docker-compose exec -T embeddings python scripts/load_dataset.py
```

### Production Tips
- Use environment variables for all settings
- Add authentication to FastAPI endpoints
- Enable CORS for frontend integration
- Monitor GPU/CPU usage
- Set up logging with MLflow

---

## 📚 Resources

- [Ollama Documentation](https://github.com/ollama/ollama)
- [ChromaDB Documentation](https://docs.trychroma.com)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets)
- [Sentence Transformers](https://www.sbert.net)
- [Tech-QA Dataset](https://huggingface.co/datasets/rojagtap/tech-qa)

