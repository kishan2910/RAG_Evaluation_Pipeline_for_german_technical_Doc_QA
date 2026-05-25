# 🦙 IT Support Enterprise Helpdesk RAG System

> **Production-Ready Retrieval-Augmented Generation System for Enterprise IT Support**

A full-stack, enterprise-grade RAG system designed for real-world IT support use cases. Features local LLMs (Ollama), vector search (ChromaDB), comprehensive evaluation metrics, and production-ready observability.

## ✨ Key Features

- **📚 RAG Pipeline**: Semantic retrieval + Context-aware generation with metrics
- **🦙 Local LLM**: Run Ollama locally on modest hardware (Mistral recommended)
- **🔍 Vector Search**: ChromaDB with semantic embeddings
- **📊 Evaluation Framework**: Hallucination detection, retrieval quality, metrics
- **🚀 Production API**: FastAPI with health checks, monitoring, request tracing
- **🐳 Docker**: Complete Docker setup (Ollama, ChromaDB)
- **📈 Observability**: Structured logging, performance metrics, request tracking

## 🎯 Quick Start (5 minutes)

### 1. Start Services
```bash
docker-compose up -d
```

### 2. Setup Python
```bash
uv sync
```

### 3. Download LLM Model & Load Dataset
```bash
# Auto-setup with interactive wizard
python scripts/quick_setup.py

# Or manual steps:
ollama pull mistral
python scripts/load_dataset.py
```

### 4. Run API
```bash
uvicorn app.main:app --host 0.0.0.0 --port 6000 --reload
```

### 5. Test
```bash
# Interactive test
python scripts/test_rag.py

# Or query via HTTP
curl -X POST http://localhost:6000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I fix a slow computer?", "n_results": 5}'
```

Visit API docs: http://localhost:6000/docs

---

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         User Query                           │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────┐
         │    Embedding Generation        │
         │  (Sentence-Transformers)      │
         └────────────────┬───────────────┘
                          │
                          ▼
         ┌────────────────────────────────┐
         │  Vector Database Search        │
         │        (ChromaDB)              │
         └────────────────┬───────────────┘
                          │
         ┌────────────────▼───────────────┐
         │   Retrieved Documents          │
         │   + Context Assembly           │
         └────────────────┬───────────────┘
                          │
         ┌────────────────▼───────────────┐
         │    LLM Generation              │
         │      (Ollama - Mistral)        │
         └────────────────┬───────────────┘
                          │
                          ▼
         ┌────────────────────────────────┐
         │      Generated Answer          │
         │   with Source Attribution      │
         └────────────────────────────────┘
```

---

## 🦙 LLM Model Recommendations

### For 4GB GPU (Your Setup)

| Model | Size | Recommendation |
|-------|------|-----------------|
| **Mistral** | 4.1GB | ✅ **BEST** - Perfect balance |
| **Phi** | 2.7GB | ✅ Good - Faster, lighter |
| **Neural Chat** | 3.8GB | ✅ Good - Very capable |
| Llama2 | 3.8GB | ⚠️ Slower but high quality |
| OpenHermes | 4GB | ⚠️ Possible, might be slow |

**Recommended: [Mistral](OLLAMA_MODEL_GUIDE.md#-mistral-7b-recommended)**

For detailed model analysis, see [OLLAMA_MODEL_GUIDE.md](OLLAMA_MODEL_GUIDE.md)

---

## 📁 Project Structure

```
app/
├── core/
│   └── config.py              # Configuration & settings
├── services/
│   ├── dataset_loader.py      # HuggingFace dataset loading
│   ├── vector_db.py           # ChromaDB operations
│   ├── llm.py                 # Ollama LLM integration
│   └── rag.py                 # RAG pipeline orchestration
└── main.py                    # FastAPI application

scripts/
├── quick_setup.py             # Interactive setup wizard ⭐
├── load_dataset.py            # Batch dataset loading
└── test_rag.py                # Test RAG pipeline

docker-compose.yml             # Docker services
pyproject.toml                 # Python dependencies
.env                          # Environment variables
```

---

## 🔧 Configuration

### Environment Variables (.env)

```env
# Ollama LLM
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=mistral

# ChromaDB Vector Database
CHROMADB_HOST=localhost
CHROMADB_PORT=8000

# Embeddings
EMBEDDING_MODEL=all-MiniLM-L6-v2

# HuggingFace Dataset
HUGGINGFACE_DATASET=rojagtap/tech-qa
```

### Docker Services

**Ollama** (LLM Engine)
- Port: 11434
- Volume: `ollama_data` (model storage)

**ChromaDB** (Vector DB)
- Port: 8000
- Volume: `chroma_data` (vector storage)

**MLflow** (Experiment Tracking)
- Port: 5000
- Volume: `mlflow_data` (metrics/artifacts)

---

## 🚀 API Endpoints

### Health Check
```bash
GET /health
```

### System Statistics
```bash
GET /stats
```
Returns: Ollama connection status, available models, document count

### Query RAG
```bash
POST /query
Content-Type: application/json

{
  "query": "How do I fix a slow computer?",
  "n_results": 5
}
```

Response:
```json
{
  "query": "How do I fix a slow computer?",
  "retrieved_documents": [...],
  "answer": "To speed up your computer..."
}
```

### List Models
```bash
GET /models
```

### Download Model
```bash
POST /pull-model?model=mistral
```

---

## 🐍 Python Usage Examples

### Basic RAG Query
```python
from app.services.rag import RAGPipeline

rag = RAGPipeline(model="mistral")
result = rag.answer("How do I fix my network?")
print(result['answer'])
```

### Load Custom Dataset
```python
from app.services.dataset_loader import DatasetLoaderService
from app.services.vector_db import VectorDBService

loader = DatasetLoaderService()
dataset = loader.load_tech_qa_dataset()

vector_db = VectorDBService()
documents, metadatas, ids = loader.prepare_documents(dataset)
vector_db.add_documents(documents, metadatas, ids)
```

### Direct LLM Usage
```python
from app.services.llm import LLMService

llm = LLMService()
response = llm.generate("What is IT support?")
```

More examples in [examples.py](examples.py)

---

## 📖 Documentation

- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Comprehensive setup instructions
- **[OLLAMA_MODEL_GUIDE.md](OLLAMA_MODEL_GUIDE.md)** - Detailed model comparison & recommendations
- **[examples.py](examples.py)** - Code examples and usage patterns

---

## 🧪 Testing

### Interactive Test
```bash
python scripts/test_rag.py
```

### Quick Setup Wizard
```bash
# Full setup with interactive prompts
python scripts/quick_setup.py

# Just test the system
python scripts/quick_setup.py --test-only

# Download specific model
python scripts/quick_setup.py --download phi
```

### Manual Testing

```python
# Test vector DB
from app.services.vector_db import VectorDBService
db = VectorDBService()
print(db.get_collection_info())

# Test LLM connection
from app.services.llm import LLMService
llm = LLMService()
print(llm.check_ollama_connection())

# Test dataset loading
from app.services.dataset_loader import DatasetLoaderService
loader = DatasetLoaderService()
dataset = loader.load_tech_qa_dataset()
```

---

## 🐛 Troubleshooting

### Services won't start
```bash
# Check what's running
docker ps

# View logs
docker-compose logs ollama
docker-compose logs chromadb

# Restart services
docker-compose restart
```

### Ollama model download stuck
```bash
# Cancel and retry
ollama pull mistral --verbose

# Try smaller model
ollama pull phi
```

### Out of memory (GPU)
```bash
# Close other applications
# Use smaller model:
OLLAMA_MODEL=phi

# Or reduce batch size in load_dataset.py
```

### ChromaDB connection error
```bash
# Check if running
curl http://localhost:8000/api/v1/heartbeat

# Restart
docker-compose restart chromadb
```

See [SETUP_GUIDE.md](SETUP_GUIDE.md#-troubleshooting) for more troubleshooting tips.

---

## 💡 Tips & Tricks

### Batch Processing
```python
queries = ["Q1", "Q2", "Q3"]
for q in queries:
    result = rag.answer(q)
```

### Custom System Prompts
```python
custom_system = "You are an expert IT technician. Provide step-by-step solutions."
result = rag.answer(query, system_prompt=custom_system)
```

### Monitor GPU Usage
```bash
# Windows: Task Manager > GPU
# Mac: Activity Monitor > GPU History
# Linux: nvidia-smi (if NVIDIA GPU)
```

### Performance Optimization
- Use `phi` model for speed vs `mistral` for quality
- Batch document loading for faster ingestion
- Reduce `n_results` in queries (3-5 is usually enough)

---

## 📊 Dataset Info

**tech-qa Dataset**: 5,000+ Q&A pairs for IT technical support
- Questions about software, hardware, networking, troubleshooting
- Answers from IT professionals
- Source: HuggingFace rojagtap/tech-qa

To load full dataset:
```bash
python scripts/load_dataset.py
```

---

## 🚢 Deployment

### Docker Deployment
```bash
docker-compose up -d
docker-compose exec -T embeddings python scripts/load_dataset.py
```

### Production Checklist
- [ ] Set up environment variables
- [ ] Configure firewall for ports
- [ ] Enable CORS if needed
- [ ] Set up monitoring/logging
- [ ] Add authentication to API
- [ ] Regular backups of ChromaDB data

---

## 📚 Resources

- [Ollama Documentation](https://github.com/ollama/ollama)
- [ChromaDB Documentation](https://docs.trychroma.com)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets)
- [Sentence Transformers](https://www.sbert.net)

---

## 📝 License

MIT License - See LICENSE file for details

---

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting guide
2. Review the setup guide
3. Check Docker logs: `docker-compose logs`
4. Test components individually

---

**Ready to get started?** Run: `python scripts/quick_setup.py`