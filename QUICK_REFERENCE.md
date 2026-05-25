# 🚀 Quick Reference Guide

## Start Here

```bash
# 1. Start Docker services (Ollama, ChromaDB, MLflow)
docker-compose up -d

# 2. Run interactive setup (downloads model & loads dataset)
python scripts/quick_setup.py

# 3. Start the API server
uvicorn app.main:app --host 0.0.0.0 --port 6000 --reload

# 4. Open API docs
# Visit: http://localhost:6000/docs
```

---

## 📚 What Each File Does

### Core Application
- **app/main.py** - FastAPI REST API server
- **app/services/rag.py** - RAG pipeline (retrieval + generation)
- **app/services/vector_db.py** - ChromaDB operations
- **app/services/llm.py** - Ollama LLM integration
- **app/services/dataset_loader.py** - HuggingFace dataset loading

### Scripts
- **scripts/quick_setup.py** - ⭐ **START HERE** - Interactive setup wizard
- **scripts/load_dataset.py** - Load tech-qa dataset into ChromaDB
- **scripts/test_rag.py** - Test the RAG system with sample queries

### Documentation
- **SETUP_GUIDE.md** - Complete setup & configuration guide
- **OLLAMA_MODEL_GUIDE.md** - Model selection for 4GB GPU
- **README.md** - Project overview & quick start
- **.env** - Environment variables template

---

## 🦙 Recommended Model for 4GB GPU

### **Mistral 7B** ✅ BEST CHOICE
```
Size: 4.1 GB
Speed: Fast (50-100 tokens/sec)
Quality: Excellent (⭐⭐⭐⭐)
Download: ollama pull mistral
```

### Alternatives:
- **Phi 2.7B** - Faster, lighter (2.7 GB) ✅
- **Neural Chat 7B** - Very capable (3.8 GB) ✅

Avoid:
- ❌ Llama2 13B (too large)
- ❌ Mistral 8x7B (26 GB - way too large)

---

## 📋 Setup Checklist

- [ ] Docker installed
- [ ] Docker services running (`docker ps`)
- [ ] Python 3.10+ installed
- [ ] `uv sync` completed
- [ ] Model downloaded (`ollama pull mistral`)
- [ ] Dataset loaded (`python scripts/load_dataset.py`)
- [ ] API running on http://localhost:6000
- [ ] Test works (`python scripts/test_rag.py`)

---

## 🧪 Quick Tests

### Test Ollama Connection
```bash
curl http://localhost:11434/api/tags
```

### Test ChromaDB
```bash
curl http://localhost:8000/api/v1/heartbeat
```

### Test RAG API
```bash
curl -X POST http://localhost:6000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I fix a slow computer?", "n_results": 5}'
```

### Test in Python
```bash
python scripts/test_rag.py
```

---

## 🎯 Common Tasks

### Download a Different Model
```bash
ollama pull phi
# Then update .env: OLLAMA_MODEL=phi
```

### Check Available Models
```bash
ollama list
```

### Reload Dataset
```bash
# Clear old data
python -c "from app.services.vector_db import VectorDBService; VectorDBService().clear_collection()"

# Load new data
python scripts/load_dataset.py
```

### Monitor GPU Usage
- **Windows**: Task Manager → Performance → GPU
- **Mac**: Activity Monitor → GPU History
- **Linux**: `nvidia-smi` (NVIDIA GPU)

---

## 🚨 Troubleshooting

| Problem | Solution |
|---------|----------|
| Ollama not running | `docker-compose up -d ollama` |
| ChromaDB not running | `docker-compose up -d chromadb` |
| Out of memory | Close other apps, use `phi` model |
| Model download slow | Normal (10-30 min first time) |
| Port already in use | Change in .env or docker-compose |
| API not responding | Check logs: `docker-compose logs` |

---

## 📊 Performance Tips

1. **For Speed**: Use Phi (2.7GB) instead of Mistral
2. **For Quality**: Use Mistral (4.1GB)
3. **For Accuracy**: Reduce `n_results` to 3-5 (best results)
4. **For Efficiency**: Batch queries together
5. **For Reliability**: Keep GPU headroom (close other apps)

---

## 🔗 Important URLs

| Service | URL |
|---------|-----|
| API Docs | http://localhost:6000/docs |
| API Health | http://localhost:6000/health |
| ChromaDB | http://localhost:8000 |
| MLflow | http://localhost:5000 |
| Ollama | http://localhost:11434 |

---

## 📝 Example Queries

```bash
# Basic query
curl -X POST http://localhost:6000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I fix a network error?", "n_results": 5}'

# With fewer results
curl -X POST http://localhost:6000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How to reset password?", "n_results": 3}'

# Using Python
python -c "
from app.services.rag import RAGPipeline
rag = RAGPipeline()
result = rag.answer('How to fix a slow PC?')
print(result['answer'])
"
```

---

## 📚 Documentation Files

1. **README.md** - Start here for overview
2. **SETUP_GUIDE.md** - Detailed setup instructions
3. **OLLAMA_MODEL_GUIDE.md** - Model selection & optimization
4. **examples.py** - Code examples
5. **pyproject.toml** - Dependencies list

---

## ⚙️ Key Commands

```bash
# Start everything
docker-compose up -d && uv sync && python scripts/quick_setup.py

# Run API
uvicorn app.main:app --host 0.0.0.0 --port 6000

# Test RAG
python scripts/test_rag.py

# Load dataset
python scripts/load_dataset.py

# Check services
docker-compose ps

# View logs
docker-compose logs -f ollama
```

---

## 🎓 How It Works (Simple Version)

```
User asks a question
        ↓
Search vector database for similar documents
        ↓
Get top N most relevant documents
        ↓
Combine documents as context
        ↓
Send to Ollama (Mistral model)
        ↓
Model generates answer using context
        ↓
Return answer to user
```

---

## 🚀 Next Steps

1. Run: `python scripts/quick_setup.py`
2. Wait for model download (10-30 minutes)
3. Wait for dataset loading (5-10 minutes)
4. Start API: `uvicorn app.main:app --host 0.0.0.0 --port 6000`
5. Test: Visit http://localhost:6000/docs
6. Try queries!

---

**Questions?** Check the comprehensive guides:
- Full setup: [SETUP_GUIDE.md](SETUP_GUIDE.md)
- Model help: [OLLAMA_MODEL_GUIDE.md](OLLAMA_MODEL_GUIDE.md)
- Code examples: [examples.py](examples.py)

---

**TL;DR**: `docker-compose up -d && python scripts/quick_setup.py && uvicorn app.main:app --host 0.0.0.0 --port 6000`
