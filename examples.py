"""Quick start examples for the RAG system."""

# Example 1: Load dataset
# ========================
from app.services.dataset_loader import DatasetLoaderService
from app.services.vector_db import VectorDBService

dataset_loader = DatasetLoaderService()
dataset = dataset_loader.load_tech_qa_dataset(split="train")
documents, metadatas, ids = dataset_loader.prepare_documents(dataset)

vector_db = VectorDBService()
vector_db.add_documents(documents[:100], metadatas[:100], ids[:100])  # First 100 docs

print("✅ Dataset loaded into ChromaDB")


# Example 2: Query the RAG system
# ================================
from app.services.rag import RAGPipeline

rag = RAGPipeline(model="mistral")

# Ask a question
result = rag.answer("How do I fix a slow computer?", n_results=3)

print(f"\nQuestion: {result['query']}")
print(f"\nRetrieved Documents:")
for i, doc in enumerate(result['retrieved_documents'], 1):
    print(f"  {i}. {doc[:100]}...")

print(f"\nAnswer:\n{result['answer']}")


# Example 3: Direct LLM usage
# ============================
from app.services.llm import LLMService

llm = LLMService()

# Check connection
is_connected = llm.check_ollama_connection()
print(f"Ollama Connected: {is_connected}")

# List available models
models = llm.list_available_models()
print(f"Available Models: {[m.get('name') for m in models]}")

# Generate text
response = llm.generate("What is an IT help desk?")
print(f"Response:\n{response}")


# Example 4: Search without generation
# ======================================
from app.services.vector_db import VectorDBService

vector_db = VectorDBService()

# Search for similar documents
results = vector_db.search("network troubleshooting", n_results=3)
print("Top 3 Similar Documents:")
for doc in results['documents'][0]:
    print(f"  - {doc[:80]}...")


# Example 5: Advanced RAG with custom system prompt
# ==================================================
from app.services.rag import RAGPipeline

rag = RAGPipeline()

custom_system = """You are an expert IT support technician with 10 years of experience.
Provide clear, step-by-step solutions that even non-technical users can follow.
If you don't know the answer, recommend escalating to senior support."""

result = rag.answer(
    "My printer won't print",
    n_results=5,
    system_prompt=custom_system
)

print(result['answer'])
