#!/usr/bin/env python3
"""
Quick setup and demo script for the RAG system.
Run this script to initialize everything and test the system.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


def print_header(text):
    """Print a formatted header."""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")


def check_services():
    """Check if all required services are running."""
    from app.services.llm import LLMService
    from app.services.vector_db import VectorDBService

    print_header("🔍 Checking Services")

    llm = LLMService()
    print(f"✓ Ollama: ", end="", flush=True)
    if llm.check_ollama_connection():
        print("✅ Running")
    else:
        print("❌ Not running")
        print("  Start Docker: docker-compose up -d")
        return False

    vector_db = VectorDBService()
    print(f"✓ ChromaDB: ✅ Connected")

    return True


def show_available_models():
    """Show available models in Ollama."""
    from app.services.llm import LLMService

    print_header("📦 Available Models in Ollama")

    llm = LLMService()
    models = llm.list_available_models()

    if not models:
        print("ℹ️  No models downloaded yet.\n")
        print("Recommended for 4GB GPU: Mistral")
        print("\nTo download, run:")
        print("  python scripts/quick_setup.py --download mistral")
        return False

    for model in models:
        name = model.get("name", "Unknown")
        size = model.get("size", 0)
        size_gb = size / 1e9
        print(f"  • {name:<30} ({size_gb:.1f} GB)")

    return True


def download_model(model_name="mistral"):
    """Download a model from Ollama registry."""
    from app.services.llm import LLMService

    print_header(f"⬇️  Downloading {model_name} Model")

    llm = LLMService()
    print(f"This may take 10-30 minutes...\n")

    success = llm.pull_model(model_name)

    if success:
        print(f"\n✅ Successfully downloaded {model_name}!")
        return True
    else:
        print(f"❌ Failed to download {model_name}")
        return False


def load_dataset():
    """Load dataset into ChromaDB."""
    from app.services.dataset_loader import DatasetLoaderService
    from app.services.vector_db import VectorDBService

    print_header("📚 Loading Dataset into ChromaDB")

    # Check if already loaded
    vector_db = VectorDBService()
    info = vector_db.get_collection_info()

    if info["document_count"] > 0:
        print(f"✓ ChromaDB already contains {info['document_count']} documents")
        print("Skipping dataset loading...\n")
        return True

    print("Loading tech-qa dataset from HuggingFace...")
    print("(First time may take 2-5 minutes)\n")

    try:
        dataset_loader = DatasetLoaderService()
        dataset = dataset_loader.load_tech_qa_dataset(split="train")

        documents, metadatas, ids = dataset_loader.prepare_documents(dataset)

        print(f"Adding {len(documents)} documents to ChromaDB...")

        # Load in batches
        batch_size = 50
        for batch_idx, batch_docs in enumerate(
            dataset_loader.batch_documents(documents, batch_size)
        ):
            start_idx = batch_idx * batch_size
            batch_metadatas = metadatas[start_idx : start_idx + len(batch_docs)]
            batch_ids = ids[start_idx : start_idx + len(batch_docs)]

            vector_db.add_documents(batch_docs, batch_metadatas, batch_ids)

            if (batch_idx + 1) % 5 == 0:
                print(f"  Processed {(batch_idx + 1) * batch_size} documents...")

        info = vector_db.get_collection_info()
        print(f"\n✅ Successfully loaded {info['document_count']} documents!")
        return True

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False


def test_rag():
    """Test the RAG system with a sample query."""
    from app.services.rag import RAGPipeline

    print_header("🧪 Testing RAG System")

    rag = RAGPipeline()

    # Check stats
    stats = rag.get_stats()
    print(f"System Status:")
    print(f"  • Ollama: {'✅ Connected' if stats['ollama_running'] else '❌ Not connected'}")
    print(f"  • Model: {stats['current_model']}")
    print(f"  • Documents in DB: {stats['vector_db']['document_count']}")

    if stats["vector_db"]["document_count"] == 0:
        print("\n⚠️  No documents in database. Load dataset first!")
        return False

    print("\n" + "-" * 70)
    print("Running test query...")
    print("-" * 70 + "\n")

    query = "How can I speed up my computer?"
    print(f"Query: {query}\n")

    try:
        result = rag.answer(query, n_results=3)

        print("Retrieved Documents:")
        for i, doc in enumerate(result["retrieved_documents"], 1):
            preview = doc.split("\n")[0][:60]
            print(f"  {i}. {preview}...")

        print(f"\nGenerated Answer:")
        print(f"{result['answer']}")

        print("\n✅ RAG system is working!")
        return True

    except Exception as e:
        print(f"❌ Error testing RAG: {e}")
        return False


def show_next_steps():
    """Show next steps for the user."""
    print_header("🚀 Next Steps")

    print("1. Start the API server:")
    print("   uvicorn app.main:app --host 0.0.0.0 --port 6000 --reload\n")

    print("2. Access the API:")
    print("   http://localhost:6000/docs  (Interactive API docs)\n")

    print("3. Example query via curl:")
    print('   curl -X POST http://localhost:6000/query \\')
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"query": "How do I fix my network?", "n_results": 5}\'')
    print()


def main():
    """Run the setup wizard."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Quick setup wizard for IT Support RAG System"
    )
    parser.add_argument(
        "--download",
        type=str,
        help="Download a specific model (e.g., mistral, phi)",
    )
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="Skip dataset loading",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only test the RAG system",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  🦙 IT Support Enterprise Helpdesk RAG System - Quick Setup")
    print("=" * 70)

    if args.test_only:
        # Just test
        if check_services():
            test_rag()
        show_next_steps()
        return

    # Check services
    if not check_services():
        return

    # Show/download models
    show_available_models()

    if args.download:
        if not download_model(args.download):
            return
    else:
        # Check if mistral is available
        from app.services.llm import LLMService

        llm = LLMService()
        models = llm.list_available_models()
        model_names = [m.get("name") for m in models]

        if "mistral" not in model_names:
            print("\n⚠️  Mistral not found. Downloading now...")
            print("(This is a one-time setup, may take 10-30 minutes)\n")
            if not download_model("mistral"):
                return

    # Load dataset
    if not args.skip_load:
        if not load_dataset():
            return

    # Test RAG
    test_rag()

    # Show next steps
    show_next_steps()


if __name__ == "__main__":
    main()
