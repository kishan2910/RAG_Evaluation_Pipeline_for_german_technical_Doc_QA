"""Test script for RAG pipeline."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.rag import RAGPipeline


def main():
    """Test the RAG pipeline."""
    print("Initializing RAG Pipeline...\n")

    # Initialize pipeline
    rag = RAGPipeline()

    # Check system status
    stats = rag.get_stats()
    print("System Status:")
    print(f"  Ollama Running: {stats['ollama_running']}")
    print(f"  Current Model: {stats['current_model']}")
    print(f"  Vector DB Documents: {stats['vector_db']['document_count']}")
    print(f"  Available Models: {stats['available_models']}\n")

    # Test queries
    test_queries = [
        "How do I fix a slow computer?",
        "What should I do if my network is down?",
        "How to reset my password?",
    ]

    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print("=" * 70)

        try:
            result = rag.answer(query, n_results=3)

            print("\nRetrieved Documents:")
            for i, doc in enumerate(result["retrieved_documents"], 1):
                print(f"\n[Document {i}]")
                print(doc[:200] + "..." if len(doc) > 200 else doc)

            print(f"\nAnswer:\n{result['answer']}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
