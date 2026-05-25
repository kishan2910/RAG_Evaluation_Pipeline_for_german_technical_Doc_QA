"""Script to load tech-qa dataset into ChromaDB."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.dataset_loader import DatasetLoaderService
from app.services.vector_db import VectorDBService


def load_and_store_dataset(split: str = "train", batch_size: int = 100):
    """Load dataset and store in vector database.

    Args:
        split: Dataset split to load
        batch_size: Batch size for processing
    """
    # Initialize services
    dataset_loader = DatasetLoaderService()
    vector_db = VectorDBService()

    # Load dataset
    dataset = dataset_loader.load_tech_qa_dataset(split=split)

    # Prepare documents
    documents, metadatas, ids = dataset_loader.prepare_documents(dataset)

    print(f"\nPrepared {len(documents)} documents for vectorization")

    # Store in batches
    for batch_idx, batch_docs in enumerate(dataset_loader.batch_documents(documents, batch_size)):
        start_idx = batch_idx * batch_size
        batch_metadatas = metadatas[start_idx : start_idx + len(batch_docs)]
        batch_ids = ids[start_idx : start_idx + len(batch_docs)]

        vector_db.add_documents(batch_docs, batch_metadatas, batch_ids)
        print(f"Processed batch {batch_idx + 1}")

    # Print final stats
    info = vector_db.get_collection_info()
    print(f"\n✅ Successfully loaded dataset!")
    print(f"Collection: {info['collection_name']}")
    print(f"Total documents: {info['document_count']}")


if __name__ == "__main__":
    load_and_store_dataset()
