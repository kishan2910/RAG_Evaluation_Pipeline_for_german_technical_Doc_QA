"""Vector database service for managing embeddings and retrieval."""

import logging
import chromadb
from sentence_transformers import SentenceTransformer
from app.core.config import settings
from app.core.logger import logger

# Create module logger
db_logger = logging.getLogger(__name__)


class VectorDBService:
    """Service for managing ChromaDB vector database operations."""

    def __init__(self, collection_name: str = "tech_qa"):
        """Initialize the vector database service.

        Args:
            collection_name: Name of the collection in ChromaDB

        Raises:
            ConnectionError: If unable to connect to ChromaDB
            RuntimeError: If unable to load embedding model
        """
        try:
            # Initialize ChromaDB client
            self.client = chromadb.HttpClient(
                host=settings.chromadb_host,
                port=settings.chromadb_port
            )
            db_logger.info(f"Connected to ChromaDB at {settings.chromadb_host}:{settings.chromadb_port}")

            # Initialize embedding model
            db_logger.info(f"Loading embedding model: {settings.embedding_model}")
            self.embedding_model = SentenceTransformer(settings.embedding_model)
            db_logger.info("Embedding model loaded successfully")

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            db_logger.info(f"Initialized collection: {collection_name}")

            self.collection_name = collection_name

        except Exception as e:
            error_msg = f"Failed to initialize VectorDBService: {str(e)}"
            db_logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None
    ) -> None:
        """Add documents to the vector database.

        Args:
            documents: List of document texts
            metadatas: List of metadata dicts for each document
            ids: List of unique IDs for each document

        Raises:
            ValueError: If documents list is empty
            RuntimeError: If embedding generation or insertion fails
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")

        try:
            # Generate embeddings
            db_logger.debug(f"Generating embeddings for {len(documents)} documents")
            embeddings = self.embedding_model.encode(
                documents,
                convert_to_list=True
            )

            # Generate IDs if not provided
            if ids is None:
                ids = [f"doc_{i}" for i in range(len(documents))]

            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas or [{} for _ in documents],
                ids=ids
            )

            db_logger.info(f"Added {len(documents)} documents to collection '{self.collection_name}'")

        except Exception as e:
            error_msg = f"Failed to add documents to vector DB: {str(e)}"
            db_logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def search(
        self,
        query: str,
        n_results: int = 5
    ) -> dict:
        """Search for relevant documents.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            Dictionary with results from ChromaDB including scores

        Raises:
            ValueError: If query is empty or n_results is invalid
            RuntimeError: If search fails
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if n_results < 1:
            raise ValueError("n_results must be at least 1")

        try:
            db_logger.debug(f"Searching for: {query}")

            # Generate embedding for query
            query_embedding = self.embedding_model.encode(query, convert_to_list=True)

            # Search collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "distances", "metadatas"]
            )

            # Convert distances to similarity scores (cosine distance to similarity)
            # ChromaDB uses cosine distance, so similarity = 1 - distance
            distances = results.get("distances", [[]])[0]
            scores = [max(0, 1 - d) for d in distances]

            db_logger.debug(f"Retrieved {len(results.get('documents', [[]])[0])} documents")

            return {
                "documents": results.get("documents", [[]])[0],
                "scores": scores,
                "distances": distances,
                "metadatas": results.get("metadatas", [[]])[0]
            }

        except Exception as e:
            error_msg = f"Search failed: {str(e)}"
            db_logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def get_collection_info(self) -> dict:
        """Get information about the collection.

        Returns:
            Dictionary with collection metadata

        Raises:
            RuntimeError: If unable to retrieve collection info
        """
        try:
            count = self.collection.count()
            db_logger.debug(f"Collection '{self.collection_name}' has {count} documents")

            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "embedding_model": settings.embedding_model
            }
        except Exception as e:
            error_msg = f"Failed to get collection info: {str(e)}"
            db_logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def clear_collection(self) -> None:
        """Clear all documents from the collection.

        Raises:
            RuntimeError: If unable to clear collection
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            db_logger.info(f"Cleared collection '{self.collection_name}'")
        except Exception as e:
            error_msg = f"Failed to clear collection: {str(e)}"
            db_logger.error(error_msg)
            raise RuntimeError(error_msg) from e
