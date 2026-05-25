"""RAG (Retrieval-Augmented Generation) pipeline."""

import logging
import time
from app.services.vector_db import VectorDBService
from app.services.llm import LLMService
from app.evaluation.evaluator import EvaluationFramework
from app.core.logger import logger

# Create module logger
rag_logger = logging.getLogger(__name__)


class RAGPipeline:
    """Complete RAG pipeline combining retrieval and generation."""

    def __init__(self, collection_name: str = "tech_qa", model: str | None = None):
        """Initialize the RAG pipeline.

        Args:
            collection_name: Name of the ChromaDB collection
            model: LLM model to use

        Raises:
            RuntimeError: If initialization fails
        """
        try:
            self.vector_db = VectorDBService(collection_name)
            self.llm = LLMService(model)
            self.evaluator = EvaluationFramework()
            rag_logger.info("RAG pipeline initialized successfully")
        except Exception as e:
            error_msg = f"Failed to initialize RAG pipeline: {str(e)}"
            rag_logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def retrieve(self, query: str, n_results: int = 5) -> dict:
        """Retrieve relevant documents for a query.

        Args:
            query: User query
            n_results: Number of documents to retrieve

        Returns:
            Dictionary with documents, scores, and retrieval time

        Raises:
            ValueError: If query is empty
            RuntimeError: If retrieval fails
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")

        try:
            start_time = time.time()

            results = self.vector_db.search(query, n_results)

            retrieval_time = (time.time() - start_time) * 1000

            rag_logger.debug(f"Retrieved {len(results.get('documents', []))} documents in {retrieval_time:.2f}ms")

            return {
                "documents": results.get("documents", []),
                "scores": results.get("scores", []),
                "retrieval_time_ms": retrieval_time
            }

        except Exception as e:
            error_msg = f"Retrieval failed: {str(e)}"
            rag_logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def answer(
        self,
        query: str,
        n_results: int = 5,
        system_prompt: str | None = None
    ) -> dict:
        """Answer a question using RAG.

        Args:
            query: User query
            n_results: Number of documents to retrieve
            system_prompt: Custom system prompt

        Returns:
            Dictionary with query, retrieved docs, answer, and metrics

        Raises:
            ValueError: If query is empty
            RuntimeError: If RAG pipeline fails
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")

        try:
            # Retrieve relevant documents
            retrieval_result = self.retrieve(query, n_results)
            retrieved_docs = retrieval_result["documents"]
            retrieval_scores = retrieval_result["scores"]
            retrieval_time = retrieval_result["retrieval_time_ms"]

            # Combine retrieved documents as context
            context = "\n\n".join(retrieved_docs)

            # Generate answer using LLM
            answer_text, generation_time = self.llm.generate_with_context(
                query,
                context,
                system_prompt
            )

            # Evaluate the response
            evaluation = self.evaluator.evaluate_response(
                query=query,
                context=context,
                answer=answer_text,
                retrieval_scores=retrieval_scores,
                generation_time_ms=generation_time,
                retrieval_time_ms=retrieval_time
            )

            result = {
                "query": query,
                "retrieved_documents": retrieved_docs,
                "answer": answer_text,
                "retrieval_scores": retrieval_scores,
                "retrieval_time_ms": retrieval_time,
                "generation_time_ms": generation_time,
                "total_time_ms": retrieval_time + generation_time,
                "evaluation_metrics": evaluation
            }

            rag_logger.info(f"Processed query in {retrieval_time + generation_time:.2f}ms")

            return result

        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            rag_logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def get_stats(self) -> dict:
        """Get statistics about the RAG pipeline.

        Returns:
            Dictionary with system statistics

        Raises:
            RuntimeError: If unable to retrieve statistics
        """
        try:
            db_info = self.vector_db.get_collection_info()
            models = self.llm.list_available_models()

            # Get evaluation metrics
            eval_metrics = self.evaluator.get_aggregate_metrics()

            stats = {
                "vector_db": db_info,
                "available_models": [m.get("name") for m in models],
                "current_model": self.llm.model,
                "ollama_running": self.llm.check_ollama_connection(),
                "evaluation_metrics": eval_metrics
            }

            rag_logger.debug("Retrieved system statistics")

            return stats

        except Exception as e:
            error_msg = f"Failed to get statistics: {str(e)}"
            rag_logger.error(error_msg)
            raise RuntimeError(error_msg) from e
