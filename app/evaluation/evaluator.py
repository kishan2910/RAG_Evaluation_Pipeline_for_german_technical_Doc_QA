"""Evaluation metrics for RAG system performance."""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class EvaluationMetrics:
    """Container for RAG evaluation metrics."""

    # Retrieval metrics
    retrieval_score: float  # Average similarity score
    num_retrieved: int
    retrieval_scores: list[float]

    # Generation metrics
    answer: str
    context: str
    query: str

    # Optional evaluation results
    faithfulness_score: Optional[float] = None
    relevance_score: Optional[float] = None
    hallucination_detected: Optional[bool] = None


class RAGEvaluator:
    """Evaluate RAG system performance."""

    @staticmethod
    def compute_retrieval_metrics(
        scores: list[float],
        documents: list[str]
    ) -> dict:
        """Compute retrieval quality metrics.

        Args:
            scores: List of similarity scores
            documents: Retrieved documents

        Returns:
            Dictionary with retrieval metrics
        """
        return {
            "mean_retrieval_score": float(np.mean(scores)) if scores else 0.0,
            "max_retrieval_score": float(np.max(scores)) if scores else 0.0,
            "min_retrieval_score": float(np.min(scores)) if scores else 0.0,
            "num_retrieved": len(documents),
            "scores": [float(s) for s in scores]
        }

    @staticmethod
    def compute_generation_metrics(
        answer: str,
        context: str,
        query: str
    ) -> dict:
        """Compute generation quality metrics.

        Args:
            answer: Generated answer
            context: Retrieved context
            query: Original query

        Returns:
            Dictionary with generation metrics
        """
        metrics = {
            "answer_length": len(answer),
            "answer_words": len(answer.split()),
            "context_length": len(context),
            "query_length": len(query),
        }

        # Check if answer references context
        context_keywords = set()
        for word in context.lower().split():
            if len(word) > 4:  # Only significant words
                context_keywords.add(word)

        answer_words = set(answer.lower().split())
        keyword_overlap = len(context_keywords & answer_words) / max(len(context_keywords), 1)

        metrics["context_reference_ratio"] = keyword_overlap

        return metrics

    @staticmethod
    def detect_potential_hallucination(
        answer: str,
        context: str,
        threshold: float = 0.3
    ) -> bool:
        """Detect if answer might contain hallucinations.

        A simple heuristic: if answer has very few keywords from context,
        it might be hallucinating.

        Args:
            answer: Generated answer
            context: Retrieved context
            threshold: Keyword overlap threshold (0-1)

        Returns:
            True if hallucination is likely detected
        """
        context_words = set(w.lower() for w in context.split() if len(w) > 4)
        answer_words = set(w.lower() for w in answer.split() if len(w) > 4)

        if not context_words:
            return False

        overlap = len(context_words & answer_words) / len(context_words)
        return overlap < threshold


class EvaluationFramework:
    """Complete evaluation framework for RAG pipeline."""

    def __init__(self):
        """Initialize evaluation framework."""
        self.evaluator = RAGEvaluator()
        self.results_log: list[dict] = []

    def evaluate_response(
        self,
        query: str,
        context: str,
        answer: str,
        retrieval_scores: list[float],
        generation_time_ms: float,
        retrieval_time_ms: float
    ) -> dict:
        """Evaluate a complete RAG response.

        Args:
            query: User query
            context: Retrieved context
            answer: Generated answer
            retrieval_scores: Scores from retrieval
            generation_time_ms: Generation latency
            retrieval_time_ms: Retrieval latency

        Returns:
            Comprehensive evaluation metrics
        """
        documents = context.split("\n\n")

        retrieval_metrics = self.evaluator.compute_retrieval_metrics(
            retrieval_scores,
            documents
        )

        generation_metrics = self.evaluator.compute_generation_metrics(
            answer,
            context,
            query
        )

        hallucination_detected = self.evaluator.detect_potential_hallucination(
            answer,
            context
        )

        evaluation = {
            "query": query,
            "answer": answer,
            "retrieval_metrics": retrieval_metrics,
            "generation_metrics": generation_metrics,
            "hallucination_detected": hallucination_detected,
            "retrieval_latency_ms": retrieval_time_ms,
            "generation_latency_ms": generation_time_ms,
            "total_latency_ms": retrieval_time_ms + generation_time_ms
        }

        self.results_log.append(evaluation)
        return evaluation

    def get_aggregate_metrics(self) -> dict:
        """Compute aggregate metrics from all logged results."""
        if not self.results_log:
            return {}

        all_retrieval_scores = []
        all_generation_times = []
        all_retrieval_times = []
        hallucination_count = 0

        for result in self.results_log:
            all_retrieval_scores.extend(result["retrieval_metrics"]["scores"])
            all_generation_times.append(result["generation_latency_ms"])
            all_retrieval_times.append(result["retrieval_latency_ms"])

            if result["hallucination_detected"]:
                hallucination_count += 1

        return {
            "total_queries_evaluated": len(self.results_log),
            "avg_retrieval_score": float(np.mean(all_retrieval_scores)) if all_retrieval_scores else 0.0,
            "avg_generation_time_ms": float(np.mean(all_generation_times)),
            "avg_retrieval_time_ms": float(np.mean(all_retrieval_times)),
            "avg_total_latency_ms": float(np.mean(all_generation_times)) + float(np.mean(all_retrieval_times)),
            "hallucination_rate": hallucination_count / len(self.results_log),
            "queries_with_potential_hallucinations": hallucination_count,
        }

    def clear_logs(self):
        """Clear evaluation logs."""
        self.results_log = []
