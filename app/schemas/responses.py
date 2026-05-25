"""Response schemas for RAG API endpoints."""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field


class RetrievalMetrics(BaseModel):
    """Metrics for document retrieval."""

    num_retrieved: int = Field(description="Number of documents retrieved")
    retrieval_scores: list[float] = Field(description="Similarity scores for each document")
    avg_score: float = Field(description="Average similarity score")


class GenerationMetrics(BaseModel):
    """Metrics for answer generation."""

    model: str = Field(description="Model used for generation")
    temperature: float = Field(description="Temperature parameter used")
    generation_time_ms: float = Field(description="Time to generate answer in milliseconds")


class QueryResponse(BaseModel):
    """Complete response for a RAG query."""

    request_id: str = Field(description="Unique request identifier")
    query: str = Field(description="Original user query")
    retrieved_documents: list[str] = Field(description="Retrieved context documents")
    answer: str = Field(description="Generated answer using retrieved context")
    retrieval_metrics: RetrievalMetrics = Field(description="Retrieval performance metrics")
    generation_metrics: GenerationMetrics = Field(description="Generation performance metrics")
    total_time_ms: float = Field(description="Total request time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="System status")
    ollama_connected: bool = Field(description="Ollama connection status")
    chromadb_connected: bool = Field(description="ChromaDB connection status")
    model: str = Field(description="Current LLM model")
    vector_db_size: int = Field(description="Number of documents in vector DB")


class StatsResponse(BaseModel):
    """System statistics response."""

    vector_db: dict = Field(description="Vector database information")
    available_models: list[str] = Field(description="Available LLM models")
    current_model: str = Field(description="Currently active LLM model")
    ollama_running: bool = Field(description="Ollama service status")
    uptime_seconds: float = Field(description="System uptime in seconds")
    total_requests_processed: int = Field(description="Total queries processed")
    avg_response_time_ms: float = Field(description="Average response time")


class ErrorResponse(BaseModel):
    """Standard error response."""

    request_id: str = Field(description="Request identifier for tracing")
    error_code: str = Field(description="Error code")
    error_message: str = Field(description="Human-readable error message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
