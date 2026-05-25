"""FastAPI application for the RAG system."""

import logging
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.services.rag import RAGPipeline
from app.core.config import settings
from app.core.logger import logger
from app.schemas.responses import (
    QueryResponse,
    RetrievalMetrics,
    GenerationMetrics,
    HealthResponse,
    StatsResponse,
    ErrorResponse
)
from app.monitoring.metrics import metrics_collector

# Create module logger
api_logger = logging.getLogger(__name__)

# Global RAG pipeline instance
rag_pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    global rag_pipeline

    # Startup
    try:
        api_logger.info("Initializing RAG pipeline...")
        rag_pipeline = RAGPipeline(model=settings.ollama_model)
        api_logger.info("✓ RAG pipeline initialized successfully")
    except Exception as e:
        api_logger.error(f"✗ Failed to initialize RAG pipeline: {str(e)}")
        raise

    yield

    # Shutdown
    api_logger.info("Shutting down application...")


app = FastAPI(
    title="IT Support Helpdesk RAG System",
    description="Enterprise helpdesk powered by RAG with Ollama LLM",
    version="1.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    """Request model for RAG queries."""

    query: str = Field(..., description="The question to ask the RAG system", min_length=1, max_length=2000)
    n_results: int = Field(5, description="Number of documents to retrieve", ge=1, le=50)
    temperature: float = Field(0.7, description="LLM temperature", ge=0.0, le=1.0)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to all requests."""
    request.state.request_id = str(uuid.uuid4())
    request.state.start_time = None

    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


def create_error_response(request_id: str, error_code: str, error_message: str) -> dict:
    """Create a standardized error response."""
    return ErrorResponse(
        request_id=request_id,
        error_code=error_code,
        error_message=error_message
    ).model_dump()


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(request: Request):
    """Health check endpoint - verify system status and connectivity."""
    try:
        if rag_pipeline is None:
            raise RuntimeError("RAG pipeline not initialized")

        db_info = rag_pipeline.vector_db.get_collection_info()
        is_ollama_connected = rag_pipeline.llm.check_ollama_connection()

        return HealthResponse(
            status="healthy" if is_ollama_connected else "degraded",
            ollama_connected=is_ollama_connected,
            chromadb_connected=True,  # We already connected during init
            model=rag_pipeline.llm.model,
            vector_db_size=db_info.get("document_count", 0)
        )
    except Exception as e:
        api_logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=create_error_response(
                request.state.request_id,
                "HEALTH_CHECK_FAILED",
                f"Health check failed: {str(e)}"
            )
        )


@app.get("/stats", response_model=StatsResponse, tags=["System"])
async def get_stats(request: Request):
    """Get detailed system statistics and performance metrics."""
    try:
        if rag_pipeline is None:
            raise RuntimeError("RAG pipeline not initialized")

        pipeline_stats = rag_pipeline.get_stats()
        metrics = metrics_collector.get_metrics()

        return StatsResponse(
            vector_db=pipeline_stats.get("vector_db", {}),
            available_models=pipeline_stats.get("available_models", []),
            current_model=pipeline_stats.get("current_model", ""),
            ollama_running=pipeline_stats.get("ollama_running", False),
            uptime_seconds=metrics.get("uptime_seconds", 0),
            total_requests_processed=metrics.get("total_requests", 0),
            avg_response_time_ms=metrics.get("avg_response_time_ms", 0)
        )
    except Exception as e:
        api_logger.error(f"Failed to get stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                request.state.request_id,
                "STATS_RETRIEVAL_FAILED",
                f"Failed to retrieve statistics: {str(e)}"
            )
        )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(request: Request, query_request: QueryRequest):
    """Process a RAG query and return answer with retrieved context and metrics."""
    try:
        if rag_pipeline is None:
            raise RuntimeError("RAG pipeline not initialized")

        # Validate input
        if not query_request.query.strip():
            raise ValueError("Query cannot be empty")

        # Process query
        import time
        start_time = time.time()

        result = rag_pipeline.answer(
            query_request.query,
            n_results=query_request.n_results
        )

        total_time = (time.time() - start_time) * 1000

        # Record metrics
        metrics_collector.record_request(
            response_time_ms=total_time,
            endpoint="/query",
            success=True
        )

        # Build response with metrics
        retrieval_scores = result.get("retrieval_scores", [])
        avg_retrieval_score = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0

        response = QueryResponse(
            request_id=request.state.request_id,
            query=query_request.query,
            retrieved_documents=result.get("retrieved_documents", []),
            answer=result.get("answer", ""),
            retrieval_metrics=RetrievalMetrics(
                num_retrieved=len(result.get("retrieved_documents", [])),
                retrieval_scores=retrieval_scores,
                avg_score=avg_retrieval_score
            ),
            generation_metrics=GenerationMetrics(
                model=rag_pipeline.llm.model,
                temperature=query_request.temperature,
                generation_time_ms=result.get("generation_time_ms", 0)
            ),
            total_time_ms=total_time
        )

        api_logger.info(
            f"Query processed successfully | "
            f"Request: {request.state.request_id} | "
            f"Time: {total_time:.2f}ms | "
            f"Retrieved: {response.retrieval_metrics.num_retrieved} docs"
        )

        return response

    except ValueError as e:
        api_logger.warning(f"Invalid input: {str(e)}")
        metrics_collector.record_request(
            response_time_ms=0,
            endpoint="/query",
            success=False
        )
        raise HTTPException(
            status_code=400,
            detail=create_error_response(
                request.state.request_id,
                "INVALID_INPUT",
                str(e)
            )
        )
    except RuntimeError as e:
        api_logger.error(f"RAG pipeline error: {str(e)}")
        metrics_collector.record_request(
            response_time_ms=0,
            endpoint="/query",
            success=False
        )
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                request.state.request_id,
                "RAG_PIPELINE_ERROR",
                str(e)
            )
        )
    except Exception as e:
        api_logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        metrics_collector.record_request(
            response_time_ms=0,
            endpoint="/query",
            success=False
        )
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                request.state.request_id,
                "INTERNAL_SERVER_ERROR",
                "An unexpected error occurred. Please check the logs."
            )
        )


@app.get("/models", tags=["System"])
async def list_models(request: Request):
    """List available LLM models in Ollama."""
    try:
        if rag_pipeline is None:
            raise RuntimeError("RAG pipeline not initialized")

        models = rag_pipeline.llm.list_available_models()
        return {
            "request_id": request.state.request_id,
            "models": [m.get("name") for m in models],
            "current_model": rag_pipeline.llm.model
        }
    except Exception as e:
        api_logger.error(f"Failed to list models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                request.state.request_id,
                "MODEL_LIST_ERROR",
                f"Failed to list models: {str(e)}"
            )
        )


@app.post("/pull-model", tags=["System"])
async def pull_model(request: Request, model: str):
    """Pull (download) a model from Ollama registry."""
    try:
        if rag_pipeline is None:
            raise RuntimeError("RAG pipeline not initialized")

        if not model.strip():
            raise ValueError("Model name cannot be empty")

        success = rag_pipeline.llm.pull_model(model)

        if not success:
            raise RuntimeError(f"Failed to pull model: {model}")

        return {
            "request_id": request.state.request_id,
            "status": "success",
            "model": model
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        api_logger.error(f"Failed to pull model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                request.state.request_id,
                "MODEL_PULL_ERROR",
                f"Failed to pull model: {str(e)}"
            )
        )


@app.get("/metrics", tags=["System"])
async def get_metrics(request: Request):
    """Get system performance metrics."""
    try:
        metrics = metrics_collector.get_metrics()
        return {
            "request_id": request.state.request_id,
            "metrics": metrics
        }
    except Exception as e:
        api_logger.error(f"Failed to get metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                request.state.request_id,
                "METRICS_RETRIEVAL_ERROR",
                f"Failed to retrieve metrics: {str(e)}"
            )
        )


@app.get("/", tags=["System"])
async def root():
    """Root endpoint - API documentation."""
    return {
        "name": "IT Support Enterprise Helpdesk RAG System",
        "version": "1.0.0",
        "description": "Production-ready RAG system for IT support",
        "endpoints": {
            "docs": "/docs",
            "openapi": "/openapi.json",
            "health": "/health",
            "stats": "/stats",
            "metrics": "/metrics",
            "query": "/query",
            "models": "/models"
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=6000,
        log_level="info"
    )
