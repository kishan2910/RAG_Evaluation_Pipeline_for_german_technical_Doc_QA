"""LLM service for interacting with Ollama models."""

import logging
import time
import httpx
import ollama
from app.core.config import settings
from app.core.logger import logger

# Create module logger
llm_logger = logging.getLogger(__name__)


class LLMService:
    """Service for managing LLM interactions with Ollama."""

    def __init__(self, model: str | None = None):
        """Initialize the LLM service.

        Args:
            model: Model name to use (defaults to settings.ollama_model)

        Raises:
            ConnectionError: If unable to connect to Ollama
        """
        self.model = model or settings.ollama_model
        self.base_url = settings.ollama_host

        # Verify connection on init
        if not self.check_ollama_connection():
            raise ConnectionError(
                f"Failed to connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running: docker-compose up -d"
            )

        llm_logger.info(f"Initialized LLM service with model: {self.model}")

    def check_ollama_connection(self) -> bool:
        """Check if Ollama is running and accessible.

        Returns:
            True if Ollama is reachable, False otherwise
        """
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            is_connected = response.status_code == 200
            if is_connected:
                llm_logger.debug("Ollama connection successful")
            else:
                llm_logger.warning(f"Ollama returned status code {response.status_code}")
            return is_connected
        except Exception as e:
            llm_logger.debug(f"Failed to connect to Ollama: {type(e).__name__}: {e}")
            return False

    def list_available_models(self) -> list[dict]:
        """List all available models in Ollama.

        Returns:
            List of available model dictionaries

        Raises:
            RuntimeError: If unable to connect to Ollama
        """
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=10.0)
            data = response.json()
            models = data.get("models", [])
            llm_logger.debug(f"Found {len(models)} available models")
            return models
        except Exception as e:
            error_msg = f"Failed to list models: {str(e)}"
            llm_logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def pull_model(self, model: str) -> bool:
        """Pull (download) a model from Ollama registry.

        Args:
            model: Model name to pull

        Returns:
            True if successful, False otherwise
        """
        try:
            llm_logger.info(f"Pulling model '{model}'... This may take a while.")
            ollama.pull(model)
            llm_logger.info(f"Successfully pulled model '{model}'")
            return True
        except Exception as e:
            llm_logger.error(f"Failed to pull model '{model}': {str(e)}")
            return False

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> tuple[str, float]:
        """Generate text using the LLM.

        Args:
            prompt: Input prompt
            system_prompt: System prompt for context
            temperature: Sampling temperature (0-1)
            top_p: Nucleus sampling parameter

        Returns:
            Tuple of (generated_text, generation_time_ms)

        Raises:
            RuntimeError: If generation fails
        """
        try:
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            start_time = time.time()

            response = ollama.generate(
                model=self.model,
                prompt=full_prompt,
                temperature=temperature,
                top_p=top_p,
                stream=False
            )

            generation_time = (time.time() - start_time) * 1000

            generated_text = response.get("response", "").strip()
            llm_logger.debug(f"Generated response in {generation_time:.2f}ms")

            return generated_text, generation_time

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            llm_logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def generate_with_context(
        self,
        query: str,
        context: str,
        system_prompt: str | None = None
    ) -> tuple[str, float]:
        """Generate response using retrieved context (RAG).

        Args:
            query: User query
            context: Retrieved context from vector DB
            system_prompt: System prompt for context

        Returns:
            Tuple of (generated_answer, generation_time_ms)

        Raises:
            RuntimeError: If generation fails
        """
        if not system_prompt:
            system_prompt = """You are a helpful IT support assistant. 
Use the provided context to answer questions accurately and concisely.
If the answer is not found in the context, clearly state that information is not available.
Avoid making up information or hallucinating facts."""

        rag_prompt = f"""Context:
{context}

Question: {query}

Answer:"""

        return self.generate(rag_prompt, system_prompt)
