"""Configuration settings for the RAG system."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # Ollama settings
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "mistral"  # Change to your deployed model

    # ChromaDB settings
    chromadb_host: str = "localhost"
    chromadb_port: int = 8000

    # Embeddings settings
    embedding_model: str = "all-MiniLM-L6-v2"

    # Dataset settings
    huggingface_dataset: str = "rojagtap/tech-qa"

    class Config:
        env_file = ".env"


settings = Settings()
