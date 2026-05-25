"""Logging configuration for the RAG system."""

import logging
import logging.handlers
from pathlib import Path
from app.core.config import settings


def setup_logging():
    """Setup logging with both file and console handlers."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create logger
    logger = logging.getLogger("it-support-rag")
    logger.setLevel(logging.DEBUG)

    # File handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=10
    )
    file_handler.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging()
