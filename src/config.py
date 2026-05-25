import os
from dotenv import load_dotenv

load_dotenv()

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000

EMBEDDING_API = "http://localhost:6000/embed"

LLM_API_URL = os.getenv("LLM_API_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

COLLECTION_NAME = "techqa"

TOP_K = 5