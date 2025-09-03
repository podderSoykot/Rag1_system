# rag_system/config/settings.py

from pathlib import Path
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
# Load .env file from the current working directory
load_dotenv()

# Data paths
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_CHUNKS = DATA_PROCESSED / "chunks"
VECTOR_DB_DIR = BASE_DIR / "data" / "embeddings" / "chroma_db"

# Embedding & LLM
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID")
OPENAI_PROJECT = os.getenv("OPENAI_PROJECT")

# Ollama (local LLaMA) settings
USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() in ["1", "true", "yes"]
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
