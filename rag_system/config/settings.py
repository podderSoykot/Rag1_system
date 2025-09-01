# rag_system/config/settings.py

from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_CHUNKS = DATA_PROCESSED / "chunks"
VECTOR_DB_DIR = BASE_DIR / "data" / "embeddings" / "chroma_db"

# Embedding & LLM
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-3.5-turbo"
