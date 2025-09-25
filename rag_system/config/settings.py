from pathlib import Path
import os
from dotenv import load_dotenv
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv()
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_CHUNKS = DATA_PROCESSED / "chunks"
VECTOR_DB_DIR = BASE_DIR / "data" / "embeddings" / "chroma_db"
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() in ["1", "true", "yes"]
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
