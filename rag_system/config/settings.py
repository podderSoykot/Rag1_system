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

# Chunking configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))  # Increased for better context
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))  # Increased overlap for better continuity

# Search configuration
USE_HYBRID_SEARCH = os.getenv("USE_HYBRID_SEARCH", "true").lower() in ["1", "true", "yes"]
SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", "0.7"))  # Weight for semantic search in hybrid
TFIDF_WEIGHT = float(os.getenv("TFIDF_WEIGHT", "0.3"))  # Weight for TF-IDF in hybrid
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "20"))  # Number of results to rerank
USE_RERANKING = os.getenv("USE_RERANKING", "false").lower() in ["1", "true", "yes"]

# Caching configuration
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() in ["1", "true", "yes"]
CACHE_SIZE = int(os.getenv("CACHE_SIZE", "100"))  # LRU cache size

# Embedding configuration
# Increase batch size for faster processing (128 is good for CPU, 256+ for GPU)
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "128"))

# Query expansion
USE_QUERY_EXPANSION = os.getenv("USE_QUERY_EXPANSION", "true").lower() in ["1", "true", "yes"]

# Retrieval quality filters
MIN_CHUNK_LENGTH = int(os.getenv("MIN_CHUNK_LENGTH", "100"))  # Minimum chunk length to consider
LENGTH_BOOST_FACTOR = float(os.getenv("LENGTH_BOOST_FACTOR", "0.1"))  # Boost score for longer chunks