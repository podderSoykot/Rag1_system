from pathlib import Path
import os
from dotenv import load_dotenv
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_CHUNKS = DATA_PROCESSED / "chunks"
VECTOR_DB_DIR = BASE_DIR / "data" / "embeddings" / "chroma_db"
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
USE_OLLAMA = os.getenv("USE_OLLAMA", "true").lower() in ["1", "true", "yes"]  # Default to true for better performance
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))  # Increased timeout for longer responses

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
HF_HUB_OFFLINE = os.getenv("HF_HUB_OFFLINE", "false").lower() in ["1", "true", "yes"]

# Query expansion
USE_QUERY_EXPANSION = os.getenv("USE_QUERY_EXPANSION", "true").lower() in ["1", "true", "yes"]

# Retrieval quality filters
MIN_CHUNK_LENGTH = int(os.getenv("MIN_CHUNK_LENGTH", "50"))  # Minimum chunk length to consider (reduced for better coverage)
LENGTH_BOOST_FACTOR = float(os.getenv("LENGTH_BOOST_FACTOR", "0.1"))  # Boost score for longer chunks

# Pinecone configuration
USE_PINECONE = os.getenv("USE_PINECONE", "true").lower() in ["1", "true", "yes"]
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-system")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")  # For serverless, this is the region
PINECONE_DIMENSION = int(os.getenv("PINECONE_DIMENSION", "384"))  # Dimension for all-MiniLM-L6-v2 is 384

# Performance monitoring
ENABLE_PERFORMANCE_PROFILING = os.getenv("ENABLE_PERFORMANCE_PROFILING", "false").lower() in ["1", "true", "yes"]

# Answer generation configuration
MAX_ANSWER_TOKENS = int(os.getenv("MAX_ANSWER_TOKENS", "1000"))
RESEARCH_MAX_TOKENS = int(os.getenv("RESEARCH_MAX_TOKENS", "1500"))
RESEARCH_MAX_SUB_QUERIES = int(os.getenv("RESEARCH_MAX_SUB_QUERIES", "4"))

# API security
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
MAX_UPLOAD_FILES = int(os.getenv("MAX_UPLOAD_FILES", "10"))
RATE_LIMIT_QUERY_PER_MIN = int(os.getenv("RATE_LIMIT_QUERY_PER_MIN", "30"))
RATE_LIMIT_RESEARCH_PER_MIN = int(os.getenv("RATE_LIMIT_RESEARCH_PER_MIN", "5"))
RATE_LIMIT_UPLOAD_PER_MIN = int(os.getenv("RATE_LIMIT_UPLOAD_PER_MIN", "10"))
RATE_LIMIT_INGEST_PER_MIN = int(os.getenv("RATE_LIMIT_INGEST_PER_MIN", "3"))
API_DOCS_ENABLED = os.getenv("API_DOCS_ENABLED", "true").lower() in ["1", "true", "yes"]

# OpenAI configuration (supports both OPENAI_API_KEY and openai_api_key in .env)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("openai_api_key", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL") or os.getenv("model", "gpt-4o-mini")
USE_OPENAI = os.getenv(
    "USE_OPENAI",
    "true" if OPENAI_API_KEY else "false",
).lower() in ["1", "true", "yes"]