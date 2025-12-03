from sentence_transformers import SentenceTransformer

# Cache for embedding model (singleton pattern)
_embedding_model_cache = {}

def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Get or create cached embedding model instance"""
    if model_name not in _embedding_model_cache:
        print(f"[Loading] Embedding model: {model_name}")
        _embedding_model_cache[model_name] = SentenceTransformer(model_name)
        print(f"[OK] Embedding model loaded and cached")
    return _embedding_model_cache[model_name]
