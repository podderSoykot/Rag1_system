from sentence_transformers import SentenceTransformer
import torch

# Cache for embedding model (singleton pattern)
_embedding_model_cache = {}

def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Get or create cached embedding model instance"""
    if model_name not in _embedding_model_cache:
        print(f"[Loading] Embedding model: {model_name}")
        # Use GPU if available for faster embeddings
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            print(f"  Using GPU for faster embeddings")
        else:
            print(f"  Using CPU (slower, but will work)")
        _embedding_model_cache[model_name] = SentenceTransformer(model_name, device=device)
        print(f"[OK] Embedding model loaded and cached on {device}")
    return _embedding_model_cache[model_name]
