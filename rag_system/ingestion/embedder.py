from sentence_transformers import SentenceTransformer

def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)
