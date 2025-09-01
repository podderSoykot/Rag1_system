# rag_system/retrieval/retriever.py

import os
import pickle
import re
from collections import Counter

def preprocess_text(text):
    """Simple text preprocessing - same as in indexer"""
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class Retriever:
    def __init__(self, db_dir: str, model_name: str):
        # Load the TF-IDF index
        index_path = os.path.join(db_dir, "tfidf_index.pkl")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"TF-IDF index not found at {index_path}")
        
        with open(index_path, "rb") as f:
            self.index_data = pickle.load(f)
        
        print(f"✅ Loaded TF-IDF index with {len(self.index_data['documents'])} documents")

    def search(self, query: str, top_k: int = 3):
        # Preprocess query
        query_terms = preprocess_text(query).split()
        
        # Calculate query-document similarity scores
        scores = []
        for i, doc_scores in enumerate(self.index_data['tfidf_scores']):
            score = 0
            for term in query_terms:
                if term in doc_scores:
                    score += doc_scores[term]
            scores.append((i, score))
        
        # Sort by score (descending) and get top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, score in scores[:top_k] if score > 0]
        
        # Return documents
        results = []
        for idx in top_indices:
            results.append(self.index_data['documents'][idx])
        
        # If no results found, return empty list
        return results if results else ["No relevant documents found."]
