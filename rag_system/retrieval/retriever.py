# rag_system/retrieval/retriever.py

import os
import pickle
import re
import numpy as np
from collections import Counter
from ingestion.embedder import get_embedding_model
from config.settings import (
    USE_HYBRID_SEARCH, SEMANTIC_WEIGHT, TFIDF_WEIGHT,
    USE_RERANKING, RERANK_TOP_K, USE_QUERY_EXPANSION
)
try:
    from retrieval.query_expansion import expand_query
except ImportError:
    def expand_query(query: str):
        return [query]

# Try to import FAISS
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

def preprocess_text(text):
    """Simple text preprocessing - same as in indexer"""
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class Retriever:
    def __init__(self, db_dir: str, model_name: str):
        self.db_dir = db_dir
        self.model_name = model_name
        
        # Load vector index (FAISS or numpy)
        self.use_vector_search = False
        self.vector_index = None
        self.vector_metadata = None
        
        # Try FAISS first
        faiss_path = os.path.join(db_dir, "faiss_index.bin")
        embeddings_path = os.path.join(db_dir, "embeddings.npy")
        metadata_path = os.path.join(db_dir, "vector_metadata.pkl")
        
        if HAS_FAISS and os.path.exists(faiss_path) and os.path.exists(metadata_path):
            try:
                self.vector_index = faiss.read_index(faiss_path)
                with open(metadata_path, "rb") as f:
                    self.vector_metadata = pickle.load(f)
                self.use_vector_search = True
                print(f"[OK] Loaded FAISS index with {len(self.vector_metadata['documents'])} documents")
            except Exception as e:
                print(f"[Warning] FAISS index loading failed: {e}")
        
        # Fallback to numpy embeddings
        if not self.use_vector_search and os.path.exists(embeddings_path) and os.path.exists(metadata_path):
            try:
                self.embeddings_array = np.load(embeddings_path)
                with open(metadata_path, "rb") as f:
                    self.vector_metadata = pickle.load(f)
                self.use_vector_search = True
                print(f"[OK] Loaded numpy embeddings with {len(self.vector_metadata['documents'])} documents")
            except Exception as e:
                print(f"[Warning] Numpy embeddings loading failed: {e}")
        
        # Load TF-IDF index (for hybrid search or fallback)
        index_path = os.path.join(db_dir, "tfidf_index.pkl")
        if os.path.exists(index_path):
            with open(index_path, "rb") as f:
                self.index_data = pickle.load(f)
            self.use_tfidf = True
            if not self.use_chroma:
                print(f"[OK] Loaded TF-IDF index with {len(self.index_data['documents'])} documents")
        else:
            self.index_data = None
            self.use_tfidf = False
            if not self.use_chroma:
                raise FileNotFoundError(f"No index found at {db_dir}")
        
        # Get embedding model for semantic search
        if self.use_vector_search or USE_HYBRID_SEARCH:
            self.embedding_model = get_embedding_model(model_name)
        
        # Reranking model (optional, using cross-encoder)
        self.rerank_model = None
        if USE_RERANKING:
            try:
                # Use a cross-encoder model for better reranking
                from sentence_transformers import CrossEncoder
                self.rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                print("[OK] Reranking model loaded")
            except ImportError:
                print("[Warning] CrossEncoder not available. Install with: pip install sentence-transformers[reranking]")
            except Exception as e:
                print(f"[Warning] Reranking model not available: {e}")

    def _semantic_search(self, query: str, top_k: int):
        """Perform semantic search using FAISS or numpy"""
        if not self.use_vector_search or not self.vector_metadata:
            return []
        
        try:
            # Get query embedding
            query_embedding = self.embedding_model.encode(query, show_progress_bar=False)
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # Search using FAISS
            if HAS_FAISS and self.vector_index is not None:
                # FAISS search
                k = min(top_k * 2, len(self.vector_metadata['documents']), 50)
                distances, indices = self.vector_index.search(query_vector, k)
                
                # Convert distances to similarities (inverse of normalized distance)
                results = []
                for idx, dist in zip(indices[0], distances[0]):
                    if idx < len(self.vector_metadata['documents']):
                        # Convert L2 distance to similarity (higher distance = lower similarity)
                        # Normalize: similarity = 1 / (1 + distance)
                        similarity = 1.0 / (1.0 + dist)
                        doc = self.vector_metadata['documents'][idx]
                        results.append((doc, similarity))
                
                return results
            
            # Fallback: numpy-based search (brute force)
            elif hasattr(self, 'embeddings_array'):
                # Compute cosine similarity
                query_norm = np.linalg.norm(query_vector)
                if query_norm == 0:
                    return []
                
                # Normalize query
                query_normalized = query_vector / query_norm
                
                # Compute dot products (cosine similarity)
                similarities = np.dot(self.embeddings_array, query_normalized.T).flatten()
                
                # Get top-k indices
                k = min(top_k * 2, len(similarities), 50)
                top_indices = np.argsort(similarities)[::-1][:k]
                
                # Return documents with similarities
                results = []
                for idx in top_indices:
                    if similarities[idx] > 0:  # Only positive similarities
                        doc = self.vector_metadata['documents'][idx]
                        results.append((doc, float(similarities[idx])))
                
                return results
            
            return []
        except Exception as e:
            print(f"[Warning] Semantic search failed: {e}")
            return []

    def _tfidf_search(self, query: str, top_k: int):
        """Perform TF-IDF search"""
        if not self.use_tfidf or not self.index_data:
            return []
        
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
        
        # Normalize scores (0-1 range)
        max_score = max([s[1] for s in scores]) if scores else 1.0
        normalized_scores = [(idx, (score / max_score) if max_score > 0 else 0.0) 
                            for idx, score in scores[:top_k * 2] if score > 0]
        
        # Return documents with scores
        results = []
        for idx, score in normalized_scores:
            results.append((self.index_data['documents'][idx], score))
        
        return results

    def _rerank(self, query: str, candidates: list, top_k: int):
        """Rerank candidates using cross-encoder"""
        if not self.rerank_model or not candidates:
            return candidates[:top_k]
        
        try:
            # Prepare pairs for reranking
            pairs = [[query, doc] for doc, _ in candidates]
            
            # Get reranking scores
            rerank_scores = self.rerank_model.predict(pairs)
            
            # Combine with original scores
            reranked = [(doc, (orig_score * 0.3 + rerank_score * 0.7)) 
                       for (doc, orig_score), rerank_score in zip(candidates, rerank_scores)]
            
            # Sort by combined score
            reranked.sort(key=lambda x: x[1], reverse=True)
            
            return reranked[:top_k]
        except Exception as e:
            print(f"[Warning] Reranking failed: {e}")
            return candidates[:top_k]

    def search(self, query: str, top_k: int = 3):
        """Hybrid search combining semantic and TF-IDF"""
        # Expand query if enabled
        if USE_QUERY_EXPANSION:
            expanded_queries = expand_query(query)
            # Use original query primarily
            search_query = query
        
        # Perform semantic search
        semantic_results = []
        if self.use_vector_search:
            semantic_results = self._semantic_search(query, top_k * 2)
        
        # Perform TF-IDF search
        tfidf_results = []
        if self.use_tfidf:
            tfidf_results = self._tfidf_search(query, top_k * 2)
        
        # Combine results
        if USE_HYBRID_SEARCH and semantic_results and tfidf_results:
            # Hybrid search: combine semantic and TF-IDF
            # Create a combined score dictionary
            combined_scores = {}
            
            # Add semantic results with weight
            for doc, score in semantic_results:
                doc_key = doc[:100]  # Use first 100 chars as key
                combined_scores[doc_key] = {
                    'doc': doc,
                    'semantic_score': score * SEMANTIC_WEIGHT,
                    'tfidf_score': 0.0
                }
            
            # Add TF-IDF results with weight
            for doc, score in tfidf_results:
                doc_key = doc[:100]
                if doc_key in combined_scores:
                    combined_scores[doc_key]['tfidf_score'] = score * TFIDF_WEIGHT
                else:
                    combined_scores[doc_key] = {
                        'doc': doc,
                        'semantic_score': 0.0,
                        'tfidf_score': score * TFIDF_WEIGHT
                    }
            
            # Calculate combined scores
            combined = [(data['doc'], data['semantic_score'] + data['tfidf_score']) 
                       for data in combined_scores.values()]
            combined.sort(key=lambda x: x[1], reverse=True)
            
            candidates = combined
        elif semantic_results:
            # Use only semantic search
            candidates = semantic_results
        elif tfidf_results:
            # Use only TF-IDF search
            candidates = tfidf_results
        else:
            return ["No relevant documents found."]
        
        # Rerank if enabled
        if USE_RERANKING and len(candidates) > top_k:
            candidates = self._rerank(query, candidates, RERANK_TOP_K)
        
        # Return top-k documents
        results = [doc for doc, score in candidates[:top_k]]
        
        return results if results else ["No relevant documents found."]
