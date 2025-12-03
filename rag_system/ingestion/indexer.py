# rag_system/ingestion/indexer.py

import os
import pickle
import re
import numpy as np
from collections import Counter
from ingestion.embedder import get_embedding_model
from config.settings import EMBEDDING_BATCH_SIZE

# Try to import FAISS, fallback to simple vector storage if not available
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("[Warning] FAISS not available. Using simple vector storage.")

def preprocess_text(text):
    """Simple text preprocessing"""
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_tfidf_index(documents):
    """Create a simple TF-IDF index (kept for hybrid search)"""
    # Calculate term frequencies for each document
    doc_terms = []
    all_terms = set()
    
    for doc in documents:
        terms = preprocess_text(doc).split()
        doc_terms.append(terms)
        all_terms.update(terms)
    
    # Calculate document frequency (how many docs contain each term)
    doc_freq = Counter()
    for terms in doc_terms:
        doc_freq.update(set(terms))
    
    # Calculate TF-IDF scores
    tfidf_scores = []
    for terms in doc_terms:
        term_freq = Counter(terms)
        doc_scores = {}
        for term in terms:
            if term in doc_freq:
                tf = term_freq[term] / len(terms)  # Term frequency
                idf = len(documents) / doc_freq[term]  # Inverse document frequency
                doc_scores[term] = tf * idf
        tfidf_scores.append(doc_scores)
    
    return {
        "documents": documents,
        "tfidf_scores": tfidf_scores,
        "all_terms": list(all_terms),
        "doc_freq": dict(doc_freq)
    }

def index_documents(input_dir: str, db_dir: str, model_name: str):
    """Index chunked documents using ChromaDB for vector search and TF-IDF as fallback."""
    os.makedirs(db_dir, exist_ok=True)
    
    # Collect all documents
    documents = []
    metadatas = []
    ids = []
    
    doc_id = 0
    for filename in os.listdir(input_dir):
        if filename.endswith("_chunks.txt"):
            filepath = os.path.join(input_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                chunks = [line.strip() for line in f if line.strip()]

            print(f"[Processing] {filename} with {len(chunks)} chunks...")
            
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Only add non-empty chunks
                    documents.append(chunk)
                    metadatas.append({"source": filename, "chunk_id": i})
                    ids.append(f"{filename}_{i}")
                    doc_id += 1

    if not documents:
        print("[Warning] No documents found to index")
        return

    total_docs = len(documents)
    print(f"  Indexing {total_docs} document chunks...")
    
    # Get embedding model
    print(f"  Loading embedding model...", end="", flush=True)
    embedding_model = get_embedding_model(model_name)
    print(f" ✓")
    
    # Generate embeddings in batches
    print(f"  Generating embeddings (batch size: {EMBEDDING_BATCH_SIZE})...")
    embeddings = []
    num_batches = (total_docs + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE
    
    for batch_idx, i in enumerate(range(0, len(documents), EMBEDDING_BATCH_SIZE), 1):
        batch = documents[i:i + EMBEDDING_BATCH_SIZE]
        # Calculate progress: 50-75% for embedding generation
        progress = 50 + int((batch_idx / num_batches) * 25)
        print(f"    Batch {batch_idx}/{num_batches} - {progress}%", end="", flush=True)
        
        batch_embeddings = embedding_model.encode(batch, show_progress_bar=False)
        embeddings.extend(batch_embeddings.tolist())
        
        print(f" ✓")
    
    embeddings_array = np.array(embeddings, dtype=np.float32)
    embedding_dim = embeddings_array.shape[1]
    
    # Store embeddings using FAISS or simple numpy storage (75-85%)
    print(f"  Creating vector index... (75% - 85%)", end="", flush=True)
    if HAS_FAISS:
        # Create FAISS index
        index = faiss.IndexFlatL2(embedding_dim)  # L2 distance (Euclidean)
        index.add(embeddings_array)
        
        # Save FAISS index
        faiss_path = os.path.join(db_dir, "faiss_index.bin")
        faiss.write_index(index, faiss_path)
        print(f" ✓ FAISS index saved")
    else:
        # Fallback: save as numpy array
        embeddings_path = os.path.join(db_dir, "embeddings.npy")
        np.save(embeddings_path, embeddings_array)
        print(f" ✓ Numpy embeddings saved")
    
    # Save document metadata and IDs
    print(f"  Saving metadata... (85% - 90%)", end="", flush=True)
    metadata_path = os.path.join(db_dir, "vector_metadata.pkl")
    with open(metadata_path, "wb") as f:
        pickle.dump({
            "documents": documents,
            "metadatas": metadatas,
            "ids": ids,
            "embedding_dim": embedding_dim
        }, f)
    print(f" ✓")
    
    # Also create TF-IDF index for hybrid search fallback (90-100%)
    print(f"  Creating TF-IDF index... (90% - 100%)", end="", flush=True)
    index_data = create_tfidf_index(documents)
    index_data["metadatas"] = metadatas
    
    # Save TF-IDF to disk
    index_path = os.path.join(db_dir, "tfidf_index.pkl")
    with open(index_path, "wb") as f:
        pickle.dump(index_data, f)
    
    print(f" ✓ TF-IDF index saved")
    print(f"  Vocabulary size: {len(index_data['all_terms'])} terms")
