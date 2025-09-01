# rag_system/ingestion/indexer.py

import os
import pickle
import re
from collections import Counter
from tqdm import tqdm

def preprocess_text(text):
    """Simple text preprocessing"""
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_tfidf_index(documents):
    """Create a simple TF-IDF index"""
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
    """Index chunked documents using simple TF-IDF approach."""
    os.makedirs(db_dir, exist_ok=True)
    
    # Collect all documents
    documents = []
    metadatas = []
    
    doc_id = 0
    for filename in os.listdir(input_dir):
        if filename.endswith("_chunks.txt"):
            filepath = os.path.join(input_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                chunks = [line.strip() for line in f if line.strip()]

            print(f"🔄 Processing {filename} with {len(chunks)} chunks...")
            
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Only add non-empty chunks
                    documents.append(chunk)
                    metadatas.append({"source": filename, "chunk_id": i})
                    doc_id += 1

    print(f"🔄 Creating TF-IDF index for {len(documents)} documents...")
    
    # Create TF-IDF index
    index_data = create_tfidf_index(documents)
    index_data["metadatas"] = metadatas
    
    # Save to disk
    index_path = os.path.join(db_dir, "tfidf_index.pkl")
    with open(index_path, "wb") as f:
        pickle.dump(index_data, f)
    
    print(f"✅ Indexed {doc_id} chunks using TF-IDF at {index_path}")
    print(f"📊 Vocabulary size: {len(index_data['all_terms'])} terms")
