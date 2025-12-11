from ingestion.data_loader import process_pdfs
from ingestion.chunker import process_files
from ingestion.indexer import index_documents
from retrieval.retriever import Retriever
from synthesis.prompt_builder import build_prompt
from synthesis.generator import generate_answer
from config.settings import (
    DATA_RAW, DATA_PROCESSED, DATA_CHUNKS, VECTOR_DB_DIR, EMB_MODEL_NAME,
    CACHE_ENABLED, CACHE_SIZE
)
import argparse
import hashlib
import time
import os
from pathlib import Path

# Global retriever instance (cached)
_retriever_instance = None

def get_retriever():
    """Get or create cached retriever instance"""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = Retriever(str(VECTOR_DB_DIR), EMB_MODEL_NAME)
    return _retriever_instance

# Result cache
_result_cache = {}
_cache_hits = 0
_cache_misses = 0

def _get_cache_key(query: str, top_k: int):
    """Generate cache key from query and parameters"""
    key_string = f"{query.lower().strip()}:{top_k}"
    return hashlib.md5(key_string.encode()).hexdigest()

def needs_ingestion():
    """Check if ingestion is needed by comparing file modification times"""
    # Check if indexes exist
    faiss_path = Path(VECTOR_DB_DIR) / "faiss_index.bin"
    embeddings_path = Path(VECTOR_DB_DIR) / "embeddings.npy"
    metadata_path = Path(VECTOR_DB_DIR) / "vector_metadata.pkl"
    tfidf_path = Path(VECTOR_DB_DIR) / "tfidf_index.pkl"
    
    # If no indexes exist, ingestion is needed
    if not (faiss_path.exists() or embeddings_path.exists()) or not metadata_path.exists() or not tfidf_path.exists():
        return True, "Indexes not found"
    
    # Get the most recent index modification time
    index_times = []
    if faiss_path.exists():
        index_times.append(faiss_path.stat().st_mtime)
    if embeddings_path.exists():
        index_times.append(embeddings_path.stat().st_mtime)
    if metadata_path.exists():
        index_times.append(metadata_path.stat().st_mtime)
    if tfidf_path.exists():
        index_times.append(tfidf_path.stat().st_mtime)
    
    if not index_times:
        return True, "No indexes found"
    
    latest_index_time = max(index_times)
    
    # Check if any source PDFs are newer than indexes
    if DATA_RAW.exists():
        for pdf_file in DATA_RAW.glob("*.pdf"):
            pdf_time = pdf_file.stat().st_mtime
            if pdf_time > latest_index_time:
                return True, f"Source file {pdf_file.name} is newer than indexes"
    
    # Check if processed text files are newer than indexes
    if DATA_PROCESSED.exists():
        for txt_file in DATA_PROCESSED.glob("*.txt"):
            txt_time = txt_file.stat().st_mtime
            if txt_time > latest_index_time:
                return True, f"Processed file {txt_file.name} is newer than indexes"
    
    # Check if chunks are newer than indexes
    if DATA_CHUNKS.exists():
        for chunk_file in DATA_CHUNKS.glob("*_chunks.txt"):
            chunk_time = chunk_file.stat().st_mtime
            if chunk_time > latest_index_time:
                return True, f"Chunk file {chunk_file.name} is newer than indexes"
    
    # Check if chunks exist for all processed files
    if DATA_PROCESSED.exists() and DATA_CHUNKS.exists():
        txt_files = list(DATA_PROCESSED.glob("*.txt"))
        for txt_file in txt_files:
            chunk_file = DATA_CHUNKS / f"{txt_file.stem}_chunks.txt"
            if not chunk_file.exists():
                return True, f"Chunks missing for {txt_file.name}"
    
    # Everything is up to date
    return False, "All files up to date"

def run_ingestion(force=False):
    """Run ingestion if needed, or skip if everything is up to date"""
    reason = "Force mode"
    # Check if ingestion is needed
    if not force:
        needs_it, reason = needs_ingestion()
        if not needs_it:
            print("\n" + "="*60)
            print("✓ INGESTION SKIPPED - All files are up to date!")
            print(f"  Reason: {reason}")
            print("="*60 + "\n")
            return
    
    start_time = time.time()
    print("\n" + "="*60)
    print("RAG SYSTEM INGESTION - Processing Documents")
    if force:
        print("(Force mode: Re-processing all files)")
    else:
        print(f"(Reason: {reason})")
    print("="*60)
    
    # Stage 1: PDF Processing (0-25%)
    stage1_start = time.time()
    print("\n[Stage 1/4] Extracting text from PDFs... (0% - 25%)")
    process_pdfs(DATA_RAW, DATA_PROCESSED)
    stage1_time = time.time() - stage1_start
    print(f"✓ PDF extraction complete! (25%) - Time: {stage1_time:.1f}s")
    
    # Stage 2: Chunking (25-50%)
    stage2_start = time.time()
    print(f"\n[Stage 2/4] Creating semantic chunks... (25% - 50%)")
    process_files(DATA_PROCESSED, DATA_CHUNKS)
    stage2_time = time.time() - stage2_start
    print(f"✓ Chunking complete! (50%) - Time: {stage2_time:.1f}s")
    
    # Stage 3 & 4: Indexing (50-100%)
    stage3_start = time.time()
    print(f"\n[Stage 3-4/4] Generating embeddings and indexing... (50% - 100%)")
    index_documents(DATA_CHUNKS, str(VECTOR_DB_DIR), EMB_MODEL_NAME)
    stage3_time = time.time() - stage3_start
    print(f"✓ Indexing complete! (100%) - Time: {stage3_time:.1f}s")
    
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    
    print("\n" + "="*60)
    print(f"✓ INGESTION COMPLETE! Total time: {minutes}m {seconds}s")
    print("="*60 + "\n")
    
    # Clear retriever cache after re-indexing
    global _retriever_instance
    _retriever_instance = None

def rag_pipeline(query: str, top_k: int = 3, use_cache: bool = None):
    """RAG pipeline with optional result caching"""
    if use_cache is None:
        use_cache = CACHE_ENABLED
    
    # Check cache
    if use_cache:
        cache_key = _get_cache_key(query, top_k)
        if cache_key in _result_cache:
            global _cache_hits
            _cache_hits += 1
            print(f"[Cache Hit] Returning cached result")
            return _result_cache[cache_key]
    
    # Cache miss - perform retrieval and generation
    global _cache_misses
    _cache_misses += 1
    
    retriever = get_retriever()
    docs = retriever.search(query, top_k=top_k)
    prompt = build_prompt(query, docs)
    answer = generate_answer(prompt)
    
    # Store in cache
    if use_cache:
        cache_key = _get_cache_key(query, top_k)
        # Implement LRU: remove oldest if cache is full
        if len(_result_cache) >= CACHE_SIZE:
            # Remove first (oldest) item
            oldest_key = next(iter(_result_cache))
            del _result_cache[oldest_key]
        _result_cache[cache_key] = answer
    
    return answer
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAG pipeline")
    parser.add_argument("--query", type=str, default=None, help="Single question to run non-interactively")
    parser.add_argument("--top_k", type=int, default=3, help="Number of chunks to retrieve")
    parser.add_argument("--force-ingestion", action="store_true", help="Force re-ingestion even if files are up to date")
    args = parser.parse_args()
    run_ingestion(force=args.force_ingestion)
    if args.query:
        print(f"\nProcessing: {args.query}")
        print("-" * 40)
        try:
            answer = rag_pipeline(args.query, top_k=args.top_k)
            print(f"\nAnswer:\n{answer}")
            print("-" * 40)
        except Exception as e:
            print(f"\nError: {e}")
        raise SystemExit(0)
    print("\n" + "="*60)
    print("RAG System Ready! Ask questions about your documents.")
    print("="*60)
    while True:
        try:
            query = input("\nEnter your question (or 'quit' to exit): ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            if not query:
                print("Please enter a question.")
                continue
            print(f"\nProcessing: {query}")
            print("-" * 40)
            answer = rag_pipeline(query)
            print(f"\nAnswer:\n{answer}")
            print("-" * 40)
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again with a different question.")
