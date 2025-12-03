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

def run_ingestion():
    print("\n" + "="*60)
    print("RAG SYSTEM INGESTION - Processing Documents")
    print("="*60)
    
    # Stage 1: PDF Processing (0-25%)
    print("\n[Stage 1/4] Extracting text from PDFs... (0% - 25%)")
    process_pdfs(DATA_RAW, DATA_PROCESSED)
    print("✓ PDF extraction complete! (25%)")
    
    # Stage 2: Chunking (25-50%)
    print("\n[Stage 2/4] Creating semantic chunks... (25% - 50%)")
    process_files(DATA_PROCESSED, DATA_CHUNKS)
    print("✓ Chunking complete! (50%)")
    
    # Stage 3 & 4: Indexing (50-100%)
    print("\n[Stage 3-4/4] Generating embeddings and indexing... (50% - 100%)")
    index_documents(DATA_CHUNKS, str(VECTOR_DB_DIR), EMB_MODEL_NAME)
    print("✓ Indexing complete! (100%)")
    
    print("\n" + "="*60)
    print("✓ INGESTION COMPLETE! System ready for queries.")
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
    args = parser.parse_args()
    run_ingestion()
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
