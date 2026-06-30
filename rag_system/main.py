import sys

if __name__ == "__main__" and "--mcp" in sys.argv:
    import time

    print("[Info] Loading RAG MCP server...", file=sys.stderr, flush=True)
    t0 = time.time()
    from mcp_server import log_mcp_startup, mcp

    log_mcp_startup(time.time() - t0)
    mcp.run()
    raise SystemExit(0)

from ingestion.data_loader import process_pdfs
from ingestion.chunker import process_files
from ingestion.indexer import index_documents
from rag_agent.graph import run_rag_graph, reset_retriever
from config.settings import (
    DATA_RAW, DATA_PROCESSED, DATA_CHUNKS, VECTOR_DB_DIR, EMB_MODEL_NAME,
)
from debug_log import debug_log
import argparse
import time
from pathlib import Path

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
        # #region agent log
        debug_log(
            "main.py:run_ingestion",
            "ingestion_check",
            {"needs_ingestion": needs_it, "reason": reason, "force": force},
            hypothesis_id="H5",
        )
        # #endregion
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
    # #region agent log
    debug_log("main.py:run_ingestion", "stage1_done", {"stage1_s": round(stage1_time, 2)}, hypothesis_id="H1")
    # #endregion
    print(f"✓ PDF extraction complete! (25%) - Time: {stage1_time:.1f}s")
    
    # Stage 2: Chunking (25-50%)
    stage2_start = time.time()
    print(f"\n[Stage 2/4] Creating semantic chunks... (25% - 50%)")
    process_files(DATA_PROCESSED, DATA_CHUNKS)
    stage2_time = time.time() - stage2_start
    # #region agent log
    debug_log("main.py:run_ingestion", "stage2_done", {"stage2_s": round(stage2_time, 2)}, hypothesis_id="H3")
    # #endregion
    print(f"✓ Chunking complete! (50%) - Time: {stage2_time:.1f}s")
    
    # Stage 3 & 4: Indexing (50-100%)
    stage3_start = time.time()
    print(f"\n[Stage 3-4/4] Generating embeddings and indexing... (50% - 100%)")
    index_documents(DATA_CHUNKS, str(VECTOR_DB_DIR), EMB_MODEL_NAME)
    stage3_time = time.time() - stage3_start
    # #region agent log
    debug_log("main.py:run_ingestion", "stage3_done", {"stage3_s": round(stage3_time, 2)}, hypothesis_id="H4")
    # #endregion
    print(f"✓ Indexing complete! (100%) - Time: {stage3_time:.1f}s")
    
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    
    print("\n" + "="*60)
    print(f"✓ INGESTION COMPLETE! Total time: {minutes}m {seconds}s")
    print("="*60 + "\n")
    
    # Clear retriever cache after re-indexing
    reset_retriever()

def rag_pipeline(query: str, top_k: int = 3, use_cache: bool = None, show_timing: bool = True):
    """RAG pipeline orchestrated by LangGraph (retrieve -> prompt -> generate)."""
    result = run_rag_graph(
        query,
        top_k=top_k,
        use_cache=use_cache,
        show_timing=show_timing,
    )
    return result.get("answer", "")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAG pipeline")
    parser.add_argument("--query", type=str, default=None, help="Single question to run non-interactively")
    parser.add_argument("--top_k", type=int, default=3, help="Number of chunks to retrieve")
    parser.add_argument("--force-ingestion", action="store_true", help="Force re-ingestion even if files are up to date")
    parser.add_argument("--no-timing", action="store_true", help="Disable timing information")
    parser.add_argument("--mcp", action="store_true", help="Run MCP server (stdio) instead of CLI")
    args = parser.parse_args()

    show_timing = not args.no_timing
    
    # Test generation backend connection
    from config.settings import USE_OLLAMA, USE_OPENAI, OPENAI_API_KEY, OPENAI_MODEL
    if USE_OPENAI and OPENAI_API_KEY:
        print(f"[Info] OpenAI is enabled (model: {OPENAI_MODEL})")
    elif USE_OLLAMA:
        from synthesis.local_generator import get_generator
        try:
            generator = get_generator()
            if hasattr(generator, 'use_ollama') and generator.use_ollama:
                print("[Info] Ollama is enabled and ready")
        except Exception as e:
            print(f"[Warning] Ollama setup issue: {e}")
    
    run_ingestion(force=args.force_ingestion)
    if args.query:
        print(f"\nProcessing: {args.query}")
        print("-" * 40)
        try:
            answer = rag_pipeline(args.query, top_k=args.top_k, show_timing=show_timing)
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
            answer = rag_pipeline(query, show_timing=show_timing)
            print(f"\nAnswer:\n{answer}")
            print("-" * 40)
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again with a different question.")
