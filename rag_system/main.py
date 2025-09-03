# rag_system/main.py

from ingestion.data_loader import process_pdfs
from ingestion.chunker import process_files
from ingestion.indexer import index_documents
from retrieval.retriever import Retriever
from synthesis.prompt_builder import build_prompt
from synthesis.generator import generate_answer
from config.settings import DATA_RAW, DATA_PROCESSED, DATA_CHUNKS, VECTOR_DB_DIR, EMB_MODEL_NAME
import argparse

def run_ingestion():
    print("📥 Ingestion started...")
    process_pdfs(DATA_RAW, DATA_PROCESSED)
    process_files(DATA_PROCESSED, DATA_CHUNKS)
    index_documents(DATA_CHUNKS, str(VECTOR_DB_DIR), EMB_MODEL_NAME)

def rag_pipeline(query: str, top_k: int = 3):
    retriever = Retriever(str(VECTOR_DB_DIR), EMB_MODEL_NAME)
    docs = retriever.search(query, top_k=top_k)
    prompt = build_prompt(query, docs)
    return generate_answer(prompt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAG pipeline")
    parser.add_argument("--query", type=str, default=None, help="Single question to run non-interactively")
    parser.add_argument("--top_k", type=int, default=3, help="Number of chunks to retrieve")
    args = parser.parse_args()

    run_ingestion()

    if args.query:
        print(f"\n🔄 Processing: {args.query}")
        print("-" * 40)
        try:
            answer = rag_pipeline(args.query, top_k=args.top_k)
            print(f"\n💡 Answer:\n{answer}")
            print("-" * 40)
        except Exception as e:
            print(f"\n❌ Error: {e}")
        raise SystemExit(0)

    print("\n" + "="*60)
    print("🤖 RAG System Ready! Ask questions about your documents.")
    print("="*60)
    
    while True:
        try:
            # Get user input
            query = input("\n🔍 Enter your question (or 'quit' to exit): ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                print("❌ Please enter a question.")
                continue
            
            print(f"\n🔄 Processing: {query}")
            print("-" * 40)
            
            # Get answer from RAG pipeline
            answer = rag_pipeline(query)
            print(f"\n💡 Answer:\n{answer}")
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("🔄 Please try again with a different question.")
