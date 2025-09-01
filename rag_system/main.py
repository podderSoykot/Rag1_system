# rag_system/main.py

from ingestion.data_loader import process_pdfs
from ingestion.chunker import process_files
from ingestion.indexer import index_documents
from retrieval.retriever import Retriever
from synthesis.prompt_builder import build_prompt
from synthesis.generator import generate_answer
from config.settings import DATA_RAW, DATA_PROCESSED, DATA_CHUNKS, VECTOR_DB_DIR, EMB_MODEL_NAME

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
    run_ingestion()

    query = "Summarize the main concepts in chapman_Machine.pdf"
    print("🔍 Query:", query)
    print("💡 Answer:", rag_pipeline(query))
