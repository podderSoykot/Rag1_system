import os

# Define folder structure
folders = [
    "rag_system/data/raw",
    "rag_system/data/processed",
    "rag_system/data/embeddings",
    "rag_system/ingestion",
    "rag_system/retrieval",
    "rag_system/synthesis",
    "rag_system/config",
]

# Define files with starter content
files = {
    "rag_system/main.py": "# Entry point for RAG pipeline\n\nif __name__ == '__main__':\n    print('Run ingestion → retrieval → synthesis')\n",
    "rag_system/requirements.txt": "# Add your dependencies here\nopenai\nlangchain\nfaiss-cpu\n",
    "rag_system/README.md": "# RAG System\n\nRetrieval-Augmented Generation pipeline with ingestion, retrieval, and synthesis modules.\n",
    "rag_system/config/settings.py": "# Configurations for vector DB, API keys, and models\nVECTOR_DB_PATH = 'rag_system/data/embeddings/'\n",
    
    # Ingestion
    "rag_system/ingestion/__init__.py": "",
    "rag_system/ingestion/data_loader.py": "# Load raw documents from data/raw\n",
    "rag_system/ingestion/chunker.py": "# Split documents into smaller chunks\n",
    "rag_system/ingestion/embedder.py": "# Generate embeddings for document chunks\n",
    "rag_system/ingestion/indexer.py": "# Push embeddings into vector DB\n",
    
    # Retrieval
    "rag_system/retrieval/__init__.py": "",
    "rag_system/retrieval/retriever.py": "# Retrieve top-k results from vector DB\n",
    "rag_system/retrieval/query_expansion.py": "# Optional: expand queries for better recall\n",
    
    # Synthesis
    "rag_system/synthesis/__init__.py": "",
    "rag_system/synthesis/prompt_builder.py": "# Build final prompt from query + retrieved context\n",
    "rag_system/synthesis/generator.py": "# Call LLM to generate answers\n",
    "rag_system/synthesis/postprocessor.py": "# Optional: clean and format output\n",
}

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create files with starter content
for filepath, content in files.items():
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

print("✅ RAG system project structure created successfully!")
