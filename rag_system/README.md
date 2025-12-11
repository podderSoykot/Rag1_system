# RAG System

Retrieval-Augmented Generation (RAG) pipeline for document-based question answering using local LLMs (Ollama) with Pinecone vector database and TF-IDF retrieval.

## Features
- **Document Ingestion:** PDF to text conversion and chunking
- **Indexing:** Pinecone vector database (cloud-based, scalable) with TF-IDF fallback
- **Retrieval:** Hybrid search combining semantic (Pinecone) and keyword (TF-IDF) retrieval
- **Synthesis:** Generate answers using a local LLM (Ollama or TinyLlama)
- **Configurable:** Easily switch between Pinecone, FAISS, or local storage

## Directory Structure
```
rag_system/
├── config/           # Configuration files
│   └── settings.py
├── data/             # Data storage (raw, processed, embeddings)
├── ingestion/        # Data loading, chunking, and indexing
├── retrieval/        # Retriever logic (TF-IDF)
├── synthesis/        # Prompt building and answer generation
├── main.py           # Main CLI entry point
├── requirements.txt  # Python dependencies
```

## Installation
1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd rag_system
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up environment variables:**
   - Create a `.env` file in the root directory with your configuration:
   ```bash
   # Pinecone Configuration (recommended)
   USE_PINECONE=true
   PINECONE_API_KEY=your-pinecone-api-key-here
   PINECONE_INDEX_NAME=rag-system
   PINECONE_ENVIRONMENT=us-east-1
   
   # Optional: Other settings
   USE_HYBRID_SEARCH=true
   CHUNK_SIZE=800
   CHUNK_OVERLAP=100
   ```
   - Get your Pinecone API key from: https://www.pinecone.io/
   - If you don't want to use Pinecone, set `USE_PINECONE=false` to use local FAISS storage

## Usage
### Command Line
Run the full pipeline (ingestion, retrieval, synthesis):
```bash
python main.py --query "What is the main topic of the document?" --top_k 3
```
- `--query`: (optional) Run a single question non-interactively
- `--top_k`: (optional) Number of chunks to retrieve (default: 3)

If no query is provided, the system will enter interactive mode for Q&A.

### Programmatic Usage
You can import and use the pipeline in your own scripts:
```python
from synthesis.local_generator import generate_answer
answer = generate_answer("Your question here")
print(answer)
```

## Configuration
- All settings are managed in `config/settings.py` and via environment variables (`.env` file).
- Supports only local LLMs (Ollama, TinyLlama). No OpenAI API key is required.
- Change model or Ollama settings in your `.env` file as needed.

## Requirements
See `requirements.txt` for all dependencies. Key packages:
- transformers
- torch
- pypdf
- tqdm
- python-dotenv
- pinecone-client (for Pinecone vector database)
- faiss-cpu (fallback for local vector storage)

## Contact
For questions or support, please contact:
- [Soykot Podder]
- [diptopodder95@gmail.com]

---
Feel free to customize this README for your specific use case or deployment environment.
