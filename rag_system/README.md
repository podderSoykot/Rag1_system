# RAG System

Retrieval-Augmented Generation (RAG) pipeline for document-based question answering using local LLMs (Ollama) with Pinecone vector database and TF-IDF retrieval.

## How It Works

The RAG system uses a two-stage process:

1. **Retrieval (Pinecone/FAISS)**: Finds relevant document chunks based on your question
2. **Generation (Ollama)**: Creates the answer using the retrieved context

```
User Question
    ↓
Pinecone searches → Finds relevant document chunks
    ↓
Chunks + Question → Sent to Ollama
    ↓
Ollama generates → Final Answer
```

**Key Point**: Pinecone is used for **finding** relevant information, while Ollama is used for **generating** the answer.

## Features
- **Document Ingestion:** PDF to text conversion and intelligent chunking with structured content preservation
- **Indexing:** Pinecone vector database (cloud-based, scalable) with TF-IDF fallback
- **Retrieval:** Hybrid search with parallelized semantic (Pinecone) and keyword (TF-IDF) retrieval, with result caching
- **Synthesis:** Generate answers using Ollama (default, fast) or local LLM (TinyLlama) with answer post-processing
- **Performance:** Optimized for speed with parallel searches, caching, and performance monitoring
- **Quality:** Enhanced prompts and answer post-processing for better, more direct responses

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
3. **Set up Ollama (Recommended for best performance):**
   ```bash
   # Install Ollama from https://ollama.ai
   # Pull a model (e.g., llama3.2:3b)
   ollama pull llama3.2:3b
   ```

4. **Set up environment variables:**
   - Create a `.env` file in the root directory with your configuration:
   ```bash
   # Ollama Configuration (default, recommended for speed)
   USE_OLLAMA=true
   OLLAMA_MODEL=llama3.2:3b
   OLLAMA_BASE_URL=http://localhost:11434
   
   # Pinecone Configuration (optional, for cloud vector storage)
   USE_PINECONE=true
   PINECONE_API_KEY=your-pinecone-api-key-here
   PINECONE_INDEX_NAME=rag-system
   PINECONE_ENVIRONMENT=us-east-1
   
   # Optional: Performance and quality settings
   USE_HYBRID_SEARCH=true
   CHUNK_SIZE=800
   CHUNK_OVERLAP=100
   ```
   - Get your Pinecone API key from: https://www.pinecone.io/
   - If you don't want to use Pinecone, set `USE_PINECONE=false` to use local FAISS storage
   - If you don't want to use Ollama, set `USE_OLLAMA=false` to use local TinyLlama (slower)

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

All settings are managed in `config/settings.py` and via environment variables (`.env` file).

### Key Components

**Retrieval (Pinecone/FAISS):**
- `USE_PINECONE=true` - Use Pinecone cloud vector database (recommended)
- `PINECONE_API_KEY` - Your Pinecone API key
- Falls back to FAISS if Pinecone is unavailable

**Generation (Ollama/TinyLlama):**
- `USE_OLLAMA=true` - Use Ollama for answer generation (default, recommended)
- `OLLAMA_MODEL=llama3.2:3b` - Model to use
- Falls back to TinyLlama if Ollama is unavailable

**Note:** Supports only local LLMs (Ollama, TinyLlama). No OpenAI API key is required.

## Requirements
See `requirements.txt` for all dependencies. Key packages:
- transformers
- torch
- pypdf
- tqdm
- python-dotenv
- pinecone-client (for Pinecone vector database)
- faiss-cpu (fallback for local vector storage)
- requests (for Ollama API)

**Note:** For best performance, install and run Ollama separately from https://ollama.ai

## Performance Tuning

### Speed Optimizations
- **Use Ollama**: 5-10x faster than local TinyLlama (default enabled)
- **Use Pinecone**: Cloud-based vector search is faster than local FAISS for large datasets
- **Parallel Retrieval**: Semantic and TF-IDF searches run in parallel when both enabled
- **Result Caching**: Similar queries are cached for instant responses

### Quality Improvements
- **Answer Post-processing**: Removes repetition, formats lists, cleans meta-phrases
- **Enhanced Prompts**: Optimized for direct answers based on query type
- **Structured Content**: Better chunking preserves chapters, sections, and structured content
- **Validation**: Answer completeness validation with helpful feedback

### Monitoring
The system provides detailed timing information:
- **Retrieval time**: Breakdown of semantic (Pinecone) vs TF-IDF search
- **Prompt building time**: Usually <1ms
- **Generation time**: Ollama generation time
- **Total time**: End-to-end query processing time
- **Answer quality metrics**: Length, word count, completeness warnings

## Architecture Overview

### Components

1. **Ingestion** (`ingestion/`)
   - `data_loader.py`: PDF to text extraction
   - `chunker.py`: Intelligent text chunking
   - `embedder.py`: Generate embeddings for chunks
   - `indexer.py`: Store embeddings in Pinecone/FAISS

2. **Retrieval** (`retrieval/`)
   - `retriever.py`: Hybrid search (Pinecone + TF-IDF)
   - `query_expansion.py`: Query enhancement

3. **Synthesis** (`synthesis/`)
   - `prompt_builder.py`: Build prompts with context
   - `generator.py`: Generate answers (Ollama/TinyLlama)
   - `postprocessor.py`: Clean and format answers
   - `local_generator.py`: Ollama API integration

### Data Flow

```
PDF Documents
    ↓ [Ingestion]
Text Chunks → Embeddings → Pinecone Index
    ↓ [Query Time]
User Question → Embed Query → Search Pinecone
    ↓
Retrieve Top-K Chunks → Build Prompt
    ↓
Send to Ollama → Generate Answer
    ↓
Post-process → Return Answer
```

## Contact
For questions or support, please contact:
- [Soykot Podder]
- [diptopodder95@gmail.com]

---
Feel free to customize this README for your specific use case or deployment environment.
