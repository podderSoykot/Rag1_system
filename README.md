# RAG System - Retrieval-Augmented Generation Pipeline

A comprehensive Retrieval-Augmented Generation (RAG) system that processes PDF documents, creates searchable vector indexes, and generates contextual answers using local LLMs (Ollama) with Pinecone vector database and TF-IDF retrieval.

## 📋 Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This RAG system combines:
- **Pinecone** (cloud vector database) for fast semantic search
- **Ollama** (local LLM) for fast answer generation
- **TF-IDF** for keyword-based retrieval
- **Hybrid search** combining semantic and keyword matching

**Key Point**: Pinecone is used for **finding** relevant information, while Ollama is used for **generating** the answer.

## 🔄 How It Works

The RAG system uses a two-stage process:

```
User Question
    ↓
1. RETRIEVAL (Pinecone/FAISS)
   - Query is embedded using sentence-transformers
   - Pinecone searches for similar document chunks
   - Returns top-k most relevant chunks
    ↓
2. PROMPT BUILDING
   - Relevant chunks + user query → formatted prompt
    ↓
3. GENERATION (Ollama/TinyLlama)
   - Prompt sent to Ollama API
   - Ollama generates answer based on context
    ↓
4. POST-PROCESSING
   - Clean answer (remove repetition, format)
   - Return final answer
```

### Component Roles

**Pinecone (Vector Database)**
- **Purpose**: Store and search document embeddings
- **Used for**: Finding relevant document chunks
- **NOT used for**: Generating answers

**Ollama (LLM)**
- **Purpose**: Generate answers from context
- **Used for**: Answer generation
- **Input**: Prompt with retrieved chunks + query
- **Output**: Generated answer text

## ✨ Features

- **Document Ingestion**: PDF to text conversion and intelligent chunking with structured content preservation
- **Indexing**: Pinecone vector database (cloud-based, scalable) with FAISS fallback
- **Retrieval**: Hybrid search with parallelized semantic (Pinecone) and keyword (TF-IDF) retrieval, with result caching
- **Synthesis**: Generate answers using Ollama (default, fast) or local LLM (TinyLlama) with answer post-processing
- **Performance**: Optimized for speed with parallel searches, caching, and performance monitoring
- **Quality**: Enhanced prompts and answer post-processing for better, more direct responses
- **Monitoring**: Detailed timing information and answer quality metrics

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Ollama (for best performance) - [Download](https://ollama.ai)
- Pinecone account (optional, for cloud vector storage) - [Sign up](https://www.pinecone.io/)

### Step 1: Clone and Install Dependencies

```bash
# Clone the repository
git clone <your-repo-url>
cd Rag1_system

# Install Python dependencies
pip install -r rag_system/requirements.txt
```

### Step 2: Install Ollama (Recommended)

**Windows:**
1. Download from https://ollama.com/download
2. Run `OllamaSetup.exe`
3. Follow the installation wizard
4. Ollama will start automatically

**After installation, pull a model:**
```bash
ollama pull llama3.2:3b
```

**Verify installation:**
```bash
ollama list
```

### Step 3: Set Up Environment Variables

Create a `.env` file in the root directory (`Rag1_system/.env`):

```bash
# Ollama Configuration (default, recommended for speed)
USE_OLLAMA=true
OLLAMA_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT=120

# Pinecone Configuration (optional, for cloud vector storage)
USE_PINECONE=true
PINECONE_API_KEY=your-pinecone-api-key-here
PINECONE_INDEX_NAME=rag-system
PINECONE_ENVIRONMENT=us-east-1

# Search Configuration
USE_HYBRID_SEARCH=true
SEMANTIC_WEIGHT=0.7
TFIDF_WEIGHT=0.3

# Chunking Configuration
CHUNK_SIZE=800
CHUNK_OVERLAP=100

# Performance Settings
CACHE_ENABLED=true
CACHE_SIZE=100
EMBEDDING_BATCH_SIZE=128
```

**Get your API keys:**
- Pinecone API key: https://www.pinecone.io/
- If you don't want to use Pinecone, set `USE_PINECONE=false` to use local FAISS storage
- If you don't want to use Ollama, set `USE_OLLAMA=false` to use local TinyLlama (slower)

## 🏃 Quick Start

1. **Place your PDF documents** in `rag_system/data/raw/`

2. **Run the system:**
   ```bash
   cd rag_system
   python main.py
   ```

3. **The system will:**
   - Automatically process PDFs (if not already processed)
   - Create embeddings and index them in Pinecone/FAISS
   - Enter interactive mode for Q&A

4. **Ask questions:**
   ```
   Enter your question (or 'quit' to exit): What are the chapters in this book?
   ```

## 📖 Usage

### Command Line Interface

**Single Query (Non-Interactive):**
```bash
cd rag_system
python main.py --query "What are the chapters in this book?" --top_k 3
```

**Interactive Mode:**
```bash
cd rag_system
python main.py
```

**Command Line Options:**
- `--query`: Run a single question non-interactively
- `--top_k`: Number of chunks to retrieve (default: 3)
- `--force-ingestion`: Force re-processing of all documents
- `--no-timing`: Disable timing information

### Programmatic Usage

```python
from rag_system.main import rag_pipeline

# Run a query
answer = rag_pipeline("What is the main topic?", top_k=3)
print(answer)
```

### Re-indexing Documents

If you add new PDFs or want to re-process existing ones:

```bash
cd rag_system
python main.py --force-ingestion
```

## ⚙️ Configuration

All settings are managed in `rag_system/config/settings.py` and via environment variables (`.env` file).

### Key Configuration Options

**Retrieval (Pinecone/FAISS):**
- `USE_PINECONE=true` - Use Pinecone cloud vector database (recommended)
- `PINECONE_API_KEY` - Your Pinecone API key
- Falls back to FAISS if Pinecone is unavailable

**Generation (Ollama/TinyLlama):**
- `USE_OLLAMA=true` - Use Ollama for answer generation (default, recommended)
- `OLLAMA_MODEL=llama3.2:3b` - Model to use
- Falls back to TinyLlama if Ollama is unavailable

**Chunking:**
- `CHUNK_SIZE=800` - Size of text chunks (characters)
- `CHUNK_OVERLAP=100` - Overlap between chunks for better context

**Performance:**
- `CACHE_ENABLED=true` - Enable result caching
- `USE_HYBRID_SEARCH=true` - Combine semantic and keyword search

## 🏗️ Architecture

### Directory Structure

```
Rag1_system/
├── rag_system/
│   ├── config/              # Configuration files
│   │   └── settings.py
│   ├── data/                # Data storage
│   │   ├── raw/             # Original PDF files
│   │   ├── processed/       # Extracted text files
│   │   └── embeddings/      # Vector database files
│   ├── ingestion/          # Document processing
│   │   ├── data_loader.py  # PDF to text extraction
│   │   ├── chunker.py      # Intelligent text chunking
│   │   ├── embedder.py     # Generate embeddings
│   │   └── indexer.py      # Store in Pinecone/FAISS
│   ├── retrieval/          # Document search
│   │   ├── retriever.py    # Hybrid search (Pinecone + TF-IDF)
│   │   └── query_expansion.py
│   ├── synthesis/          # Answer generation
│   │   ├── prompt_builder.py
│   │   ├── generator.py    # Main generator interface
│   │   ├── local_generator.py  # Ollama/TinyLlama integration
│   │   └── postprocessor.py   # Answer cleaning
│   ├── main.py             # CLI entry point
│   └── requirements.txt    # Python dependencies
├── README.md               # This file
├── RAG_ARCHITECTURE.md    # Detailed architecture docs
└── INSTALL_OLLAMA.md      # Ollama installation guide
```

### Components

1. **Ingestion** (`rag_system/ingestion/`)
   - `data_loader.py`: PDF to text extraction using pypdf
   - `chunker.py`: Intelligent text chunking with sentence boundaries
   - `embedder.py`: Generate embeddings using sentence-transformers
   - `indexer.py`: Store embeddings in Pinecone/FAISS and create TF-IDF index

2. **Retrieval** (`rag_system/retrieval/`)
   - `retriever.py`: Hybrid search combining Pinecone (semantic) and TF-IDF (keyword)
   - `query_expansion.py`: Query enhancement for better recall

3. **Synthesis** (`rag_system/synthesis/`)
   - `prompt_builder.py`: Build context-aware prompts
   - `generator.py`: Main generator interface with post-processing
   - `local_generator.py`: Ollama API integration
   - `postprocessor.py`: Clean and format answers

### Data Flow

```
PDF Documents (data/raw/)
    ↓ [Ingestion]
Text Extraction → Chunking → Embeddings
    ↓
Pinecone Index / FAISS Index
    ↓ [Query Time]
User Question → Embed Query → Search Pinecone
    ↓
Retrieve Top-K Chunks → Build Prompt
    ↓
Send to Ollama → Generate Answer
    ↓
Post-process → Return Answer
```

## ⚡ Performance

### Speed Optimizations

- **Use Ollama**: 5-10x faster than local TinyLlama (default enabled)
- **Use Pinecone**: Cloud-based vector search is faster than local FAISS for large datasets
- **Parallel Retrieval**: Semantic and TF-IDF searches run in parallel when both enabled
- **Result Caching**: Similar queries are cached for instant responses

### Quality Improvements

- **Answer Post-processing**: Removes repetition, formats lists, cleans meta-phrases
- **Enhanced Prompts**: Optimized for direct answers based on query type (list, complex, simple)
- **Structured Content**: Better chunking preserves chapters, sections, and structured content
- **Validation**: Answer completeness validation with helpful feedback

### Monitoring

The system provides detailed timing information:
- **Retrieval time**: Breakdown of semantic (Pinecone) vs TF-IDF search
- **Prompt building time**: Usually <1ms
- **Generation time**: Ollama generation time
- **Total time**: End-to-end query processing time
- **Answer quality metrics**: Length, word count, completeness warnings

**Example Output:**
```
[Timing]
  Retrieval:  2053ms
  Prompt:     1ms
  Generation: 34124ms
  Total:      42352ms (42.35s)

[Answer Quality]
  Length: 526 chars, 89 words
```

## 🔧 Troubleshooting

### Ollama Issues

**Ollama not found:**
- Restart your terminal/PowerShell after installation
- Verify Ollama is running: `ollama list`
- Check if Ollama service is running (should start automatically)

**Model not found:**
- Pull the model: `ollama pull llama3.2:3b`
- Verify: `ollama list`
- Check model name in `.env` matches pulled model

### Pinecone Issues

**Connection failed:**
- Verify API key in `.env` file
- Check internet connection
- Verify index name exists in Pinecone dashboard
- System will fall back to FAISS if Pinecone fails

### Performance Issues

**Slow generation:**
- Ensure Ollama is enabled (`USE_OLLAMA=true`)
- Use a smaller model: `OLLAMA_MODEL=phi3:mini`
- Reduce `top_k` for faster retrieval

**Slow retrieval:**
- Use Pinecone for large datasets
- Reduce `CHUNK_SIZE` for faster processing
- Enable caching: `CACHE_ENABLED=true`

### Import Errors

**Module not found:**
- Ensure you're running from `rag_system/` directory
- Or install as package: `pip install -e .`
- Check Python path includes project root

## 📦 Requirements

See `rag_system/requirements.txt` for all dependencies. Key packages:

- `transformers>=4.41.0` - Hugging Face transformers
- `torch` - PyTorch for model inference
- `sentence-transformers` - Embedding generation
- `pinecone-client` - Pinecone vector database client
- `faiss-cpu` - Local vector search (fallback)
- `pypdf` - PDF text extraction
- `nltk>=3.8` - Natural language processing
- `python-dotenv` - Environment variable management
- `requests` - HTTP requests for Ollama API

## 📝 Notes

- **No OpenAI Required**: System uses only local LLMs (Ollama/TinyLlama)
- **Privacy**: All processing happens locally (except Pinecone storage)
- **Scalability**: Pinecone handles large document collections efficiently
- **Fallbacks**: System gracefully falls back to FAISS/TinyLlama if cloud services unavailable

## 📧 Contact

For questions or support:
- **Author**: Soykot Podder
- **Email**: diptopodder95@gmail.com

## 📄 License

[Add your license information here]

---

**Happy Querying!** 🚀
