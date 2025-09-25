# RAG System

Retrieval-Augmented Generation (RAG) pipeline for document-based question answering using local LLMs and TF-IDF retrieval.

## Features
- **Document Ingestion:** PDF to text conversion and chunking
- **Indexing:** Simple TF-IDF vector store for fast retrieval
- **Retrieval:** Query relevant document chunks using TF-IDF
- **Synthesis:** Generate answers using a local LLM (TinyLlama or Ollama)
- **Configurable:** Easily switch between local and API-based LLMs

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
   - Copy or create a `.env` file in the root directory with your API keys and settings (see `config/settings.py` for required variables).

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
- Supports both local LLMs (TinyLlama) and Ollama API.
- Change model or API settings in your `.env` file as needed.

## Requirements
See `requirements.txt` for all dependencies. Key packages:
- transformers
- torch
- openai
- pypdf
- tqdm
- python-dotenv

## Contact
For questions or support, please contact:
- [Your Name or Team]
- [Your Email or GitHub]

---
Feel free to customize this README for your specific use case or deployment environment.
