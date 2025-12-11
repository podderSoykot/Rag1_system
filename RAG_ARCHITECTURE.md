# RAG System Architecture - How It Works

## Answer Generation Flow

```
User Query
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

## Components Breakdown

### Pinecone (Vector Database)
- **Purpose**: Store and search document embeddings
- **Used for**: Finding relevant document chunks
- **NOT used for**: Generating answers

### Ollama (LLM)
- **Purpose**: Generate answers from context
- **Used for**: Answer generation
- **Input**: Prompt with retrieved chunks + query
- **Output**: Generated answer text

## Configuration

In your `.env` file:
- `USE_PINECONE=true` → Use Pinecone for retrieval (faster, cloud-based)
- `USE_OLLAMA=true` → Use Ollama for generation (faster than TinyLlama)

## Summary

**Pinecone = Retrieval (finding relevant info)**
**Ollama = Generation (creating the answer)**

Both work together:
1. Pinecone finds what's relevant
2. Ollama generates the answer from that relevant content

