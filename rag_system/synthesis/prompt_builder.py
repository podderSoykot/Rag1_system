# rag_system/synthesis/prompt_builder.py

def build_prompt(query: str, docs: list):
    """Build a prompt with retrieved context and user question for RAG"""
    # Show user what context was found
    print(f"📚 Found {len(docs)} relevant document chunks:")
    for i, doc in enumerate(docs[:3], 1):  # Show first 3 chunks
        preview = doc[:100] + "..." if len(doc) > 100 else doc
        print(f"   {i}. {preview}")
    
    if len(docs) > 3:
        print(f"   ... and {len(docs) - 3} more chunks")
    
    # Build a better RAG prompt
    context = "\n\n".join(docs)
    
    # Create a more structured prompt for better LLM understanding
    prompt = f"""You are a helpful AI assistant. Use the following document context to answer the user's question.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Answer based ONLY on the information provided in the context above
- If the context doesn't contain enough information, say "Based on the provided context, I cannot fully answer this question"
- Provide a clear, coherent response
- Use the specific details from the context when possible

ANSWER:"""
    
    return prompt
