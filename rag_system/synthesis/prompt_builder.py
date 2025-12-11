# rag_system/synthesis/prompt_builder.py

def build_prompt(query: str, docs: list):
    """Build an enhanced prompt with retrieved context and user question for RAG"""
    # Show user what context was found
    print(f"[Info] Found {len(docs)} relevant document chunks:")
    for i, doc in enumerate(docs[:3], 1):  # Show first 3 chunks
        preview = doc[:100] + "..." if len(doc) > 100 else doc
        print(f"   {i}. {preview}")
    
    if len(docs) > 3:
        print(f"   ... and {len(docs) - 3} more chunks")
    
    # Format context with numbered chunks for better reference
    context_parts = []
    for i, doc in enumerate(docs, 1):
        # Clean up document text
        doc_clean = doc.strip()
        if doc_clean:
            context_parts.append(f"[Document {i}]\n{doc_clean}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Determine if query is complex (multiple parts, "how", "why", "explain")
    is_complex = any(word in query.lower() for word in ['how', 'why', 'explain', 'describe', 'compare', 'difference'])
    
    # Create enhanced prompt with better structure
    if is_complex:
        # Simplified but still structured for complex questions
        prompt = f"""Based on the documents below, provide a detailed answer to the question.

Documents:
{context}

Question: {query}

Answer (be detailed and use information from the documents):"""
    else:
        # Simplified prompt for better TinyLlama performance
        prompt = f"""Based on the following documents, answer the question directly and concisely.

Documents:
{context}

Question: {query}

Answer (use only information from the documents above):"""
    
    return prompt
