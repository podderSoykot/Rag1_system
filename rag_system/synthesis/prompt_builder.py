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
    
    # Determine query type for better prompting
    query_lower = query.lower()
    is_list_query = any(word in query_lower for word in ['list', 'what are', 'chapters', 'topics', 'sections'])
    is_complex = any(word in query_lower for word in ['how', 'why', 'explain', 'describe', 'compare', 'difference'])
    
    # Create optimized prompts based on query type
    if is_list_query:
        # For list queries, emphasize extracting items
        prompt = f"""Extract and list the information requested from these documents.

Documents:
{context}

Question: {query}

Provide a clear list or enumeration based on the documents:"""
    elif is_complex:
        # For complex questions, provide structured guidance
        prompt = f"""Answer the question using information from the documents below.

Documents:
{context}

Question: {query}

Provide a detailed answer with specific information from the documents:"""
    else:
        # Simple direct answer prompt
        prompt = f"""Answer the question using only the information in these documents.

Documents:
{context}

Question: {query}

Answer:"""
    
    return prompt
