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
        prompt = f"""You are an expert AI assistant specialized in answering questions based on provided documents. Your task is to provide accurate, detailed, and well-structured answers.

RELEVANT DOCUMENTS:
{context}

USER QUESTION: {query}

ANSWERING GUIDELINES:
1. **Accuracy**: Base your answer ONLY on the information provided in the documents above. Do not use external knowledge.
2. **Completeness**: Provide a comprehensive answer that addresses all aspects of the question.
3. **Structure**: Organize your answer logically:
   - Start with a direct answer to the question
   - Provide supporting details and explanations
   - Include relevant examples or specifics from the documents when available
4. **Citations**: When referencing specific information, mention which document it came from (e.g., "According to Document 1..." or "As stated in Document 2...").
5. **Uncertainty**: If the documents do not contain sufficient information to fully answer the question, clearly state what information is missing or what aspects cannot be answered.

EXAMPLE FORMAT:
Question: "What is the main concept?"
Answer: "The main concept is [direct answer]. According to Document 1, [supporting detail]. Additionally, Document 2 explains that [additional detail]."

Now, provide your answer to the user's question:

ANSWER:"""
    else:
        prompt = f"""You are a helpful AI assistant. Answer the user's question using ONLY the information provided in the documents below.

DOCUMENTS:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Provide a clear, direct answer based on the documents above
- If you reference specific information, mention the document number (e.g., "Document 1 states...")
- If the documents don't contain enough information, say: "Based on the provided documents, I cannot fully answer this question. The documents do not contain sufficient information about [missing aspect]."
- Be concise but complete
- Use specific details and examples from the documents when available

ANSWER:"""
    
    return prompt
