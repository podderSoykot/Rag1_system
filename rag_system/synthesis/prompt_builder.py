# rag_system/synthesis/prompt_builder.py

def build_prompt(query: str, docs: list):
    context = "\n\n".join(docs)
    return f"Context:\n{context}\n\nQuestion:\n{query}\nAnswer:"
