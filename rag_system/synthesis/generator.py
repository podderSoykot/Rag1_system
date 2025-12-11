import os
from .local_generator import generate_answer as local_generate_answer
from .postprocessor import clean_answer, validate_answer_completeness


def generate_answer(prompt: str, query: str = None):
    """Generate answer using local model or Ollama, with post-processing."""
    raw_answer = local_generate_answer(prompt)
    
    # Post-process the answer
    cleaned_answer = clean_answer(raw_answer)
    
    # Validate completeness if query is provided
    if query:
        is_complete, validation_msg = validate_answer_completeness(cleaned_answer, query)
        if not is_complete and len(cleaned_answer) < 50:
            # If answer is too short, return a more helpful message
            return f"Based on the retrieved documents, I found limited information. {cleaned_answer}"
    
    return cleaned_answer
