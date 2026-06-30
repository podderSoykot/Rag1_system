from config.settings import USE_OPENAI, OPENAI_API_KEY, OPENAI_MODEL
from .local_generator import generate_answer as local_generate_answer
from .postprocessor import clean_answer, validate_answer_completeness


def generate_answer(prompt: str, query: str = None):
    """Generate answer using OpenAI, Ollama, or local model based on .env settings."""
    if USE_OPENAI and OPENAI_API_KEY:
        from .openai_generator import generate_answer as openai_generate_answer
        print(f"[Info] Using OpenAI model: {OPENAI_MODEL}")
        raw_answer = openai_generate_answer(prompt, query=query)
    else:
        # Pass query to enable dynamic continuation
        raw_answer = local_generate_answer(prompt, query=query)
    
    # Post-process the answer
    cleaned_answer = clean_answer(raw_answer)
    
    # Final validation (for display purposes)
    if query:
        is_complete, validation_msg = validate_answer_completeness(cleaned_answer, query)
        if not is_complete and len(cleaned_answer) < 50:
            # If answer is still too short after dynamic generation, return helpful message
            return f"Based on the retrieved documents, I found limited information. {cleaned_answer}"
    
    return cleaned_answer
