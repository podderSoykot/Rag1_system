import os
from .local_generator import generate_answer as local_generate_answer


def generate_answer(prompt: str):
    """Generate answer using local small model (TinyLlama)."""
    return local_generate_answer(prompt)
