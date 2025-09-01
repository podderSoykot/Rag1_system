# rag_system/synthesis/generator.py

import os
from openai import OpenAI
from config.settings import OPENAI_API_KEY, OPENAI_MODEL

def generate_answer(prompt: str):
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()
