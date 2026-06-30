from openai import OpenAI

from config.settings import OPENAI_API_KEY, OPENAI_MODEL, MAX_ANSWER_TOKENS
from synthesis.prompt_builder import RAG_SYSTEM_INSTRUCTIONS

_client = None


def get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def generate_answer(prompt: str, query: str = None, system_instructions: str = None, max_tokens: int = None):
    try:
        client = get_client()
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_instructions or RAG_SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens or MAX_ANSWER_TOKENS,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI generation failed: {e}")
        return f"OpenAI generation failed: {str(e)}"
