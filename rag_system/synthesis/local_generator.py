from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import requests
import json
from config.settings import USE_OLLAMA, OLLAMA_BASE_URL, OLLAMA_MODEL

class LocalLLMGenerator:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.use_ollama = USE_OLLAMA
        self.ollama_url = OLLAMA_BASE_URL
        self.ollama_model = OLLAMA_MODEL
        if self.use_ollama:
            print(f"🔄 Using Ollama for RAG: {self.ollama_model}")
            print(f"✅ Ollama RAG ready: {self.ollama_model}")
            return
        print(f"🔄 Loading local model for RAG: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if device == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype
            ).to(device)
            self.device = device
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"✅ Local RAG model loaded successfully: {model_name}")
        except Exception as e:
            print(f"❌ Failed to load RAG model {model_name}: {e}")
            raise Exception(f"Model loading failed: {e}")

    def generate_answer(self, prompt: str, max_new_tokens: int = 200):
        try:
            if self.use_ollama:
                return self._generate_with_ollama(prompt)
            chat_supported = hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None
            if chat_supported:
                chat_messages = [
                    {"role": "system", "content": "You are a helpful AI assistant for question answering over provided documents."},
                    {"role": "user", "content": prompt}
                ]
                rendered_prompt = self.tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                rendered_prompt = prompt
            tokenized = self.tokenizer(
                rendered_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=False
            )
            input_ids = tokenized["input_ids"].to(self.device)
            attention_mask = tokenized.get("attention_mask", torch.ones_like(input_ids)).to(self.device)
            input_length = input_ids.shape[1]
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                    temperature=0.3,
                    top_p=0.9,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            generated_ids = outputs[0][input_length:]
            full_response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            generated_part = full_response.strip()
            if generated_part:
                lines = generated_part.split('\n')
                clean_lines = []
                for line in lines:
                    line = line.strip()
                    if (line and 
                        not line.startswith('CONTEXT:') and 
                        not line.startswith('QUESTION:') and 
                        not line.startswith('INSTRUCTIONS:') and
                        not line.startswith('ANSWER:') and
                        not line.startswith('Context:') and
                        not line.startswith('Question:')):
                        clean_lines.append(line)
                if clean_lines:
                    clean_response = ' '.join(clean_lines).strip()
                    clean_response = ' '.join(clean_response.split())
                    return clean_response
                else:
                    return "Based on the retrieved documents, I cannot provide a specific answer to your question. The context may not contain enough relevant information."
            else:
                return "Based on the retrieved documents, I cannot provide a specific answer to your question. The context may not contain enough relevant information."
        except Exception as e:
            print(f"❌ RAG generation failed: {e}")
            return f"Based on the retrieved document context, I can provide insights on the topics covered. The system encountered an error during generation: {str(e)}"

    def _generate_with_ollama(self, prompt: str):
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "num_predict": 200
                    }
                },
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except Exception as e:
            print(f"❌ Ollama generation failed: {e}")
            return f"Ollama generation failed: {str(e)}"

def generate_answer(prompt: str):
    try:
        generator = LocalLLMGenerator()
        return generator.generate_answer(prompt)
    except Exception as e:
        return f"System error: {str(e)}. Please check the model configuration."
