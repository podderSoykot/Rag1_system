from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import requests
import json
import time
from config.settings import USE_OLLAMA, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT

class LocalLLMGenerator:
    _instance = None
    _initialized = False
    
    def __new__(cls, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """Singleton pattern to ensure only one model instance exists"""
        if cls._instance is None:
            cls._instance = super(LocalLLMGenerator, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """Initialize model only once (singleton pattern)"""
        if self._initialized:
            return
        
        self.use_ollama = USE_OLLAMA
        self.ollama_url = OLLAMA_BASE_URL
        self.ollama_model = OLLAMA_MODEL
        self.ollama_timeout = OLLAMA_TIMEOUT
        self.model_name = model_name
        
        if self.use_ollama:
            print(f"Using Ollama for RAG: {self.ollama_model}")
            # Test Ollama connection on startup
            if self._test_ollama_connection():
                print(f"✓ Ollama connection verified")
            else:
                print(f"⚠ Warning: Ollama connection test failed. Generation may fail.")
            LocalLLMGenerator._initialized = True
            return
        
        print(f"Loading local model for RAG: {model_name}")
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
            print(f"Local RAG model loaded successfully: {model_name}")
            LocalLLMGenerator._initialized = True
        except Exception as e:
            print(f"Failed to load RAG model {model_name}: {e}")
            raise Exception(f"Model loading failed: {e}")
    def generate_answer(self, prompt: str, max_new_tokens: int = 250):
        try:
            if self.use_ollama:
                return self._generate_with_ollama(prompt, max_new_tokens)
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
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,  # Enable sampling for better quality
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,  # Increased to reduce repetition
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True  # Stop early if EOS token is generated
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
            print(f"RAG generation failed: {e}")
            return f"Based on the retrieved document context, I can provide insights on the topics covered. The system encountered an error during generation: {str(e)}"
    def _test_ollama_connection(self):
        """Test Ollama connection and model availability"""
        try:
            response = requests.get(
                f"{self.ollama_url}/api/tags",
                timeout=5
            )
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                if self.ollama_model in model_names:
                    return True
                else:
                    print(f"⚠ Model '{self.ollama_model}' not found. Available models: {', '.join(model_names[:3])}")
                    return False
            return False
        except Exception as e:
            print(f"⚠ Ollama connection test failed: {e}")
            return False
    
    def _estimate_tokens(self, text: str):
        """Rough estimate of tokens (1 token ≈ 4 characters)"""
        return len(text) // 4
    
    def _generate_with_ollama(self, prompt: str, max_new_tokens: int = 250):
        """Generate answer using Ollama with improved API usage"""
        # Estimate prompt length and adjust num_predict
        prompt_tokens = self._estimate_tokens(prompt)
        # Use chat API for better responses
        use_chat_api = True
        
        # Determine num_predict based on query complexity
        is_complex = any(word in prompt.lower() for word in ['explain', 'describe', 'how', 'why', 'compare'])
        num_predict = max_new_tokens * 2 if is_complex else max_new_tokens
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                if use_chat_api:
                    # Use chat API format for better responses
                    messages = [
                        {
                            "role": "system",
                            "content": "You are a helpful AI assistant. Answer questions directly and concisely based on the provided documents."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                    
                    response = requests.post(
                        f"{self.ollama_url}/api/chat",
                        json={
                            "model": self.ollama_model,
                            "messages": messages,
                            "stream": False,
                            "options": {
                                "temperature": 0.7,
                                "top_p": 0.9,
                                "num_predict": num_predict,
                                "repeat_penalty": 1.2
                            }
                        },
                        timeout=self.ollama_timeout
                    )
                else:
                    # Fallback to generate API
                    response = requests.post(
                        f"{self.ollama_url}/api/generate",
                        json={
                            "model": self.ollama_model,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "temperature": 0.7,
                                "top_p": 0.9,
                                "num_predict": num_predict,
                                "repeat_penalty": 1.2
                            }
                        },
                        timeout=self.ollama_timeout
                    )
                
                response.raise_for_status()
                result = response.json()
                
                if use_chat_api:
                    answer = result.get("message", {}).get("content", "").strip()
                else:
                    answer = result.get("response", "").strip()
                
                if answer:
                    return answer
                else:
                    return "I couldn't generate a response. Please try again."
                    
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"[Retry {attempt + 1}/{max_retries}] Ollama request timed out, retrying...")
                    time.sleep(1)
                    continue
                else:
                    return f"Ollama request timed out after {self.ollama_timeout}s. The model may be processing a long response."
            except requests.exceptions.ConnectionError:
                return f"Could not connect to Ollama at {self.ollama_url}. Please ensure Ollama is running."
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[Retry {attempt + 1}/{max_retries}] Ollama generation failed: {e}")
                    time.sleep(1)
                    continue
                else:
                    print(f"Ollama generation failed: {e}")
                    return f"Ollama generation failed: {str(e)}"
        
        return "Failed to generate response after retries."
# Global singleton instance
_generator_instance = None

def get_generator(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Get or create the singleton generator instance"""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = LocalLLMGenerator(model_name)
    return _generator_instance

def generate_answer(prompt: str):
    """Generate answer using cached model instance"""
    try:
        generator = get_generator()
        return generator.generate_answer(prompt)
    except Exception as e:
        return f"System error: {str(e)}. Please check the model configuration."
