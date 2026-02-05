import requests
from .base import BaseLLM

class HuggingFaceLLM(BaseLLM):
    def __init__(self, api_token: str, model: str):
        self.token = api_token
        self.model = model
        self.api_url = f"https://api-inference.huggingface.co/models/{model}"

    def generate(self, prompt: str) -> str:
        headers = {"Authorization": f"Bearer {self.token}"}
        
        # For T5 models, use simpler parameters
        if "t5" in self.model.lower():
            payload = {"inputs": prompt}
        else:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 300,
                    "temperature": 0.7,
                    "return_full_text": False
                }
            }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "No response")
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"