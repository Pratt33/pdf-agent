import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"

def ask_ollama(prompt):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    return response.json()["response"]
