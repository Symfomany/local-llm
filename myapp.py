import os
import requests
import json

# Récupérer les variables d'environnement
# vllm_api_url = os.environ.get(
#     "VLLM_API_URL", "http://localhost:8000/v1/completions")
vllm_api_url = "http://host.docker.internal:8000/v1/completions"

hf_api_key = os.environ.get("HUGGING_FACE_API_KEY")

if not hf_api_key:
    raise ValueError(
        "La clé API Hugging Face n'est pas définie dans les variables d'environnement.")

# Récupérer les paramètres d'échantillonnage depuis les variables d'environnement
temperature = float(os.environ.get("TEMPERATURE", 0.2))
top_p = float(os.environ.get("TOP_P", 0.95))
max_tokens = int(os.environ.get("MAX_TOKENS", 500))

# Définir les paramètres de la requête
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {hf_api_key}"
}

data = {
    "model": "Qwen/Qwen2.5-Coder-1.5B",
    "prompt": "Écrivez une routine en Go",
    "max_tokens": max_tokens,
    "temperature": temperature,
    "top_p": top_p
}

# Envoyer la requête à l'API vLLM
response = requests.post(vllm_api_url, headers=headers,
                         json=data, timeout=30, retries=3)

# Vérifier si la requête a réussi
if response.status_code == 200:
    result = response.json()
    generated_text = result['choices'][0]['text']
    print(generated_text)
else:
    print(f"Erreur lors de la requête : {response.status_code}")
    print(response.text)
