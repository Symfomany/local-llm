from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import os
import requests
import json
import emoji
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

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
prompt = (os.environ.get("PROMPT",  "Écrivez une routine en Go"))
model = (os.environ.get("MODEL_NAME",  "Qwen/Qwen2.5-Coder-1.5B"))

# Définir les paramètres de la requête
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {hf_api_key}"
}

data = {
    "model": model,
    "prompt": prompt,
    "max_tokens": max_tokens,
    "temperature": temperature,
    "top_p": top_p
}

# Initialisation de la console Rich
console = Console()

# Création du contenu avec emojis
content = Text()
content.append(emoji.emojize(":thermometer: Température: "))
content.append(f"{temperature}\n", style="bold cyan")
content.append(emoji.emojize(":chart_increasing: Top P: "))
content.append(f"{top_p}\n", style="bold magenta")
content.append(emoji.emojize(":abacus: Tokens max: "))
content.append(f"{max_tokens}\n", style="bold green")
content.append(emoji.emojize(":speech_balloon: Prompt: "))
content.append(f"{prompt}\n", style="bold yellow")
content.append(emoji.emojize(":robot: Modèle: "))
content.append(f"{model}\n", style="bold blue")

# Affichage dans un panneau
console.print(Panel(content, title="Paramètres de la requête", expand=False, border_style="bold"))

# Affichage des en-têtes et des données
console.print(emoji.emojize("\n:envelope: En-têtes de la requête:"))
console.print(headers, style="italic")

console.print(emoji.emojize("\n:package: Données de la requête:"))
console.print(data, style="italic")

# Envoyer la requête à l'API vLLM


def requests_retry_session(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504)):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


# Utilisation
try:
    response = requests_retry_session().post(
        vllm_api_url, headers=headers, json=data, timeout=30)
    response.raise_for_status()
    
    # Vérifier si la requête a réussi
    if response.status_code == 200:
        result = response.json()
        generated_text = result['choices'][0]['text']
        print(generated_text)
    else:
        print(f"Erreur lors de la requête : {response.status_code}")
        print(response.text)
        
except requests.exceptions.RequestException as e:
    print(f"Erreur lors de la requête : {e}")



