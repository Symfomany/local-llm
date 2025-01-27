import os
from vllm import LLM, SamplingParams

# Récupérer la clé API Hugging Face depuis les variables d'environnement
hf_api_key = os.environ.get("HUGGING_FACE_API_KEY")

# Initialiser le modèle
model = LLM(model="Qwen/Qwen2.5-Coder-1.5B", trust_remote_code=True)

# Définir les paramètres d'échantillonnage
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=100)

# Prompt pour générer du code
prompt = "Écrivez une fonction Python pour calculer la factorielle d'un nombre."

# Générer la réponse
outputs = model.generate([prompt], sampling_params)

# Afficher la réponse
for output in outputs:
    print(output.text)
