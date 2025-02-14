import os
from vllm import LLM, SamplingParams

# Récupérer la clé API Hugging Face depuis les variables d'environnement
hf_api_key = os.environ.get("HUGGING_FACE_API_KEY")

if not hf_api_key:
    raise ValueError("La clé API Hugging Face n'est pas définie dans les variables d'environnement.")

# Initialiser le modèle
model = LLM(model="Qwen/Qwen2.5-Coder-1.5B", trust_remote_code=True)

temperature = float(os.environ.get("TEMPERATURE", 0.7))
top_p = float(os.environ.get("TOP_P", 0.95))
max_tokens = int(os.environ.get("MAX_TOKENS", 100))

# Définir les paramètres d'échantillonnage
sampling_params = SamplingParams(
    temperature=temperature, top_p=top_p, max_tokens=max_tokens)
# Prompt pour générer du code
prompt = "Écrivez une fonction Python pour calculer la factorielle d'un nombre."

# Générer la réponse
outputs = model.generate([prompt], sampling_params)

# Afficher la réponse
for output in outputs:
    print(output.text)
    

try:
    # Lire un fichier texte
    with open("mon_fichier.txt", "r") as fichier:
        contenu = fichier.read()

    # Écrire dans un fichier texte
    with open("nouveau_fichier.txt", "w") as fichier:
fichier.write("Ceci est une nouvelle ligne de texte.\n")

    # Lire un fichier binaire
    with open("image.jpg", "rb") as fichier:
        donnees = fichier.read()

except FileNotFoundError:
    print("Un ou plusieurs fichiers n'ont pas été trouvés.")
except Exception as e:
    print(f"Une erreur s'est produite: {e}")
