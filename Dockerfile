# Utilisez l'image de base de votre service
FROM vllm/vllm-openai:latest

# Copiez les fichiers nécessaires (si besoin)
# COPY . /app

# Commande par défaut pour démarrer le serveur
# CMD ["python", "-m", "vllm.entrypoints.openai.api_server", "--model", "Qwen/Qwen2.5-Coder-1.5B", "--host", "0.0.0.0", "--port", "8000"]
CMD ["vllm.entrypoints.openai.api_server", "--model", "${MODEL_NAME}", "--host", "0.0.0.0", "--port", "8000", "--trust-remote-code"]
