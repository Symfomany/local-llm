# Utiliser l'image officielle vLLM
FROM vllm/vllm-openai:latest

# Définir le répertoire de travail
WORKDIR /app

# Copier le script Python dans le conteneur
COPY myapp.py .

# Installer les dépendances supplémentaires si nécessaire
RUN pip install requests

# Définir la variable d'environnement pour la clé Hugging Face
ENV HUGGING_FACE_API_KEY=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

# Exécuter le script Python au démarrage du conteneur
CMD ["python", "myapp.py"]
# CMD ["python", "-m", "vllm.entrypoints.api_server", "--model", "Qwen/Qwen2.5-Coder-1.5B"]