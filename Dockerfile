# Utilisez l'image de base de votre service
FROM vllm/vllm-openai:latest


# WORKDIR /app

# # Copiez les fichiers nécessaires (si besoin)
# COPY . /app

ENV HOST=0.0.0.0
# EXPOSE 8080

ENV HUGGING_FACE_API_KEY=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
ENV TEMPERATURE=0.7
ENV TOP_P=0.95
ENV MAX_TOKENS=100
ENV MODEL_NAME=Qwen2.5-Coder-1.5B-Instruct-Q8_0.gguf
ENV PORT=8080

# Exposer le port (facultatif pour Cloud Run)
EXPOSE 8080


# Commande par défaut pour démarrer le serveur
# CMD ["python3", "-m", "vllm.entrypoints.openai.api_server", "--model", "Qwen/Qwen2.5-Coder-1.5B"]
CMD [ "vllm.entrypoints.openai.api_server", "--model", "${MODEL_NAME}"]