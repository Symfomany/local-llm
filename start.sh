#!/bin/bash

# Démarrer le service vLLM en arrière-plan
# vllm serve "google/gemma-2b" &

# Attendre un court instant que vLLM démarre
# sleep 5


# Démarrer le service Ollama
ollama serve &

# Attendre un court instant que le service démarre
sleep 5

echo "Starting FastAPI application..."
exec uvicorn main:app --host 0.0.0.0 --port 8080