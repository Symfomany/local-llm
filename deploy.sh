#!/bin/bash
set -a
source .env
set +a
# Variables
PROJECT_ID="decent-destiny-448418-p1"
REGION="europe-west4"
REPO_NAME="my-docker-repo"
SERVICE_NAME="vllm-server"

# Authentification
gcloud auth login
gcloud config set project $PROJECT_ID
gcloud config set run/region $REGION

# Créer un dépôt Artifact Registry (une seule fois)
# gcloud artifacts repositories create $REPO_NAME \
#     --repository-format=docker \
#     --location=$REGION || echo "Repository already exists"

# Construire et pousser l'image Docker
docker build -t $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$SERVICE_NAME .
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$SERVICE_NAME

# Déployer sur Cloud Run
gcloud  beta run deploy $SERVICE_NAME \
    --image=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$SERVICE_NAME \
    --platform=managed \
    --allow-unauthenticated \
    --cpu=4 \
    --memory 16Gi \
    --min-instances=1 \
    --max-instances=2 \
    --set-env-vars=HUGGING_FACE_API_KEY=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,MODEL_NAME=Qwen/Qwen2.5-Coder-1.5B
