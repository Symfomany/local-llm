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
