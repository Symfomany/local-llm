import pytest
from fastapi.testclient import TestClient
import requests
import json
import os 
from dotenv import load_dotenv
import random

# Configuration ---------------------------------------------------------------
load_dotenv()


# URL de base de votre API
BASE_URL = os.getenv('API_URL', "http://localhost:8080")
MODEL_NAME = os.getenv('MODEL_NAME', "Qwen/Qwen2.5-Coder-1.5B-Instruct-AWQ")


# Exemple de données pour les requêtes
CHAT_COMPLETION_REQUEST = {
    "model":MODEL_NAME,
    "messages": [{"role": "user", "content": "Bonjour, comment allez-vous ?"}],
    "max_tokens": 100
}

COMPLETION_REQUEST = {
    "model":MODEL_NAME,
    "prompt": "Bonjour, comment allez-vous ?",
    "max_tokens": 100
}

# Requêtes avec des paramètres incorrects
INVALID_CHAT_COMPLETION_REQUEST = {
    "model": MODEL_NAME,
    # Manque de messages
}

INVALID_COMPLETION_REQUEST = {
    "model": MODEL_NAME,
    # Manque de prompt
}

def test_health_check():
    response = requests.get(
        f"{BASE_URL}/health", json=CHAT_COMPLETION_REQUEST)
    assert response.status_code == 200
    assert response.text == "OK"
    
def test_chat_completion():
    response = requests.post(
        f"{BASE_URL}/v1/chat/completions", json=CHAT_COMPLETION_REQUEST)
    assert response.status_code == 200
    assert "choices" in response.json()



def test_completion():
    response = requests.post(
        f"{BASE_URL}/v1/completions", json=COMPLETION_REQUEST)
    assert response.status_code == 200
    assert "choices" in response.json()


# Test pour vérifier que l'API retourne une erreur si le modèle n'est pas supporté


def test_unsupported_model():
    unsupported_request = {"model": "UnsupportedModel"}
    response = requests.post(
        f"{BASE_URL}/v1/completions", json=unsupported_request)
    assert response.status_code == 422  # ou le code d'erreur attendu

# Test pour vérifier que l'API gère correctement les requêtes invalides


def test_invalid_request():
    invalid_request = {}  # Requête vide ou avec des champs manquants
    response = requests.post(
        f"{BASE_URL}/v1/completions", json=invalid_request)
    assert response.status_code == 422  # ou le code d'erreur attendu


# Test pour vérifier que l'API gère correctement les requêtes invalides
def test_invalid_chat_completion_request():
    response = requests.post(
        f"{BASE_URL}/v1/chat/completions", json=INVALID_CHAT_COMPLETION_REQUEST)
    assert response.status_code == 422


def test_invalid_completion_request():
    response = requests.post(
        f"{BASE_URL}/v1/completions", json=INVALID_COMPLETION_REQUEST)
    assert response.status_code == 422


# Test pour vérifier que l'API gère correctement les requêtes vides


def test_empty_request():
    response = requests.post(f"{BASE_URL}/v1/completions", json={})
    assert response.status_code == 422
