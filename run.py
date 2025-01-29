from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os
import time
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import httpx


# Strategy of request Retry

def requests_retry_session(
    retries=5,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504),
    session=None,
    timeout=30
):
    session = session or requests.Session()
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

# from pydantic import  SettingsConfigDict
# from pydantic_settings import BaseSettings

app = FastAPI()


# class Settings(BaseSettings):
#     VLLM_SERVER_URL: str
#     model_config = SettingsConfigDict(
#         env_file=".env",
#         env_file_encoding='utf-8',
#         ignore_env_file_vars=True
#     )
#     class Config:
#         env_file = ".env"
        

# settings = Settings()
class Query(BaseModel):
    prompt: str


class ModelLoad(BaseModel):
    model_name: str


# Configuration du client vLLM
# Ajustez l'URL selon votre configuration
VLLM_SERVER_URL = "http://vllm_server:8000"


@app.get("/models")
def list_models():
    try:
        session = requests_retry_session()
        response = session.get(f"{VLLM_SERVER_URL}/v1/models", timeout=30)
        response.raise_for_status()
        models = response.json()
        return {"models": [model["id"] for model in models["data"]]}
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération des modèles : {str(e)}"
        )

        
        
@app.post("/load_model")
async def load_model(model: ModelLoad):
    try:
        # Vérifier si le modèle est disponible sur le serveur vLLM
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(f"{VLLM_SERVER_URL}/v1/models")
            if response.status_code == 200:
                available_models = response.json()
                if model.model_name not in [m["id"] for m in available_models["data"]]:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Le modèle {model.model_name} n'est pas disponible sur le serveur vLLM"
                    )
                return {"message": f"Modèle {model.model_name} est disponible sur le serveur vLLM"}
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail="Erreur lors de la vérification des modèles disponibles"
                )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur lors de la vérification du modèle : {str(e)}")


@app.post("/generate")
async def generate(query: Query):
    try:
        # Démarrer le chronomètre pour mesurer la durée totale
        start_time = time.time()

        # Effectuer une requête POST au serveur vLLM
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{VLLM_SERVER_URL}/v1/completions",
                json={
                    "model": "Qwen/Qwen2.5-Coder-1.5B",  # Modèle utilisé
                    "prompt": query.prompt,  # Prompt fourni par l'utilisateur
                    "max_tokens": 300,       # Nombre maximum de tokens générés
                    "temperature": 0.2,      # Température pour contrôler la créativité
                    "top_p": 0.9             # Top-p pour le filtrage des probabilités
                }
            )

        # Calculer la durée totale de génération
        end_time = time.time()

        # Vérifier si la requête a réussi
        if response.status_code == 200:
            result = response.json()
            return {
                # Texte généré par le modèle
                "response": result["choices"][0]["text"],
                "total_duration": end_time - start_time,   # Durée totale de la génération
                # Tokens utilisés pour le prompt
                "prompt_tokens": result["usage"]["prompt_tokens"],
                # Tokens générés en sortie
                "output_tokens": result["usage"]["completion_tokens"],
                "model": "Qwen/Qwen2.5-Coder-1.5B",  # Nom du modèle utilisé
            }
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Erreur du serveur vLLM : {response.text}"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur de génération : {str(e)}"
        )



@app.get("/health")
async def health_check():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{VLLM_SERVER_URL}/v1/models")
            if response.status_code == 200:
                return {"status": "OK", "vllm_server": "Connected"}
            else:
                return {"status": "OK", "vllm_server": "Not Connected"}
    except Exception as e:
        return {"status": "OK", "vllm_server": f"Error: {str(e)}"}
