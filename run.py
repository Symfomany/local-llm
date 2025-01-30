from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os
import time
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from typing import List, Optional
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse


import json

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
    temperature: float = 0.2
    top_p: float = 0.9
    model: str = "Qwen/Qwen2.5-Coder-1.5B"

class ModelLoad(BaseModel):
    model_name: str


class ChatMessage(BaseModel):
    role: str  # Peut être "system", "user" ou "assistant"
    content: str  # Contenu du message

class ChatCompletionRequest(BaseModel):
    model: str = "gpt-3.5-turbo"  # Modèle utilisé
    messages: List[ChatMessage]  # Liste des messages dans le contexte
    max_tokens: Optional[int] = 512  # Nombre maximum de tokens générés
    temperature: Optional[float] = 0.7  # Contrôle de la créativité
    stream: Optional[bool] = False  # Si True, active le streaming des réponses



# Configuration du client vLLM
# Ajustez l'URL selon votre configuration
VLLM_SERVER_URL = "http://vllm_server:8000"



@app.post("/load_model")
async def load_model(model: ModelLoad):
    try:
        model_path = "/model/gemma-2b.gguf"  # Chemin vers votre modèle

        # Vérifier si le fichier existe
        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=404,
                detail=f"Le fichier modèle {model_path} n'existe pas"
            )
        
        print("Model existing ...⚡")

        # Charger le modèle via l'API vLLM
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                # Endpoint pour charger un modèle
                f"{VLLM_SERVER_URL}/v1/load_model",
                json={
                    "model_name": "gemma-2b",  # Nom du modèle que vous souhaitez charger
                    "model_path": model_path  # Chemin vers le fichier du modèle
                }
            )

        if response.status_code == 200:
            return {"message": f"Modèle {model.model_name} chargé avec succès"}
        else:
            print("No model ...😊")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Erreur lors du chargement du modèle : {response.text}"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur lors du chargement du modèle : {str(e)}")


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
                    "model": query.model,
                    "prompt": query.prompt,
                    "max_tokens": 100,
                    "temperature": query.temperature,
                    "top_p": query.top_p          # Top-p pour le filtrage des probabilités
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


async def generate_response(prompt: str, model: str, temperature: float, max_tokens: int) -> str:
    """
    Fonction pour appeler le serveur vLLM et obtenir une réponse.
    """
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        response = await client.post(
            f"{VLLM_SERVER_URL}/v1/completions",
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Erreur du serveur vLLM : {response.text}"
        )

    result = response.json()
    return result["choices"][0]["text"]


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        prompt = "\n".join(
            [f"{msg.role}: {msg.content}" for msg in request.messages])

        if request.stream:
            return StreamingResponse(stream_response(request, prompt), media_type="text/event-stream")
        else:
            return await generate_complete_response(request, prompt)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def stream_response(request: ChatCompletionRequest, prompt: str):
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            f"{VLLM_SERVER_URL}/v1/completions",
            json={
                "model": request.model,
                "prompt": prompt,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "stream": True,
            },
            timeout=60.0
        ) as response:
            if response.status_code != 200:
                yield f"data: {json.dumps({'error': f'Erreur du serveur vLLM : {response.text}'})}\n\n"
                return

            start_time = time.time()
            async for line in response.aiter_lines():
                if line:
                    try:
                        data = json.loads(line.split('data: ')[1])
                        yield format_sse_event(data, start_time)
                    except json.JSONDecodeError:
                        continue
                    except IndexError:
                        continue

            yield "data: [DONE]\n\n"


def format_sse_event(data, start_time):
    event = {
        "id": str(uuid.uuid4()),
        "object": "chat.completion.chunk",
        "created": int(start_time),
        "model": data.get("model", ""),
        "choices": [{
            "index": 0,
            "delta": {
                "content": data["choices"][0]["text"],
            },
            "finish_reason": data["choices"][0].get("finish_reason"),
        }],
    }
    return f"data: {json.dumps(event)}\n\n"


async def generate_complete_response(request: ChatCompletionRequest, prompt: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{VLLM_SERVER_URL}/v1/completions",
            json={
                "model": request.model,
                "prompt": prompt,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
            }
        )

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code,
                            detail=f"Erreur du serveur vLLM : {response.text}")

    result = response.json()
    generated_text = result["choices"][0]["text"]

    return {
        "id": str(uuid.uuid4()),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": generated_text,
            },
            "finish_reason": result.get("finish_reason", "stop"),
        }],
        "usage": {
            "prompt_tokens": result["usage"]["prompt_tokens"],
            "completion_tokens": result["usage"]["completion_tokens"],
            "total_tokens": result["usage"]["total_tokens"],
        },
    }


async def stream_generate_response(prompt: str, model: str, temperature: float, max_tokens: int):
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        async with client.stream(
            "POST",
            f"{VLLM_SERVER_URL}/v1/completions",
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
            },
        ) as response:
            if response.status_code != 200:
                yield f"data: {json.dumps({'error': f'Erreur du serveur vLLM : {response.text}'})}\n\n"
                return

            start_time = time.time()  
            s
            async for line in response.aiter_lines():
                if line:
                    try:
                        data = json.loads(line.split('data: ')[1])
                        yield format_sse_event(data, start_time)
                    except json.JSONDecodeError:
                        continue
                    except IndexError:
                        continue

            yield "data: [DONE]\n\n"


def format_sse_event(data, start_time):
    event = {
        "id": str(uuid.uuid4()),
        "object": "text_completion",
        "created": int(start_time),
        "model": data.get("model", ""),
        "choices": [{
            "text": data["choices"][0]["text"],
            "index": 0,
            "logprobs": None,
            "finish_reason": data["choices"][0].get("finish_reason"),
        }],
    }
    return f"data: {json.dumps(event)}\n\n"


def format_complete_response(response, model, prompt):
    return {
        "id": str(uuid.uuid4()),
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "text": response,
            "index": 0,
            "logprobs": None,
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(response.split()),
            "total_tokens": len(prompt.split()) + len(response.split())
        }
    }

@app.post("/v1/completions")
async def completions(request: Request):
    """
    Route pour générer des complétions de texte à partir d'un modèle, avec support du streaming.
    """
    try:
        data = await request.json()
        # Extraire les paramètres de la requête
        prompt = data.get("prompt", "")
        model = data.get("model", "default_model")
        temperature = data.get("temperature", 1.0)
        max_tokens = data.get("max_tokens", 100)
        stream = data.get("stream", False)

        if stream:
            return StreamingResponse(stream_generate_response(prompt, model, temperature, max_tokens), media_type="text/event-stream")
        else:
            response = await generate_response(prompt, model, temperature, max_tokens)
            return format_complete_response(response, model, prompt)

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503, detail="Erreur de connexion au serveur vLLM.")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur interne : {str(e)}")


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
