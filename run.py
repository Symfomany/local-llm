from fastapi.responses import JSONResponse, StreamingResponse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os
import time
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import asyncio

import uvicorn


from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.transformers_utils.tokenizer import get_tokenizer
import json
import logging

logging.basicConfig(level=logging.DEBUG)
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


    


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await init()
        yield
    except Exception as e:
        print("Error:" +  str(e))
        raise

app = FastAPI(lifespan=lifespan)


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
MODEL = "/model/qwen2.5-coder-7b-instruct-q3_k_m.gguf"


async def init():
    print("Init...")
    engine_args = AsyncEngineArgs(
        model=MODEL,  device="cuda")
    app.state.engine = await AsyncLLMEngine.from_engine_args(engine_args)
    app.state.tokenizer = get_tokenizer(engine_args.model)
    model_config = await app.state.engine.get_model_config()
    app.state.openai_serving_chat = OpenAIServingChat(
        app.state.engine,
        model_config,
        [MODEL],
        "assistant"
    )

####
# Real
####

def chat(request: Request) -> Optional[OpenAIServingChat]:
    return request.app.state.openai_serving_chat


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    try:
        generator = await asyncio.wait_for(
            app.state.openai_serving_chat.create_chat_completion(
                request.dict(), raw_request),
            timeout=60  # Augmentez cette valeur si nécessaire
        )

        if request.stream:
            return StreamingResponse(generator, media_type="text/event-stream")
        else:
            response = await anext(generator)
            return JSONResponse(content=response)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


####################################################################################################################################
# Chat 
####################################################################################################################################



# async def stream_response(request: ChatCompletionRequest, prompt: str):
#     async with httpx.AsyncClient() as client:
#         async with client.stream(
#             "POST",
#             f"{VLLM_SERVER_URL}/v1/completions",
#             json={
#                 "model": request.model,
#                 "prompt": prompt,
#                 "max_tokens": request.max_tokens,
#                 "temperature": request.temperature,
#                 "stream": True,
#             },
#             timeout=60.0
#         ) as response:
#             if response.status_code != 200:
#                 yield f"data: {json.dumps({'error': f'Erreur du serveur vLLM : {response.text}'})}\n\n"
#                 return

#             start_time = time.time()
#             async for line in response.aiter_lines():
#                 if line:
#                     try:
#                         data = json.loads(line.split('data: ')[1])
#                         yield format_sse_event(data, start_time)
#                     except json.JSONDecodeError:
#                         continue
#                     except IndexError:
#                         continue

#             yield "data: [DONE]\n\n"


# async def generate_complete_response(request: ChatCompletionRequest, prompt: str):
#     async with httpx.AsyncClient() as client:
#         response = await client.post(
#             f"{VLLM_SERVER_URL}/v1/completions",
#             json={
#                 "model": request.model,
#                 "prompt": prompt,
#                 "max_tokens": request.max_tokens,
#                 "temperature": request.temperature,
#             }
#         )

#     if response.status_code != 200:
#         raise HTTPException(status_code=response.status_code,
#                             detail=f"Erreur du serveur vLLM : {response.text}")

#     result = response.json()
#     generated_text = result["choices"][0]["text"]

#     return {
#         "id": str(uuid.uuid4()),
#         "object": "chat.completion",
#         "created": int(time.time()),
#         "model": request.model,
#         "choices": [{
#             "index": 0,
#             "message": {
#                 "role": "assistant",
#                 "content": generated_text,
#             },
#             "finish_reason": result.get("finish_reason", "stop"),
#         }],
#         "usage": {
#             "prompt_tokens": result["usage"]["prompt_tokens"],
#             "completion_tokens": result["usage"]["completion_tokens"],
#             "total_tokens": result["usage"]["total_tokens"],
#         },
#     }



# @app.post("/v1/chat/completions")
# async def chat_completions(request: ChatCompletionRequest):
#     try:
#         prompt = "\n".join(
#             [f"{msg.role}: {msg.content}" for msg in request.messages])

#         if request.stream:
#             return StreamingResponse(stream_response(request, prompt), media_type="text/event-stream")
#         else:
#             return await generate_complete_response(request, prompt)

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


##############################################################################################################
# Autocomplete
##########################################################################################################################



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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
