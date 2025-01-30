from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
import os
import time
import requests
from requests.adapters import HTTPAdapter
from typing import List, Optional
import uuid
import json
import asyncio

from requests.packages.urllib3.util.retry import Retry

app = FastAPI()

# ... (Le reste du code reste inchangé jusqu'à la fonction chat_completions)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    async def generate():
        try:
            prompt = "\n".join(
                [f"{msg.role}: {msg.content}" for msg in request.messages])

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
                    timeout=None
                ) as response:
                    if response.status_code != 200:
                        raise HTTPException(
                            status_code=response.status_code,
                            detail=f"Erreur du serveur vLLM : {response.text}",
                        )

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = json.loads(line[6:])
                            if data["choices"][0]["finish_reason"] is not None:
                                break

                            chunk = {
                                "id": str(uuid.uuid4()),
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": request.model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {
                                        "content": data["choices"][0]["text"],
                                    },
                                    "finish_reason": None,
                                }],
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"

                    # Send the final chunk
                    final_chunk = {
                        "id": str(uuid.uuid4()),
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }],
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

        except Exception as e:
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "internal_error",
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/v1/completions")
async def completions(request: Request):
    async def generate():
        try:
            data = await request.json()
            prompt = data.get("prompt", "")
            model = data.get("model", "default_model")
            temperature = data.get("temperature", 1.0)
            max_tokens = data.get("max_tokens", 100)

            async with httpx.AsyncClient() as client:
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
                    timeout=None
                ) as response:
                    if response.status_code != 200:
                        raise HTTPException(
                            status_code=response.status_code,
                            detail=f"Erreur du serveur vLLM : {response.text}",
                        )

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = json.loads(line[6:])
                            if data["choices"][0]["finish_reason"] is not None:
                                break

                            chunk = {
                                "id": str(uuid.uuid4()),
                                "object": "text_completion",
                                "created": int(time.time()),
                                "model": model,
                                "choices": [{
                                    "text": data["choices"][0]["text"],
                                    "index": 0,
                                    "logprobs": None,
                                    "finish_reason": None,
                                }],
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"

                    # Send the final chunk
                    final_chunk = {
                        "id": str(uuid.uuid4()),
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{
                            "text": "",
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": "stop",
                        }],
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

        except Exception as e:
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "internal_error",
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

# ... (Le reste du code reste inchangé)
