from vllm import SamplingParams  # Ajouter cet import
from fastapi import FastAPI
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from typing import Dict
from contextlib import asynccontextmanager
from pydantic import BaseModel
from uuid import uuid4  # Ajouter en haut du fichier
import logging

# Activer le logging détaillé
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)

app = FastAPI()

class GenerateTextRequest(BaseModel):
    model_type: str
    prompt: str
    max_tokens: int = 200
    temperature: float = 0.7
    
MODELS = {
    "base": {
        "model_id": "Qwen/Qwen2.5-Coder-1.5B",  # "Qwen/Qwen2.5-Coder-1.5B",
        "max_model_len": 2048
    },
    # "instruct": {
    #     "model_id": "Qwen/Qwen2.5-Coder-1.5B",
    #     "quantization": "awq",
    #     "max_model_len": 2048
    # }
}

engines: Dict[str, AsyncLLMEngine] = {}

import torch
@asynccontextmanager
async def lifespan(app: FastAPI):
    for model_name, config in MODELS.items():
        print("Run model ✨", model_name)
        engine_args = AsyncEngineArgs(
            model=config["model_id"],
            quantization="fp8",
            enforce_eager=False,
            disable_log_stats=False,  # Activer les statistiques détaillées
            disable_log_requests=False,  # Logger chaque requête
            tensor_parallel_size=1,
            gpu_memory_utilization=0.40,
            max_model_len=config["max_model_len"],
            dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        )
        engines[model_name] = AsyncLLMEngine.from_engine_args(engine_args)
        
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/generate")
async def generate_text(request: GenerateTextRequest):
    if request.model_type not in engines:
        return {"error": "Modèle non trouvé"}

    engine = engines[request.model_type]

    # Modifier la section generate :
    sampling_params = SamplingParams(
        max_tokens=request.max_tokens,
        temperature=request.temperature
    )
    # Génération d'un ID unique pour la requête
    request_id = str(uuid4())

    # Appel correct avec tous les paramètres requis
    results_generator = engine.generate(
        request.prompt,
        sampling_params,
        request_id
    )

    # Récupération du premier résultat (adapté au mode non-streaming)
    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    return {
        "model": request.model_type,
        "response": final_output.outputs[0].text
    }


@app.get("/health")
def health_check():
    vram_usage = sum(engine.get_memory_stats()["allocated"] for engine in engines.values())
    return {
        "status": "OK",
        "vram_used": f"{vram_usage / 1024**3:.2f} Go",
        "models_loaded": list(engines.keys())
    }
