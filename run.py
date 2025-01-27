from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams, EngineArgs
import os

import asyncio

from huggingface_hub import login

# Méthode 1 : Utiliser un token

app = FastAPI()

class Query(BaseModel):
    prompt: str

class ModelLoad(BaseModel):
    model_name: str

# Dictionnaire global pour stocker les instances de modèle
models = {}

@app.post("/load_model")
async def load_model(model: ModelLoad):
    try:
        model_path = "/model/gemma-2b.gguf"
        
        # Vérifier l'existence du fichier
        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=404,
                detail=f"Le fichier modèle {model_path} n'existe pas"
            )
        print("File existed...", model_path)
        
        # return {"response": True}
            
        tokenizer = "google/gemma-2b"
        llm = LLM(
            model=model_path,
            tokenizer=tokenizer,
            trust_remote_code=True,
            device='cuda',
            # max_model_len=4096,
        )
        models[model.model_name] = llm

        print('Going generate...')
        # Générer un texte simple pour vérifier le chargement
        sampling_params = SamplingParams(temperature=0.1, max_tokens=100)
        outputs = llm.generate("Generate a simple hello world program in Python", sampling_params)

        return {
            "message": f"Model {model.model_name} loaded successfully",
            "sample_output": outputs[0].outputs[0].text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.post("/generate")
async def generate(query: Query):
    try:
        # Récupérer le modèle chargé
        llm = models.get("google/gemma-2b")

        # Vérifier si le modèle est chargé
        if not llm:
            raise HTTPException(
                status_code=400,
                detail="Modèle non chargé. Utilisez /load_model d'abord."
            )

        # Paramètres de génération flexibles
        sampling_params = SamplingParams(
            temperature=0.7,     # Créativité
            max_tokens=100,      # Longueur maximale
            top_p=0.9,           # Diversité des tokens
            presence_penalty=0.1  # Réduire les répétitions
        )

        # Mesurer le temps de génération
        start_time = time.time()
        outputs = llm.generate(query.prompt, sampling_params)
        end_time = time.time()

        # Extraire les informations de génération
        generated_text = outputs[0].outputs[0].text

        return {
            "response": generated_text,
            "total_duration": end_time - start_time,
            "prompt_tokens": len(outputs[0].prompt_token_ids),
            "output_tokens": len(outputs[0].outputs[0].token_ids),
            "model": "google/gemma-2b"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur de génération : {str(e)}"
        )

@app.get("/health")
async def health_check():
    return {"status": "OK"}
