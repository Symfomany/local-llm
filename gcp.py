from fastapi import FastAPI, HTTPException
from google.cloud import storage
import os
from ollama import Client
from pydantic import BaseModel
from contextlib import asynccontextmanager
import asyncio
import sys
import uvicorn

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

@asynccontextmanager
async def lifespan(app: FastAPI):
    await download_gguf()
    await create_ollama_model()
    yield



app = FastAPI(lifespan=lifespan)
client = Client()
model_loaded = False


async def download_gguf():
    print("Begin loading...")
    bucket_name = "llms-devaug"
    file_name = "phi-3-mini-4k-instruct-q4.gguf"
    destination_folder = "./model"
    destination_file_name = os.path.join(destination_folder, file_name)

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    if not os.path.exists(destination_file_name):
        print("File not existing, start downloading")
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        blob.download_to_filename(destination_file_name)
        print(f"Fichier téléchargé : {destination_file_name}")
    else:
        print(f"Fichier déjà présent : {destination_file_name}")



async def create_ollama_model():
    modelfile_path = "Modelfile"
    if not os.path.exists(modelfile_path):
        raise FileNotFoundError(f"Modelfile not found at {modelfile_path}")
    
        # Lire et afficher le contenu du fichier
    with open(modelfile_path, "r") as file:
        content = file.read()

    print(content)  # Affiche le contenu du fichier
    
    try:
        process = await asyncio.create_subprocess_exec(
            "ollama", "create", "alpha", "-f", modelfile_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            print(f"Error creating model: {stderr.decode()}")
            raise Exception(f"Failed to create model: {stderr.decode()}")
        else:
            print(f"Model created successfully: {stdout.decode()}")
    except Exception as e:
        print(f"An error occurred while creating the model: {str(e)}")
        raise
    

@app.get("/health")
async def health():
    return {"message": "OK"}


@app.get("/load-model")
async def load_model():
    global model_loaded
    if not model_loaded:
        try:
            client.create(model="phi-3-mini-4k-instruct",
                          path="./model/phi-3-mini-4k-instruct-q4.gguf")
            model_loaded = True
            return {"message": "Modèle chargé avec succès"}
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Erreur lors du chargement du modèle : {str(e)}")
    return {"message": "Le modèle est déjà chargé"}


class PromptRequest(BaseModel):
    prompt: str


@app.post("/generate")
async def generate(request: PromptRequest):
    try:
        response = client.generate(
            model="alpha", prompt=request.prompt)
        return {"generated_text": response['response']}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur lors de la génération : {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
