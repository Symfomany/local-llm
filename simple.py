from ollama import Client
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import ollama
import os
import subprocess
import requests
import time
app = FastAPI()

class Query(BaseModel):
    prompt: str

class ModelLoad(BaseModel):
    model_name: str

@app.post("/load_model")
async def load_model(model: ModelLoad):
    async with httpx.AsyncClient() as client:
        try:
            # Chemin du Modelfile pour Qwen
            # Lire le contenu du fichier Modelfile
            modelfile_path = "Modelfile"
            if os.path.exists(modelfile_path):
                with open(modelfile_path, 'r') as file:
                    modelfile_content = file.read()
            else:
                raise FileNotFoundError(f"Le fichier {modelfile_path} n'existe pas.")
        #     modelfile_content = '''
        #     FROM ./model/qwen2.5-coder.gguf
        #     PARAMETER temperature 0.7
        #     SYSTEM "Tu es un développeur Python expérimenté avec Flask, Django et Gitlab."
        #     '''
            print(modelfile_content, "modelfile_content")
        #    Nom du modèle
            model_name = "qwen2_5_coder"
        #    Envoyer une requête POST à l'API /create
            # response = requests.post('http://localhost:11434/api/create', json={
            #     'name': model_name,
            #     'modelfile': modelfile_content
            # })
            

        #     # Afficher les détails de la réponse
            print("response", response)
            print(f"Modèle '{model_name}' créé avec succès.")
            print("Status code:", response.status_code)
            print("Response content:", response.text)
        except requests.exceptions.RequestException as e:
            print("Erreur de connexion:", str(e))
            
        #     # Vérification de la réponse
            if response.status_code == 200:
                
        #         # Attente courte pour s'assurer que le modèle est prêt
                time.sleep(1)
                
        #         # Génération d'un exemple de texte
                try:
                    response = requests.post('http://localhost:11434/api/generate', json={
                        'model': model_name,
                        'prompt': "Votre prompt ici"
                    })

                    print("Réponse:", response.json()['response'])

                    return {
                        "message": f"Model {model_name} loaded successfully",
                    }
                except Exception as e:
                    return {"error": f"Erreur lors de la génération de texte: {str(e)}"}      
                
                
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to load model: {str(e)}")
        except httpx.RequestError:
            raise HTTPException(status_code=500, detail="Failed to connect to Ollama server")


@app.post("/generate")
async def generate(query: Query):
    try:
        response = ollama.generate(model="qwen2.5-coder", prompt=query.prompt)
        return {
            "response": response['response'],
            "total_duration": response['total_duration'],
            "load_duration": response['load_duration'],
            "prompt_eval_count": response['prompt_eval_count'],
            "eval_count": response['eval_count'],
            "eval_duration": response['eval_duration']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "OK"}
