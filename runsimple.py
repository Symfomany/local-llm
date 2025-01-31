# from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
# from vllm.engine.arg_utils import AsyncEngineArgs
# from vllm.engine.async_llm_engine import AsyncLLMEngine
# from vllm.transformers_utils.tokenizer import get_tokenizer



# from vllm import LLM, SamplingParams

# # Définition du modèle
# model_name = "Qwen/Qwen2.5-1.5B-Instruct"

# # Initialisation du modèle
# llm = LLM(model=model_name)

# # Définition du prompt
# prompt = "Écris une fonction Python qui additionne deux nombres."

# # Paramètres d'échantillonnage (ajustables)
# sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=200)

# # Génération du texte
# outputs = llm.generate([prompt], sampling_params)

# # Affichage du résultat
# print(outputs[0].outputs[0].text)


from fastapi import FastAPI, Request
from typing import List, Optional
import uvicorn
from fastapi.responses import StreamingResponse, JSONResponse

from contextlib import asynccontextmanager
from vllm.entrypoints.openai.serving_tokenization import (
    OpenAIServingTokenization)
from vllm.entrypoints.openai.serving_models import OpenAIServingModels, BaseModelPath
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.engine.async_llm_engine import AsyncLLMEngine, AsyncEngineArgs
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.utils import with_cancellation
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              ChatCompletionResponse,
                                              CompletionRequest,
                                              CompletionResponse,
                                              ErrorResponse)


# Définition du modèle
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


"""
    App from FastAPI
"""
@asynccontextmanager
async def lifespan(app: FastAPI):
    engine, openai_serving_chat = await init_app()
    app.state.engine = engine
    app.state.openai_serving_chat = openai_serving_chat
    yield

    await app.state.engine.close()



app = FastAPI(lifespan=lifespan)


"""
    Init App
"""
async def init_app():

    # Initialisation du moteur de manière asynchrone
    engine_args = AsyncEngineArgs(model=MODEL_NAME)
    engine =  AsyncLLMEngine.from_engine_args(engine_args)

    # Obtention de la configuration du modèle
    model_config = await engine.get_model_config()
    

    """
        engine_client : Une instance de EngineClient, qui est le moteur d'inférence asynchrone pour le modèle de langage.
        model_config : Un objet ModelConfig contenant la configuration du modèle.
        base_model_paths : Une liste de BaseModelPath, où chaque élément contient le nom et le chemin du modèle de base. Dans ce cas, il n'y a qu'un seul modèle spécifié avec le nom et le chemin définis par MODEL_NAME1.
        lora_modules : Défini à None, ce paramètre permet d'ajouter des modules LoRA (Low-Rank Adaptation) au modèle. LoRA est une technique d'adaptation efficace pour les grands modèles de langage2.
        prompt_adapters : Également défini à None, ce paramètre permet d'ajouter des adaptateurs de prompt au modèle. Les adaptateurs de prompt sont des méthodes d'apprentissage efficaces pour adapter les modèles de vision et de langage à de nouvelles tâches68.
    """

     # Création de OpenAIServingModels
    openai_serving_models = OpenAIServingModels(
        engine_client=engine,
        model_config=model_config,
        base_model_paths=[BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME)],
        lora_modules=None,  # Ajustez selon vos besoins
        prompt_adapters=None,  # Ajustez selon vos besoins
    )
    await openai_serving_models.init_static_loras()
    
    # Création de OpenAIServingChat
    app.state.openai_serving_chat = OpenAIServingChat(
        engine_client=engine,  # Le moteur d'inférence asynchrone pour  le modèle
        model_config=model_config, # La configuration du modèle
        models=openai_serving_models, # Instance de OpenAIServingModels contenant les informations sur les modèles disponibles
        response_role="assistant", # Le rôle attribué aux réponses générées par le modèle
        request_logger=None,   # Logger pour les requêtes, désactivé ici
        chat_template=None,  # Template de chat personnalisé, non utilisé ici
        chat_template_content_format="auto" 
    )

    app.state.openai_serving_tokenization = OpenAIServingTokenization(
        engine_client = engine,
        model_config=model_config,
        models=openai_serving_models,
        request_logger=None,
        chat_template=None,  # Template de chat personnalisé, non utilisé ici
        chat_template_content_format="auto" 
    )

    app.state.openai_serving_completion = OpenAIServingCompletion(
       engine_client = engine,
       model_config=model_config,
       models=openai_serving_models,
        request_logger=None,
        return_tokens_as_token_ids=False,
    )

    return engine, app.state.openai_serving_chat 



"""
    Handlers with state of app (Management of State)
"""
def chat(request: Request) -> Optional[OpenAIServingChat]:
    return request.app.state.openai_serving_chat

def tokenization(request: Request) -> OpenAIServingTokenization:
    return request.app.state.openai_serving_tokenization

def base(request: Request) -> OpenAIServing:
    return tokenization(request)

def completion(request: Request) -> Optional[OpenAIServingCompletion]:
    return request.app.state.openai_serving_completion


"""
   Routing for the API
"""

# Endpoint pour /v1/chat/completions (compatible OpenAI Chat API)
@app.post("/v1/chat/completions")
@with_cancellation
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    handler = chat(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Chat Completions API")

    generator = await handler.create_chat_completion(request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)

    elif isinstance(generator, ChatCompletionResponse):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")


@app.post("/v1/completions")
@with_cancellation
async def create_completion(request: CompletionRequest, raw_request: Request):
    handler = completion(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Completions API")

    generator = await handler.create_completion(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    elif isinstance(generator, CompletionResponse):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")




# Lancement du serveur avec Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
