from fastapi import FastAPI, Request
from typing import List, Optional
import uvicorn
from fastapi.responses import StreamingResponse, JSONResponse
import os
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
                                              EmbeddingResponseData,
                                              EmbeddingResponse,
                                              CompletionResponse,
                                              PoolingChatRequest,
                                              PoolingCompletionRequest,
                                              PoolingRequest, PoolingResponse,
                                              EmbeddingRequest,
                                              ErrorResponse)
from vllm.entrypoints.openai.serving_pooling import OpenAIServingPooling
from vllm.logger import init_logger
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from typing import AsyncIterator, Dict, Optional, Set, Tuple, Union

from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.serving_score import OpenAIServingScores
from typing_extensions import assert_never

# Définition du modèle
# "Qwen/Qwen2.5-1.5B-Instruct"
# "/model/Qwen2.5-Coder-7B-Instruct-IQ4_XS.gguf"
# "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF"
MODEL_NAME = os.getenv('MODEL_NAME')
print("MODEL_NAME 🚀", MODEL_NAME)
logger = init_logger('vllm.entrypoints.openai.api_server')



@asynccontextmanager
async def lifespan(app: FastAPI):
    engine, openai_serving_chat = await init_app()
    app.state.engine = engine
    app.state.openai_serving_chat = openai_serving_chat
    yield
    await app.state.engine.close()



app = FastAPI(lifespan=lifespan)


"""
    Init App in localhost
"""
async def init_app():

    # Initialisation du moteur de manière asynchrone
    engine_args = AsyncEngineArgs(model=MODEL_NAME,
                                  tensor_parallel_size=1,  # Single GPU
                                  gpu_memory_utilization=0.90,
                                  max_model_len=8192,
                                  quantization="fp8",  # Conversion en GPTQ : +40% tokens/s
                                  trust_remote_code=True,
                                  enforce_eager=False,
                                  )
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
    
    request_logger = RequestLogger(max_log_len=512)
    # Création de OpenAIServingChat et OpenAIServingTokenization
    app.state.openai_serving_chat = OpenAIServingChat(
        engine_client=engine,  # Le moteur d'inférence asynchrone pour  le modèle
        model_config=model_config, # La configuration du modèle
        models=openai_serving_models, # Instance of OpenAIServingModels 
        response_role="assistant", # Le rôle attribué aux réponses générées par le modèle
        # Logger pour les requêtes, désactivé ici
        request_logger= request_logger,
        chat_template=None,  # Template de chat personnalisé
        chat_template_content_format="auto" 
    )

    app.state.openai_serving_tokenization = OpenAIServingTokenization(
        engine_client = engine,
        model_config=model_config,
        models=openai_serving_models,
        request_logger=None,
        chat_template=None,  
        chat_template_content_format="auto" 
    )
    
    
        
        
        

    app.state.openai_serving_completion = OpenAIServingCompletion(
       engine_client = engine,
       model_config=model_config,
       models=openai_serving_models,
       request_logger=request_logger,
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

def score(request: Request) -> Optional[OpenAIServingScores]:
    return request.app.state.openai_serving_scores


def pooling(request: Request) -> Optional[OpenAIServingPooling]:
    return request.app.state.openai_serving_pooling

def embedding(request: Request) -> Optional[OpenAIServingEmbedding]:
    return request.app.state.openai_serving_embedding


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


@app.post("/v1/embeddings")
@with_cancellation
async def create_embedding(request: EmbeddingRequest, raw_request: Request):
    handler = embedding(raw_request)
    if handler is None:
        fallback_handler = pooling(raw_request)
        if fallback_handler is None:
            return base(raw_request).create_error_response(
                message="The model does not support Embeddings API")

        logger.warning(
            "Embeddings API will become exclusive to embedding models "
            "in a future release. To return the hidden states directly, "
            "use the Pooling API (`/pooling`) instead.")

        res = await fallback_handler.create_pooling(request, raw_request)

        generator: Union[ErrorResponse, EmbeddingResponse]
        if isinstance(res, PoolingResponse):
            generator = EmbeddingResponse(
                id=res.id,
                object=res.object,
                created=res.created,
                model=res.model,
                data=[
                    EmbeddingResponseData(
                        index=d.index,
                        embedding=d.data,  # type: ignore
                    ) for d in res.data
                ],
                usage=res.usage,
            )
        else:
            generator = res
    else:
        generator = await handler.create_embedding(request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    elif isinstance(generator, EmbeddingResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@app.post("/pooling")
@with_cancellation
async def create_pooling(request: PoolingRequest, raw_request: Request):
    handler = pooling(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Pooling API")

    generator = await handler.create_pooling(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    elif isinstance(generator, PoolingResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)
