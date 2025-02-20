from __future__ import annotations
import asyncio
import aiohttp
import psycopg2
import pandas as pd
import logging
import argparse
from typing import AsyncIterator, Tuple, List, Optional
from contextlib import asynccontextmanager, contextmanager
from functools import partial
from dataclasses import dataclass, field
from typing import Dict
import time
import os
from dotenv import load_dotenv
import random

# Configuration ---------------------------------------------------------------
load_dotenv()


@dataclass(frozen=True)
class Config:
    TEMPERATURE: str = os.getenv('TEMPERATURE', 0.7)
    TOP_P: float = os.getenv('TOP_P', 0.95)
    MAX_TOKENS: int = os.getenv('MAX_TOKENS', 400)
    GPU_MEMORY_UTILIZATION: str = os.getenv('GPU_MEMORY_UTILIZATION', 0.95)
    MAX_MODEL_LEN: str = os.getenv('MAX_MODEL_LEN', 8192)
    QUANTIZATION: str = os.getenv('QUANTIZATION', "fp8")
    MAX_NUM_SEQ: str = os.getenv('MAX_NUM_SEQ', 256)
    MAX_NUM_BATCHED_TOKENS: str = os.getenv('MAX_NUM_BATCHED_TOKENS', 4096)
    BASE_URL: str = "http://localhost:8080/v1"
    API_URL: str = "http://localhost:8080/v1/chat/completions"
    MODEL_NAME: str = os.getenv('MODEL_NAME', "Qwen/Qwen2.5-Coder-3B-Instruct")
    POSTGRES_CONFIG: Dict[str, str] = field(
        default_factory=lambda: {
            "host": "localhost",
            "database": "llm",
            "user": "postgres",
            "password": "admin"
        }
    )
    MAX_TESTS: int = int(os.getenv('MAX_TESTS', 3))


config = Config()

# Logging ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('benchmark.log')]
)

# Domain Models ----------------------------------------------------------------


@dataclass
class BenchmarkResult:
    prompt: str
    code: Optional[str]
    latency: float
    time_to_first_token: float
    prompt_tokens: int
    completion_tokens: int
    tokens_per_second: float

# Database Layer ---------------------------------------------------------------


@contextmanager
def db_connection() -> psycopg2.extensions.connection:
    """Gestionnaire de contexte pour les connexions PostgreSQL"""
    conn = psycopg2.connect(**config.POSTGRES_CONFIG)
    try:
        yield conn
    finally:
        conn.close()


async def fetch_metrics(session: aiohttp.ClientSession) -> dict:
    """Récupère les métriques depuis l'API vLLM"""
    try:
        async with session.get("http://localhost:8080/metrics") as response:
            if response.status == 200:
                metrics_text = await response.text()
                logging.warning(
                    f"Metrics endpoint returned text/plain: {metrics_text}")
                
                return metrics_text
            else:
                logging.error(f"Erreur récupération métriques : {response.status}")
                return {}
    except Exception as e:
        logging.error(f"Erreur appel métriques : {str(e)}")
        return {}
    
def init_database(drop_tables: bool = False) -> None:
    """Initialise la structure de la base de données"""
    with db_connection() as conn, conn.cursor() as cur:
        if drop_tables:
            cur.execute("DROP TABLE IF EXISTS prompts, api_parameters CASCADE")
            logging.info("Tables existantes supprimées")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS api_parameters (
                id SERIAL PRIMARY KEY,
                model_name TEXT NOT NULL,
                max_tokens INT NOT NULL,
                temperature FLOAT NOT NULL,
                top_p FLOAT NOT NULL,
                frequency_penalty FLOAT NOT NULL,
                presence_penalty FLOAT NOT NULL,
                gpu_memory_utilization FLOAT NOT NULL, 
                max_model_len INT NOT NULL,
                quantization TEXT NOT NULL,
                max_num_seq INT NOT NULL, 
                max_num_batched_tokens  INT NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS prompts (
                id SERIAL PRIMARY KEY,
                prompt TEXT NOT NULL,
                code TEXT,
                parameter_id INT REFERENCES api_parameters(id),
                latency FLOAT,
                time_to_first_token FLOAT,
                prompt_tokens INT,
                completion_tokens INT,
                tokens_per_second FLOAT,
                processing_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                len_concurency INT
            )
        """)
        conn.commit()


def save_result(param_id: int, result: BenchmarkResult) -> None:
    """Persiste les résultats dans PostgreSQL"""
    with db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO prompts (
                prompt, code, parameter_id, latency, 
                time_to_first_token, prompt_tokens, 
                completion_tokens, tokens_per_second, len_concurency
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                result.prompt, result.code, param_id, result.latency,
                result.time_to_first_token, result.prompt_tokens,
                result.completion_tokens, result.tokens_per_second, config.MAX_TESTS
            )
        )
        conn.commit()

# API Layer --------------------------------------------------------------------


@asynccontextmanager
async def api_session() -> AsyncIterator[aiohttp.ClientSession]:
    """Gestionnaire de contexte pour les sessions API"""
    """
    Creates an asynchronous context manager for aiohttp ClientSession.

    This ensures that the session is properly closed after use, even if exceptions occur.
    It yields an aiohttp.ClientSession object for making API requests within the `async with` block.
    """
    async with aiohttp.ClientSession() as session:
        yield session

async def call_api(session: aiohttp.ClientSession, prompt: str) -> BenchmarkResult:
    """Effectue un appel à l'API LLM et retourne les métriques"""
    payload = {
        "model": config.MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": config.MAX_TOKENS,
        "temperature": config.TEMPERATURE,
        "top_p": config.TOP_P,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    }

    start_time = time.monotonic()

    try:
        async with session.post(config.API_URL, json=payload) as response:
            data = await response.json()
            latency = time.monotonic() - start_time

            usage = data.get('usage', {})
            return BenchmarkResult(
                prompt=prompt,
                code=data["choices"][0]["message"]["content"],
                latency=latency,
                time_to_first_token=latency / 2,
                prompt_tokens=usage.get('prompt_tokens', 0),
                completion_tokens=usage.get('completion_tokens', 0),
                tokens_per_second=usage.get(
                    'completion_tokens', 0) / latency if latency > 0 else 0.0
            )
    except Exception as e:
        logging.error(f"API call failed: {str(e)}")
        return BenchmarkResult(prompt, None, 0.0, 0.0, 0, 0, 0.0)

# Workflow ---------------------------------------------------------------------


async def process_prompt(session: aiohttp.ClientSession, param_id: int, prompt: str) -> None:
    """Pipeline complet de traitement d'un prompt"""
    loop = asyncio.get_running_loop()
    result = await call_api(session, prompt)
    # metrics = await fetch_metrics(session)  # Récupération des métriques
    # if metrics:
    #     logging.info(f"Métriques supplémentaires : {metrics}")
                    
    await loop.run_in_executor(None, partial(save_result, param_id, result))
    logging.info(f"Traité: {prompt[:50]}...")


async def main(args: argparse.Namespace) -> None:
    """Workflow principal"""
    # Initialisation
    init_database(args.drop_tables)

    # Lecture des prompts
    df = pd.read_csv('prompts.csv')
    #prompts = df.head(config.MAX_TESTS)['prompt'].tolist()
    prompts = random.sample(df['prompt'].tolist(), config.MAX_TESTS)
    
    # Enregistrement des paramètres
    with db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO api_parameters 
            (model_name, max_tokens, temperature, top_p, frequency_penalty, presence_penalty, gpu_memory_utilization, max_model_len, quantization, max_num_seq, max_num_batched_tokens)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (config.MODEL_NAME, config.MAX_TOKENS,
             config.TEMPERATURE, config.TOP_P, 0.0, 0.0, config.GPU_MEMORY_UTILIZATION,config.MAX_MODEL_LEN, config.QUANTIZATION, config.MAX_NUM_SEQ, config.MAX_NUM_BATCHED_TOKENS )
        )
        param_id = cur.fetchone()[0]
        conn.commit()

    # Traitement parallèle
    async with api_session() as session:
        tasks = [process_prompt(session, param_id, p) for p in prompts]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--drop-tables', action='store_true')
    args = parser.parse_args()

    try:
        asyncio.run(main(args))
    except Exception as e:
        logging.critical(f"Erreur critique: {str(e)}", exc_info=True)
        raise
