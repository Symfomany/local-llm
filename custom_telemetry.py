from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager

import psycopg2
import uvicorn
from dataclasses import dataclass, field
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from typing import Dict 


# Configuration ---------------------------------------------------------------
load_dotenv()


@dataclass(frozen=True)
class Config:
    POSTGRES_CONFIG: Dict[str, str] = field(
        default_factory=lambda: {
            "host": "localhost",
            "database": "llm",
            "user": "postgres",
            "password": "admin",
        }
    )


config = Config()


@asynccontextmanager
async def db_connection():
    """Asynchronous context manager for PostgreSQL connections."""
    conn = None
    try:
        conn = psycopg2.connect(**config.POSTGRES_CONFIG)
        print("Database connection established")
        yield conn
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)  # Exit if database connection fails
    finally:
        if conn:
            conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler to manage application startup and shutdown."""
    print("Starting up...")
    # try:
    #     async with db_connection():
    #         print("Database connection tested successfully during startup.")
    # except Exception as e:
    #     print(f"Error during database connection test: {e}")
    #     sys.exit(1)  # Exit if database connection test fails
    yield


app = FastAPI(lifespan=lifespan)


# Configuration des CORS
origins = [
    "*",  # À éviter en production, mais utile pour le développement local
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    # Autorise toutes les méthodes (GET, POST, PUT, DELETE, etc.)
    allow_methods=["*"],
    allow_headers=["*"],  # Autorise tous les headers
)


# Définition du modèle de données pour la session
class Session(BaseModel):
    sessionId: str
    ide: str
    history: List[dict]


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@app.post("/sessions")
async def create_session(session: Session):
    """Enregistre une session dans la base de données."""
    try:
        async with db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO history (history_json, ide, session_id)
                    VALUES (%s, %s, %s)
                    """,
                    (json.dumps(session.history), session.ide, session.sessionId),
                )
                conn.commit()
        return {"message": "Session enregistrée avec succès", "session_id": session.sessionId}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("Go...")
    uvicorn.run(
        "custom_telemetry:app", host="localhost", port=8002, reload=True, log_level="debug"
    )
