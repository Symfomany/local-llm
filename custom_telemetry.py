from __future__ import annotations

import sys
from contextlib import asynccontextmanager

import psycopg2
import uvicorn
from dataclasses import dataclass, field
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import List
from typing import Dict
from typing import List, Union, Dict, Any, Optional
import csv
import json
import os
from datetime import datetime
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
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
    sessionId: str = ""
    session: Optional[Dict[str, Any]] = None
    action: str = "chat"
    created: Optional[Union[str, int]]
    fileInfo: Optional[Dict[str, Any]] = None
    filepath: Optional[str] = None
    startLine: Optional[Any] = None
    endLine: Optional[Any] = None
    text: Optional[Any] = None
    title: Optional[Any] = None
    curSelectedModelTitle: Optional[Any] = None
    command: Optional[str] = None
    ide: Optional[str] = Field(default="VSCode", description="IDE used (VSCode or IntelliJ)")
    history:  Union[List[Dict[str, Any]], str] = ""

    model_config = ConfigDict(arbitrary_types_allowed=True)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    details = exc.errors()
    print(f"Validation Error: {details}")  # Affiche les détails de l'erreur dans la console
    return JSONResponse(
        status_code=422,
        content={"detail": details},
    )
    
@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@app.post("/sessions")
async def create_session(session: Session):
    """Enregistre ou met à jour une session dans la base de données."""
    try:
        async with db_connection() as conn:
            with conn.cursor() as cur:

                try:
                    # Imprime la charge utile de la requête
                    print("Request Payload", session)

                    # Convertit l'objet Session en un dictionnaire Python
                    session_dict = session.dict()

                    # Vérifie si le fichier existe
                    if os.path.exists("metrics.json"):
                        # Lit le contenu existant du fichier
                        with open("metrics.json", "r") as infile:
                            try:
                                data = json.load(infile)
                            except json.JSONDecodeError:
                                # Si le fichier est vide ou corrompu, initialise avec une structure de base
                                data = {"datas": []}
                    else:
                        # Si le fichier n'existe pas, initialise avec une structure de base
                        data = {"datas": []}

                    # Ajoute les données de la session à la liste "datas"
                    data["datas"].append(session_dict)

                    # Écrit les données mises à jour dans le fichier JSON
                    with open("metrics.json", "w") as outfile:
                        # Ecrit le dictionnaire avec indentation pour la lisibilité
                        json.dump(data, outfile, indent=4, default=str)

                    return {"message": "Session enregistrée et ajoutée à metrics.json sous la clé 'datas'"}

                except Exception as e:
                    print(
                        f"Erreur lors de l'enregistrement de la session : {e}")
                    return {"error": str(e)}
                    # cur.execute(
                    #     """
                    #     INSERT INTO history (history_json, ide, session_id, action)
                    #     VALUES (%s, %s, %s, %s)
                    #     ON CONFLICT (session_id) DO UPDATE
                    #     SET history_json = COALESCE(EXCLUDED.history_json, history.history_json),
                    #         ide = COALESCE(EXCLUDED.ide, history.ide),
                    #         action = COALESCE(EXCLUDED.action, history.action)
                    #     """,
                    #     (json.dumps(session.history), session.ide, session.sessionId, session.action,
                    #      json.dumps(session.history), session.ide, session.action),
                    # )
                    # conn.commit()
                    return {"ok": True}
                    # return {"message": "Session enregistrée/mise à jour avec succès", "session_id": session.sessionId}
                except psycopg2.Error as sql_error:
                    conn.rollback()  # Annuler la transaction en cas d'erreur
                    raise HTTPException(
                        status_code=500, detail=f"Erreur SQL : {sql_error.diag.message_detail or sql_error}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Define the User model for Login


class User(BaseModel):
    email: str = ""
    password: str = ""
    keyApi: str = ""

# Define the User model for CSV


class UserCSV(BaseModel):
    prenom: str
    nom: str
    nni: str
    email: str
    pole: str
    projet: str
    api_key: str

# Load users from the CSV file


def load_users_from_csv(csv_file: str) -> list[UserCSV]:
    users = []
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            users.append(UserCSV(**row))
    return users


users_data = load_users_from_csv('licenses.csv')

# Endpoint to authenticate user


@app.post("/login")
async def login(user: User):
    print(user, "user", users_data)
    for user_csv in users_data:
        print("user.keyApi", user.keyApi)
        if user_csv.email == user.email and user_csv.api_key == user.keyApi:
            return {'ok': True, 'message': 'Login successful', 'user': user_csv}
            break

    # Raise exception if no match is found
    raise HTTPException(status_code=401, detail="Incorrect login credentials")


@app.exception_handler(500)
async def internal_exception_handler(request: Request, exc: Exception):
    # Affiche les détails de l'erreur dans la console
    

    return JSONResponse(
        status_code=500,
        content=jsonable_encoder(
            {"code": 500, "msg": "Internal Server Error"}),
    )


if __name__ == "__main__":
    uvicorn.run(
        "custom_telemetry:app", host="localhost", port=8002, reload=True, log_level="debug"
    )
