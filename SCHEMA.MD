# Schema GCP Explicatif

```mermaid
graph TD
    A[Développeur] -->|Utilise| B[Cloud Shell]
    B -->|Crée| C[Compte de Service IAM]
    B -->|Utilise| D[Hugging Face CLI]
    D -->|Télécharge| E[Modèle LLM google/gemma-2-2b-it]
    B -->|Crée| F[Dockerfile]
    B -->|Crée| G[cloudbuild.yaml]
    B -->|Exécute| H[Cloud Build]
    H -->|Construit et pousse l'image| I[Artifact Registry]
    H -->|Déploie| J[Cloud Run avec GPU]
    C -->|Fournit autorisations| J
    E -->|Intégré dans| J
    K[Utilisateur] -->|Requête API| J

    subgraph "Préparation"
        C
        D
        E
        F
        G
    end

    subgraph "Déploiement"
        H
        I
    end

    subgraph "Exécution"
        J
    end

```


# Voici un schéma Mermaid explicatif des flux et de l'architecture de votre projet FastAPI avec Ollama, déployé sur Google Cloud Run :

```mermaid
flowchart TD
    subgraph Local_Development
        A["FastAPI Application"] --> B["Ollama Server"]
        B --> C["Docker Container"]
    end

    subgraph Cloud_Deployment
        C --> D["Cloud Run"]
        D --> E["GCP Load Balancer"]
        E --> F["Client Requests"]
    end

    subgraph Model_Management
        B --> G["Model Pull (qwen:2.5-7b-instruct)"]
    end
```

Ce schéma illustre les principaux composants et flux de votre architecture :

## Local Development
- Votre application FastAPI communique avec le serveur Ollama.
- Le serveur Ollama et l'application sont encapsulés dans un conteneur Docker.

## Cloud Deployment
- Le conteneur Docker est déployé sur Google Cloud Run.
- Un équilibreur de charge GCP gère les requêtes entrantes.
- Les clients envoient des requêtes à l'API via l'équilibreur de charge.

## Model Management
- Le serveur Ollama télécharge et gère le modèle LLM (qwen:2.5-7b-instruct).

Cette architecture permet une scalabilité efficace et une gestion simplifiée du déploiement, tout en offrant les performances nécessaires pour exécuter des modèles LLM via Ollama.