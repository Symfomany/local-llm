Bien sûr, je peux vous aider à configurer ces variables pour gcloud CLI. Voici les commandes pour définir ces variables d'environnement dans votre session de terminal actuelle :


# Build Cloud Run

## Deploy vLLM 
```bash
gcloud beta run deploy vllm-server \
  --image vllm/vllm-openai:latest \
  --platform managed \
  --region europe-west4 \
  --cpu 4 \
  --memory 16Gi \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --set-env-vars HUGGING_FACE_HUB_TOKEN=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 \
  --set-env-vars MODEL_NAME=Qwen/Qwen2.5-Coder-1.5B \
  --timeout 3600 \
  --args "vllm.entrypoints.openai.api_server","--model","Qwen/Qwen2.5-Coder-1.5B","--host","0.0.0.0","--port","8000","--trust-remote-code" \
  --port 8000 \
  --no-cpu-throttling
```


# Déployer sur Cloud Run avec variables (apres artifact)

```bash
gcloud beta run deploy $SERVICE_NAME \
    --image=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$SERVICE_NAME \
    --platform=managed \
    --cpu 4 \
    --region europe-west4 \
    --timeout 3600 \
    --memory 16Gi \
    --gpu 1 \
    --port 8000 \
    --no-cpu-throttling \
    --allow-unauthenticated \
    --set-env-vars=HUGGING_FACE_API_KEY=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,MODEL_NAME=Qwen/Qwen2.5-Coder-1.5B


gcloud beta run deploy vllm-server \
    --image=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$SERVICE_NAME \
    --platform=managed \
    --allow-unauthenticated \
    --region=$REGION \
    --set-env-vars=HUGGING_FACE_API_KEY=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,MODEL_NAME=Qwen/Qwen2.5-Coder-1.5B \
    --cpu=4 \
    --memory 16Gi \
    --min-instances=1 \
    --max-instances=2
```

# Créer dans ma regfion  un artifact basé sur Docker:
```bash
gcloud artifacts repositories create my-docker-repo \
    --repository-format=docker \
    --location=europe-west4 
```

# Construire et taguer vos images Docker
```bash
docker build -t europe-west4-docker.pkg.dev/decent-destiny-448418-p1/my-docker-repo/vllm_server .
```

# Pusher mon image docker
```bash
docker push europe-west4-docker.pkg.dev/decent-destiny-448418-p1/my-docker-repo/vllm_server
```

# Verifier: avec variables
```bash
gcloud run deploy vllm-server \
    --image=[REGION]-docker.pkg.dev/[PROJECT_ID]/my-docker-repo/vllm_server \
    --platform=managed \
    --allow-unauthenticated \
    --set-env-vars=HUGGING_FACE_API_KEY=your_token,MODEL_NAME=your_model
```
# Lister les repositories des artifact
```bash
gcloud artifacts repositories list --location=europe-west4
```

# Read the logs
```bash
gcloud run services logs read vllm-server --region europe-west4
```

# Builder en local
docker build -t europe-west2-docker.pkg.dev/decent-destiny-448418-p1/my-docker-repo/vllm-server .



# Tester en lcoal
docker push europe-west2-docker.pkg.dev/decent-destiny-448418-p1/my-docker-repo/vllm-server



# Syn Docker avec  Artifacty Registry
gcloud auth configure-docker europe-west2-docker.pkg.dev



# Test Curl 
```bash
  curl -X POST https://vllm-server-666271192600.europe-west4.run.app/v1/completions \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -H "Content-Type: application/json" \
  -d '{
      "model": "Qwen/Qwen2.5-Coder-1.5B",
      "prompt": "Écrivez une fonction Python pour calculer la factorielle",
      "max_tokens": 100,
      "temperature": 0.7
  }'
```

# Permission Artifact Repository
```bash
gcloud auth login
gcloud auth configure-docker europe-west4-docker.pkg.dev
```

Service [vllm-server] revision [vllm-server-00001-qbz] has been deployed and is serving 100 percent of traffic.
Service URL: https://vllm-server-666271192600.europe-west4.run.app


```bash
export PROJECT_ID=decent-destiny-448418-p1
export REGION=europe-west4
export SERVICE_NAME=llm-runtime
```



Ces commandes définissent les variables d'environnement que vous pourrez utiliser dans vos commandes gcloud. Pour vérifier que les variables ont été correctement définies, vous pouvez utiliser la commande `echo` :

```bash
echo $PROJECT_ID
echo $REGION
echo $SERVICE_NAME
```

Maintenant, vous pouvez utiliser ces variables dans vos commandes gcloud. Par exemple :

```bash
gcloud config set project $PROJECT_ID
gcloud config set run/region $REGION
```

Puis auth:
```bash
gcloud auth login
```

Pour utiliser ces variables dans une commande de déploiement Cloud Run, vous pouvez faire comme suit :

```bash
gcloud run deploy $SERVICE_NAME --source . --region $REGION --project $PROJECT_ID
```

N'oubliez pas que ces variables d'environnement ne persisteront que pour la durée de votre session de terminal actuelle. Si vous voulez les rendre permanentes, vous devrez les ajouter à votre fichier de configuration de shell (comme .bashrc ou .zshrc).


Run my instance

```bash
 gcloud run deploy qwen-api   --source .   --platform managed   --region us-central1   --allow-unauthenticated   --cpu 4   --memory 16Gi   --timeout 3600   --set-env-vars OLLAMA_MODELS=/models  --set-env-vars PORT=8080
 ```

 Cette erreur indique que votre conteneur n'a pas réussi à démarrer et à écouter sur le port 8080 comme attendu par Cloud Run. Voici les principales causes possibles et les solutions à essayer :

## Vérification du code

Assurez-vous que votre application écoute bien sur le port défini par la variable d'environnement PORT. Par exemple, en Node.js avec Express :

```javascript
const port = process.env.PORT || 8080;
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
```

## Configuration du Dockerfile

Vérifiez que votre Dockerfile expose le bon port :

```dockerfile
EXPOSE 8080
```

## Temps de démarrage

Si votre application prend trop de temps à démarrer, augmentez le délai d'attente du contrôle de santé :

```bash
gcloud run deploy qwen-api --timeout 300
```

## Problèmes de dépendances

Assurez-vous que toutes les dépendances nécessaires sont correctement installées dans votre conteneur.

## Logs d'application

Consultez les logs de votre application via le lien Ouvrir Cloud Logging" fourni dans le message d'erreur pour obtenir plus d'informations sur la cause de l'échec.

## Test local

Essayez de construire et d'exécuter votre conteneur localement pour vérifier s'il fonctionne correctement :

```bash
docker build -t qwen-api .
docker run    --gpus all  -p 8080:8080 -e PORT=8080 qwen-api 
docker run --name qwen-container   --gpus all  -p 8080:8080 -e PORT=8080 qwen-api 
```

Examples of ollama:
https://github.com/ollama/ollama-python/tree/main/examples

Reconsrtuire 

```bash
docker build -t qwen-api .  --no-cache
``` 

```bash
 ollama create alpha -f Modelfile
 ollama generate --model  alpha -f "Generate a simple hello world program in Python"
``` 

## Plateforme de construction

Si vous utilisez un Mac M1/M2, assurez-vous de construire votre image pour la bonne architecture :

```bash
docker buildx build --platform linux/amd64 -t qwen-api .
```

Si le problème persiste après avoir vérifié ces points, examinez attentivement les logs de l'application pour identifier d'éventuelles erreurs spécifiques à votre code ou à votre configuration[1][7][15].
