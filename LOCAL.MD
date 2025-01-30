## Test local

Essayez de construire et d'exécuter votre conteneur localement pour vérifier s'il fonctionne correctement :

```bash
docker build -t qwen-api .
docker run  --gpus all  -p 8080:8080 -e PORT=8080 qwen-api 

docker run   --runtime nvidia  --gpus all -p 8080:8080 -v //c/Users/julie/Desktop/gcloudllm/CloudRun-LLM/model:/model -e PORT=8080 qwen-api


docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    vllm/vllm-openai \
    --model Qwen/Qwen2-7B-Instruct

```