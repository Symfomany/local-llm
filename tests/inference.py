import time
import requests
from openai import OpenAI
import weave
import re
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text


# Initialisation de Weave
WEAVE_PROJECT = "llm-project"
weave.init(WEAVE_PROJECT)

# Configuration du client OpenAI pour interagir avec le serveur vLLM
client = OpenAI(
    api_key="EMPTY",  # Pas besoin d'API key pour le serveur local
    base_url="http://localhost:8000/v1"  # Adresse du serveur vLLM
)

# Modèle utilisé
model = "/model/qwen2.5-coder-7b-instruct-q3_k_m.gguf"

# Prompt d'exemple
prompt = """Explique ceci sous Docker : 
FROM vllm/vllm-openai:latest"""


@weave.op()
def create_chat_completion(client, messages, temperature, top_p, extra_body):
    start_time = time.time()

    # Appel au modèle via vLLM
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        extra_body=extra_body,
        stream=True  # Activation du streaming
    )

    # Calcul de la latence totale (temps de réponse)
    total_latency = time.time() - start_time

    return response, start_time, total_latency


@weave.op()
def process_stream(chat_response_stream, start_time):
    full_generated_code = ""
    first_token_time = None

    for chunk in chat_response_stream:
        try:
            chunk_data = chunk.model_dump()
            message_content = chunk_data['choices'][0]['delta'].get(
                'content', '')

            if message_content and first_token_time is None:
                first_token_time = time.time() - start_time

            full_generated_code += message_content

        except Exception as e:
            print(f"Erreur : {str(e)}")

    return full_generated_code, first_token_time


@weave.op()
def log_metrics_to_weave(latency, first_token_time, prompt_tokens, generated_tokens):
    # Enregistrement des métriques dans Weave
    return {
        "latency": latency,
        "time_to_first_token": first_token_time,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens
    }


def get_vllm_metrics():
    # Endpoint des métriques exposé par vLLM (Prometheus format)
    metrics_url = "http://localhost:8000/metrics"
    response = requests.get(metrics_url)

    if response.status_code == 200:
        metrics_data = response.text

        # Extraction des métriques spécifiques (exemple simplifié)
        # Utilisation d'expressions régulières pour extraire les valeurs
        prompt_tokens_match = re.search(
            r'vllm:prompt_tokens_total\{.*?\}\s+([\d.]+)', metrics_data)
        generated_tokens_match = re.search(
            r'vllm:generation_tokens_total\{.*?\}\s+([\d.]+)', metrics_data)

        prompt_tokens = int(float(prompt_tokens_match.group(1))
                            ) if prompt_tokens_match else 0
        generated_tokens = int(
            float(generated_tokens_match.group(1))) if generated_tokens_match else 0

        return prompt_tokens, generated_tokens
    else:
        print("Erreur lors de la récupération des métriques vLLM")
        return 0, 0


# Ajoutez cette fonction à votre script
def format_llm_output(prompt, full_generated_code, metrics):
    console = Console()

    # Panel pour le prompt
    prompt_panel = Panel(
        Text(prompt, style="bold cyan"),
        title="[bold green]Prompt Original",
        border_style="bright_blue"
    )
    console.print(prompt_panel)

    # Syntax highlighting pour le code généré
    generated_code_syntax = Syntax(
        full_generated_code, 
        "python", 
        theme="monokai", 
        line_numbers=True
    )
    
    code_panel = Panel(
        generated_code_syntax, 
        title="[bold green]Réponse Générée", 
        border_style="bright_green"
    )
    console.print(code_panel)

    # Tableau des métriques
    metrics_table = Table(
        title="[bold magenta]Métriques de Performance LLM", 
        show_header=True, 
        header_style="bold cyan"
    )
    metrics_table.add_column("Métrique", style="dim")
    metrics_table.add_column("Valeur", style="bold")

    # Ajout des métriques au tableau
    for key, value in metrics.items():
        metrics_table.add_row(
            key.replace("_", " ").title(), 
            str(round(value, 4) if isinstance(value, float) else value)
        )

    console.print(metrics_table)



@weave.op()
def main():
    messages = [
        {"role": "system", "content": "Tu es un développeur français de chez Enedis"},
        {"role": "user", "content": prompt},
    ]

    # Création de la complétion avec suivi des métriques
    chat_response_stream, start_time, latency = create_chat_completion(
        client,
        messages,
        temperature=0.7,
        top_p=0.8,
        extra_body={"repetition_penalty": 1.05}
    )

    # Traitement du stream et calcul du temps au premier token
    full_generated_code, first_token_time = process_stream(
        chat_response_stream, start_time)

    # Récupération des métriques depuis le serveur vLLM
    prompt_tokens, generated_tokens = get_vllm_metrics()

    metrics = log_metrics_to_weave(latency, first_token_time,
                                   prompt_tokens, generated_tokens)
    
       # Utilisez la nouvelle fonction pour formater la sortie
    format_llm_output(prompt, full_generated_code, metrics)

    print("\n")
    print("[bold yellow]✨ Génération terminée et métriques enregistrées ![/bold yellow]")


if __name__ == "__main__":
    main()
