from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
import deepeval
import requests
import json

deepeval.login_with_confident_api_key("aucxOCaP6YCAu2s063a1J037PBiPCdNOCRaEEVl6yiA=")

# Création d'une métrique personnalisée


class CustomRelevancyMetric:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def evaluate(self, test_case):
        input_text = test_case.input
        actual_output = test_case.actual_output
        retrieval_context = test_case.retrieval_context

        # Exemple simple : compter les mots clés
        keywords = [
            word for sentence in retrieval_context for word in sentence.split()]
        matching_words = [
            word for word in actual_output.split() if word in keywords]

        relevancy_score = len(matching_words) / \
            len(keywords) if keywords else 0

        return relevancy_score >= self.threshold

def call_api(prompt):
    # Définition de l'URL de votre API
    api_url = "http://localhost:8080/v1/chat/completions"

    # Préparation des données de requête
    data = {
        "model": "Qwen/Qwen2.5-Coder-3B-Instruct",  # Ajustez selon vos besoins
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 100, 
        "temperature": 0.7, 
        "top_p": 0.95, 
        "frequency_penalty": 0.0, 
        "presence_penalty": 0.0, 
    }
    
    headers = {'Content-Type': 'application/json'}
    # Envoi de la requête
    response = requests.post(api_url, data=json.dumps(
        data), headers=headers, timeout=10)

   # Traitement de la réponse
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    elif response.status_code == 422:
        print(f"Erreur 422 : {response.text}")
        raise Exception(f"Erreur API : {response.status_code}")
    else:
        raise Exception(f"Erreur API : {response.status_code}")


def run_test_case(prompt, retrieval_context):
    # answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
    custom_metric = CustomRelevancyMetric(threshold=0.5)
    actual_output = call_api(prompt)

    test_case = LLMTestCase(
        input=prompt,
        actual_output=actual_output,
        retrieval_context=retrieval_context
    )

    # Exécution du test avec la métrique personnalisée
    result = custom_metric.evaluate(test_case)
    print(f"Prompt : {prompt}")
    print(f"Réponse : {actual_output}")
    print(f"Réponse pertinente : {result}\n")
    
# Exécution de plusieurs tests
tests = [
    {
        "prompt": "Write a Python function to calculate the factorial of a number.",
        "retrieval_context": ["Factorial is muste be done."]
    },
    {
        "prompt": "Write a Python function to find the maximum number in a list.",
        "retrieval_context": ["To find the maximum number in a list, you can iterate through the list and keep track of the largest number encountered."]

    }
]

for test in tests:
    run_test_case(test["prompt"], test["retrieval_context"])