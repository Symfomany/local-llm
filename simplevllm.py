
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
