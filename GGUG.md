# Load GGUF


* --model microsoft/Phi-3-mini-4k-instruct:
Ce paramètre spécifie le modèle à utiliser. Dans ce cas, il s'agit du modèle Phi-3-mini-4k-instruct, qui est une version optimisée du modèle Phi-3 pour des tâches d'instruction. Le préfixe "microsoft/" indique que ce modèle est hébergé sur la **plateforme Hugging Face **et développé par Microsoft.

* --gpu-memory-utilization 0.95:
Ce paramètre définit la *fraction de la mémoire GPU que le modèle peut utiliser*. Ici, 0.95 signifie que 95% de la mémoire GPU disponible sera allouée au modèle. Cela permet d'optimiser l'utilisation des ressources, mais il est important de laisser un *peu de marge pour éviter les erreurs d'out-of-memory* (OOM).

--max-model-len 1024
Ce paramètre fixe la longueur maximale de la* séquence d'entrée que le modèle peut traiter*. Une valeur de 1024 signifie que le modèle peut gérer des entrées allant *jusqu'à 1024 tokens*. Cela est particulièrement utile pour les modèles de langage, car une longueur maximale trop élevée peut dépasser les capacités du cache KV ou de la mémoire GPU.

--kv-cache-dtype auto
Ce paramètre détermine le type de données utilisé pour *le cache KV (Key-Value)*. En choisissant "auto", vous permettez à vLLM de sélectionner automatiquement le type de données le plus adapté en fonction de votre matériel et des besoins du modèle. Cela peut aider à optimiser les performances et l'utilisation de la mémoire.

--max-num-seqs 64
Ce paramètre définit le nombre *maximum de séquences que le modèle peut traiter simultanément*. Une valeur de 64 signifie que jusqu'à *64 séquences peuvent être traitées en parallèle* lors d'une *opération d'inférence*. Cela permet d'améliorer le débit et l'efficacité, mais nécessite également une *gestion adéquate de la mémoire GPU*.

- vLLM ne prend pas en charge les modèles GGUF. Il fonctionne avec des modèles en Safetensors (.safetensors) ou PyTorch (.bin ou .pt).
- Solution : Si ton modèle est en GGUF, tu dois utiliser llama.cpp au lieu de vLLM.


# Params

*model*: Spécifie le modèle à utiliser, ici un modèle personnalisé "qwen2.5-coder-7b-instruct-q3_k_m.gguf".
messages: Une liste de messages définissant le contexte de la conversation. Il y a un message système définissant le rôle de l'assistant et un message utilisateur avec la requête.
*temperature*: Contrôle la créativité des réponses (0.7 est une valeur modérée).
*top_p*: Contrôle la diversité des réponses (0.8 est une valeur standard).
*max_tokens*: Limite la longueur de la réponse à 512 tokens.
*extra_body*: Ajoute un paramètre supplémentaire pour la pénalité de répétition.
*stream=True*: Active le streaming de la réponse, permettant de recevoir la réponse au fur et à mesure qu'elle est générée.1


# Longeur:

1. Qwen2.5-Coder-7B-Instruct-Q3_K_M.gguf
Taille : Environ 3.81 Go.
Qualité : Considéré comme un modèle de faible qualité.
Utilisation : Adapté pour des cas d'utilisation où la mémoire est limitée, mais la qualité des réponses peut être compromise.
Caractéristiques : Utilise une méthode de quantification standard qui réduit la taille du modèle, mais peut affecter la précision des résultats.

2. Qwen2.5-Coder-7B-Instruct-IQ4_XS.gguf
Taille : Environ 4.22 Go.
Qualité : Offre une qualité décente, comparable à celle du modèle Q4_K_S, mais avec une taille de fichier plus petite.
Performance : Recommandé pour ceux qui cherchent un bon équilibre entre taille et performance, tout en maintenant une qualité acceptable pour les tâches de codage.
Caractéristiques : Basé sur des techniques de quantification qui optimisent le modèle pour des performances plus rapides tout en conservant une qualité raisonnable.

3. Qwen2.5-Coder-7B-Instruct-IQ2_M.gguf
Taille : Environ 2.78 Go.
Qualité : Considéré comme ayant une qualité relativement faible, mais utilise des techniques avancées pour rester utilisable dans certaines situations.
Utilisation : Idéal pour les utilisateurs ayant des contraintes de mémoire sévères et qui peuvent se permettre une qualité inférieure.
Caractéristiques : Ce modèle est conçu pour *être léger tout en restant fonctionnel* dans des scénarios spécifiques où un compromis sur la qualité est acceptable.


### **Comparaison avec d'autres variantes**
| Modèle                                    | Taille approx. | Qualité                | Utilisation typique                              |
|-------------------------------------------|----------------|------------------------|-------------------------------------------------|
| **Qwen2.5-Coder-7B-Instruct-Q3_K_M.gguf** | ~3.81 Go       | Faible                 | Mémoire très limitée                            |
| **Qwen2.5-Coder-7B-Instruct-IQ4_XS.gguf** | ~4.22 Go       | Moyenne                | Bon équilibre entre taille et performance       |
| **Qwen2.5-Coder-7B-Instruct-Q5_K_L.gguf** | ~5-6 Go        | Bonne                  | Bonne qualité avec un compromis sur la taille   |
| **Qwen2.5-Coder-7B-Instruct-IQ2_M.gguf**  | ~2.78 Go       | Relativement faible    | Cas extrêmes de contraintes mémoire             |

Pour choisir la version la plus adaptée à vos besoins, tenez compte de votre matériel et de vos exigences en termes de performance. 

Les versions avec une quantification plus élevée (par exemple, Q6_K_L) offrent généralement de *meilleures performances mais nécessitent plus de ressources*, tandis que les versions avec une quantification plus faible (par exemple, IQ3_XS) sont plus *légères mais peuvent offrir des performances réduites.*

# Code Gemma 2

google/codegemma-7b

    --model google/codegemma-2b
      --gpu-memory-utilization 0.90
      --max-model-len 512
      
      --kv-cache-dtype auto
      --max-num-seqs 128
      

# Microsoft 4k

command: >
      --model microsoft/Phi-3-mini-4k-instruct
      --gpu-memory-utilization 0.95
      --max-model-len 1024
      --kv-cache-dtype auto
      --max-num-seqs 128

 # Qwuen 2.5 Coder 1.5b

 # Qwuen 2.5 Coder 7b instruct M


 # Qwuen 2.5 Coder 7b instruct S