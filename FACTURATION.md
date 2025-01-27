# Facturation

* Vous devez utiliser *la facturation basée sur les instances* pour pouvoir utiliser les GPU.
* Le *GPU L4 est facturé pendant toute la durée du cycle de vie de l'instance, même lorsqu'elle est inactive*.
* Il n'y a pas de *frais par requête pour les instances utilisant des GPU*.
* Vous devez configurer *au minimum 4 CPU et 16 Go de mémoire pour utiliser un GPU L4*.

* Les instances peuvent être réduites à zéro pour économiser des coûts lorsqu'elles ne sont pas utilisées, *mais les instances minimales configurées* seront toujours facturées au tarif plein.

* Le démarrage d'une instance avec GPU L4 prend environ 5 secondes3.
* Vous ne pouvez utiliser qu'un seul GPU L4 par instance Cloud Run

* Lorsque le nombre d'instances minimales est défini à 0, Cloud Run peut réduire le nombre d'instances à zéro en l'absence de trafic.
* Sans instances en cours d'exécution et sans requêtes entrantes, il n'y a pas de consommation de ressources facturables.
* Ce *comportement permet une optimisation des coûts*, car vous ne payez que lorsque votre service est réellement utilisé


# Cold Start

Le démarrage à froid de Cloud Run peut effectivement prendre un certain temps, mais Google a travaillé pour optimiser ce processus. Voici ce qui se passe pendant le démarrage à froid :

1. Téléchargement de l'image : Cloud Run utilise* une technologie de streaming d'image de conteneur* pour accélérer ce processus.
2. Démarrage du conteneur : Le système exécute la *commande d'entrée (entrypoint) du conteneur*.
3. Attente de l'écoute : Le système attend que le conteneur commence à écouter sur le port configuré.
4. Chargement du modèle (pour l'IA) : Dans le cas des workloads d'IA, le modèle est chargé dans le GPU.
5. Initialisation du framework : Les frameworks comme Ollama, vLLM ou PyTorch sont initialisés.

Pour les instances Cloud Run avec GPU L4, Google annonce un temps de démarrage d'*environ 5 secondes pour que le conteneur soit prêt à utiliser le GPU*. Ensuite, quelques secondes supplémentaires sont nécessaires pour charger et initialiser le framework et le modèle.

Pour des modèles d'IA légers, Google a fourni des exemples de temps de démarrage à froid allant de *11 à 35 secondes*, ce qui inclut *le démarrage de l'instance, le chargement du modèle, et la génération du premier mot*.

Pour réduire l'impact des démarrages à froid, vous pouvez :

- Optimiser votre code et vos dépendances pour un *démarrage rapide.*
- Utiliser l'*optimisation du processeur* au démarrage pour réduire la latence.
- Configurer un *nombre minimal d'instances* pour maintenir des instances "chaudes".

Il est important de noter que le démarrage à froid ne se produit que *lorsqu'une nouvelle instance doit être activée*, et pas à chaque requête.

