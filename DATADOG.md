
#  Datadog Agent

DD_LLMOBS_ENABLED=1 DD_LLMOBS_ML_APP=onboarding-quickstart \
DD_API_KEY=bb607f65-b13f-4522-a8cd-42b355ff2734 DD_SITE=https://app.datadoghq.eu \
DD_LLMOBS_AGENTLESS_ENABLED=1 ddtrace-run python tracing.py

# LLM : Observabilité des LLMs
L'observabilité des LLMs de Datadog fournit une visibilité de bout en bout sur la performance et le comportement de vos applications LLM. Les fonctionnalités clés incluent :

- Surveillance des performances : Suivez la latence, l'utilisation des tokens et les taux d'erreur dans vos applications LLM. Cela aide à identifier les goulets d'étranglement en matière de performance et à garantir que vos modèles fournissent des réponses efficacement.
- Analyse des causes profondes : Analysez les traces pour résoudre les problèmes au sein des flux de travail des LLM. Cette capacité vous permet de localiser la source des erreurs ou des sorties inattendues, comme les hallucinations ou les réponses non pertinentes.
- Contrôles de qualité : Utilisez des évaluations de qualité intégrées pour évaluer la qualité fonctionnelle des sorties de vos LLM. Cela inclut le suivi de métriques telles que le "taux d'échec à répondre" et la "pertinence du sujet", qui sont cruciales pour maintenir l'exactitude des réponses générées par vos modèles.

# Intégration avec les Frameworks d'IA
Datadog s'intègre parfaitement avec divers frameworks d'IA, vous permettant de surveiller les modèles déployés sur des plateformes comme NVIDIA Triton Inference Server ou Google Vertex AI. Cette intégration facilite :

- Visualisation des métriques clés : Obtenez des informations sur la latence d'inférence, l'utilisation des ressources (CPU, GPU) et la performance globale du modèle grâce à des tableaux de bord prêts à l'emploi adaptés aux applications d'IA.

- Suivi des interactions utilisateur : Surveillez les parcours et interactions des utilisateurs avec vos applications LLM pour vous assurer qu'elles répondent aux attentes en matière de performance dans différentes régions et démographies.

# Surveillance de la Sécurité
Datadog met également l'accent sur la sécurité en vous aidant à suivre les vulnérabilités potentielles dans vos applications LLM :
Détection d'injections dans les prompts : Surveillez les expositions à la sécurité telles que les injections dans les prompts qui pourraient entraîner des sorties inappropriées ou des fuites de données.
Analyse des données sensibles : Nettoyez automatiquement les informations personnellement identifiables (PII) dans les traces pour protéger les données utilisateur lors des interactions avec vos modèles.

# Métriques et Alertes Personnalisées
Vous pouvez définir des métriques personnalisées spécifiques aux besoins de votre application, comme le suivi des latences d'API ou des tailles de réponses. Datadog vous permet de configurer des alertes basées sur ces métriques, garantissant que vous êtes rapidement informé de toute anomalie ou problème de performance.

# Suivi des Événements
Utilisez les capacités de suivi d'événements de Datadog pour enregistrer des événements significatifs pendant les tests, tels que l'exécution de scripts ou les erreurs rencontrées lors des interactions avec le LLM. Cela aide à maintenir une trace d'audit et facilite le débogage.
Conclusion
En tirant parti des outils d'observabilité de Datadog pour les LLMs, vous pouvez efficacement surveiller, résoudre et améliorer la performance de vos modèles linguistiques tout en garantissant la sécurité et la conformité. Cette approche complète permet aux développeurs et aux ingénieurs en IA d'améliorer la fiabilité et l'efficacité de leurs applications en temps réel, faisant ainsi de Datadog un atout précieux pour toute organisation utilisant la technologie LLM.