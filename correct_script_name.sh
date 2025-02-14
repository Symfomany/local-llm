#!/bin/bash

# Mise à jour des packages
sudo apt update

# Installation des pilotes NVIDIA et CUDA
sudo apt install -y nvidia-driver-525 nvidia-cuda-toolkit

# Installation de Docker (si nécessaire)
sudo apt install -y docker.io

# Démarrage du service Docker
sudo systemctl start docker
sudo systemctl enable docker


# Pull de l'image Docker spécifiée
sudo docker pull ${docker_image}

# Exécution de l'image Docker
sudo docker run -d ${docker_image}
