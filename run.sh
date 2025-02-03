#!/bin/bash
# Installation des dépendances système
apt-get update && apt-get install -y \
    pciutils \
    curl \
    docker.io \
    nvidia-cuda-toolkit

# Configuration Docker pour GPU
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update && apt-get install -y nvidia-container-toolkit
systemctl restart docker

# Déploiement du conteneur
docker pull ${docker_image}
docker run -d --gpus all -p 8000:8000 ${docker_image}
