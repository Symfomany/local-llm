variable "project_id" {
  description = "ID du projet GCP"
  type        = string
  default     = "decent-destiny-448418-p1"
}

variable "region" {
  description = "Région GCP"
  type        = string
  default     = "europe-west4"
  validation {
    condition     = contains(["europe-west4"], var.region)
    error_message = "Région non supportée"
  }
}

variable "gpu_types" {
  description = "Liste des types de GPU supportés"
  type        = list(string)
  default     = ["nvidia-l4", "nvidia-tesla-v100", "nvidia-tesla-t4"]
}

variable "gpu_type" {
  description = "Type de GPU NVIDIA"
  type        = string
  default     = "nvidia-l4"
  validation {
    condition     = contains(var.gpu_types, var.gpu_type)
    error_message = "Type de GPU non supporté"
  }
}

variable "docker_image" {
  description = "Image Docker à déployer"
  type        = string
  default     = "test_vllm:latest"
}

variable "enable_gpu" {
  description = "Activer l'Accélérateur GPU"
  type        = bool
  default     = true
}
