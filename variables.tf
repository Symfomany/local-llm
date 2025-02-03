variable "project_id" {
  description = "decent-destiny-448418-p1"
  type        = string
}

variable "region" {
  description = "Région GCP"
  type        = string
  default     = "europe-west4"
}

variable "gpu_type" {
  description = "Type de GPU NVIDIA (l4, v100, etc.)"
  type        = string
  default     = "nvidia-l4"
  validation {
    condition     = contains(["nvidia-l4", "nvidia-tesla-v100", "nvidia-tesla-t4"], var.gpu_type)
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