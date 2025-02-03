output "instance_ip" {
  description = "IP publique de l'instance"
  value       = google_compute_instance.vllm_gpu.network_interface[0].access_config[0].nat_ip
}

output "gpu_status" {
  description = "Statut de l'accélérateur GPU"
  value       = var.enable_gpu ? "Activé (${var.gpu_type})" : "Désactivé"
}
