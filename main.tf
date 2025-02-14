resource "google_compute_instance" "vllm_gpu" {
  name         = "vllm-gpu-instance-${var.gpu_type}"
  machine_type = var.enable_gpu ? "g2-standard-8" : "n1-standard-4"
  zone         = "${var.region}-a"

  boot_disk {
    initialize_params {
       image = "ubuntu-2204-lts" # Family dynamique
       size  = 200
       type  = "pd-ssd"
      }
  }

  network_interface {
    network = "default"
    access_config {
      network_tier = "PREMIUM"
    }
  }

  dynamic "guest_accelerator" {
    for_each = var.enable_gpu ? [1] : []
    content {
      type  = var.gpu_type
      count = 1
    }
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = false
  }

  metadata = {
    docker-image = var.docker_image
  }

  # Correction du chemin du script de démarrage
  metadata_startup_script = templatefile("${path.module}/correct_script_name.sh", {
    docker_image = var.docker_image
  })
}
