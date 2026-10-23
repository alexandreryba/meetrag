locals {
  use_vertex = var.llm_provider == "anthropic-vertex"
  services = concat(
    [
      "run.googleapis.com",
      "artifactregistry.googleapis.com",
      "secretmanager.googleapis.com",
      "iamcredentials.googleapis.com",
      "sts.googleapis.com",
    ],
    local.use_vertex ? ["aiplatform.googleapis.com"] : [],
  )
  # Secrets are created empty here; values are added out of band so they never reach the
  # Terraform state:  echo -n "$KEY" | gcloud secrets versions add <name> --data-file=-
  secrets = local.use_vertex ? ["polyrag-api-key"] : ["polyrag-api-key", "anthropic-api-key"]
}

resource "google_project_service" "enabled" {
  for_each           = toset(local.services)
  service            = each.value
  disable_on_destroy = false
}

# --- Container registry ---------------------------------------------------------------------
resource "google_artifact_registry_repository" "polyrag" {
  location      = var.region
  repository_id = "polyrag"
  format        = "DOCKER"
  description   = "PolyRAG API images"

  cleanup_policies {
    id     = "keep-last-10"
    action = "KEEP"
    most_recent_versions {
      keep_count = 10
    }
  }

  depends_on = [google_project_service.enabled]
}

# --- Runtime identity and secrets -------------------------------------------------------------
resource "google_service_account" "runtime" {
  account_id   = "${var.service_name}-runtime"
  display_name = "PolyRAG Cloud Run runtime"
}

resource "google_secret_manager_secret" "secrets" {
  for_each  = toset(local.secrets)
  secret_id = each.value

  replication {
    auto {}
  }

  depends_on = [google_project_service.enabled]
}

resource "google_secret_manager_secret_iam_member" "runtime_access" {
  for_each  = google_secret_manager_secret.secrets
  secret_id = each.value.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.runtime.email}"
}

resource "google_project_iam_member" "runtime_vertex" {
  count   = local.use_vertex ? 1 : 0
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.runtime.email}"
}

# --- Cloud Run service --------------------------------------------------------------------------
resource "google_cloud_run_v2_service" "api" {
  name                = var.service_name
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false

  template {
    service_account                  = google_service_account.runtime.email
    max_instance_request_concurrency = 10
    timeout                          = "120s"

    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }

    containers {
      # Placeholder until the first CI deployment; the deploy workflow then owns the image.
      image = "us-docker.pkg.dev/cloudrun/container/hello"

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
        cpu_idle          = true
        startup_cpu_boost = true
      }

      env {
        name  = "POLYRAG_LLM_PROVIDER"
        value = var.llm_provider
      }
      env {
        name  = "POLYRAG_LLM_MODEL"
        value = var.llm_model
      }
      env {
        name  = "POLYRAG_LLM_FAST_MODEL"
        value = var.llm_fast_model
      }
      dynamic "env" {
        for_each = local.use_vertex ? [1] : []
        content {
          name  = "POLYRAG_GCP_PROJECT"
          value = var.project_id
        }
      }
      env {
        name = "POLYRAG_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.secrets["polyrag-api-key"].secret_id
            version = "latest"
          }
        }
      }
      dynamic "env" {
        for_each = local.use_vertex ? [] : [1]
        content {
          name = "ANTHROPIC_API_KEY"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.secrets["anthropic-api-key"].secret_id
              version = "latest"
            }
          }
        }
      }

      startup_probe {
        http_get {
          path = "/healthz"
        }
        initial_delay_seconds = 2
        period_seconds        = 3
        failure_threshold     = 20
      }

      liveness_probe {
        http_get {
          path = "/healthz"
        }
        period_seconds = 30
      }
    }
  }

  lifecycle {
    ignore_changes = [
      template[0].containers[0].image, # rolled out by .github/workflows/deploy.yml
      client,
      client_version,
    ]
  }

  depends_on = [google_secret_manager_secret_iam_member.runtime_access]
}

resource "google_cloud_run_v2_service_iam_member" "public" {
  count    = var.allow_unauthenticated ? 1 : 0
  name     = google_cloud_run_v2_service.api.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}
