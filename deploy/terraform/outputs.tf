output "service_url" {
  description = "Public URL of the API and demo UI."
  value       = google_cloud_run_v2_service.api.uri
}

output "image_repository" {
  description = "Artifact Registry path for images."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.polyrag.repository_id}"
}

# Values to set as GitHub Actions repository variables for deploy.yml.
output "github_variables" {
  value = {
    GCP_PROJECT_ID             = var.project_id
    GCP_REGION                 = var.region
    CLOUD_RUN_SERVICE          = google_cloud_run_v2_service.api.name
    GCP_WIF_PROVIDER           = google_iam_workload_identity_pool_provider.github.name
    GCP_DEPLOY_SERVICE_ACCOUNT = google_service_account.deployer.email
  }
}
