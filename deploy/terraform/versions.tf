terraform {
  required_version = ">= 1.6"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }

  # Recommended: keep state in a GCS bucket.
  # backend "gcs" {
  #   bucket = "my-terraform-state"
  #   prefix = "polyrag"
  # }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
