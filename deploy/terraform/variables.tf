variable "project_id" {
  description = "Google Cloud project ID."
  type        = string
}

variable "region" {
  description = "Region for Cloud Run and Artifact Registry."
  type        = string
  default     = "europe-west1"
}

variable "service_name" {
  description = "Cloud Run service name."
  type        = string
  default     = "polyrag"
}

variable "github_repository" {
  description = "GitHub repository allowed to deploy, as owner/name."
  type        = string
  default     = "alexandreryba/polyrag"
}

variable "llm_provider" {
  description = "anthropic (Claude API key in Secret Manager) or anthropic-vertex (Claude on Vertex AI)."
  type        = string
  default     = "anthropic"

  validation {
    condition     = contains(["anthropic", "anthropic-vertex"], var.llm_provider)
    error_message = "llm_provider must be anthropic or anthropic-vertex."
  }
}

variable "llm_model" {
  type    = string
  default = "claude-opus-5-5"
}

variable "llm_fast_model" {
  type    = string
  default = "claude-opus-5-5"
}

variable "allow_unauthenticated" {
  description = "Expose the service publicly; /v1 endpoints still require the X-API-Key secret."
  type        = bool
  default     = true
}

variable "min_instances" {
  type    = number
  default = 0
}

variable "max_instances" {
  type    = number
  default = 3
}
