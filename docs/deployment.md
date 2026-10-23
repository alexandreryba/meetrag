# Deploying PolyRAG on Google Cloud Run

The infrastructure lives in [`deploy/terraform`](../deploy/terraform); the
[`deploy.yml`](../.github/workflows/deploy.yml) workflow ships new images.

## What gets created

| Resource | Purpose |
|---|---|
| Artifact Registry repository `polyrag` | Docker images, last 10 kept |
| Cloud Run service `polyrag` | 2 vCPU, 2 GiB, 0 to 3 instances, probes on `/healthz` |
| Service account `polyrag-runtime` | Runtime identity: reads its secrets (and calls Vertex AI if enabled) |
| Secret Manager `polyrag-api-key`, `anthropic-api-key` | Injected as environment variables; values never pass through Terraform |
| Workload Identity pool `github` + provider | GitHub Actions authenticates with OIDC, restricted to this repository |
| Service account `polyrag-deployer` | Pushes images and deploys revisions |

With `llm_provider = "anthropic-vertex"`, no Anthropic key is needed: the runtime service
account gets `roles/aiplatform.user` and Claude is called through Vertex AI.

## Steps

```bash
cd deploy/terraform
cp example.tfvars terraform.tfvars   # set project_id
terraform init

# 1. Enable the APIs and create the (empty) secrets first
terraform apply -target=google_project_service.enabled -target=google_secret_manager_secret.secrets

# 2. Add the secret values (they never touch the Terraform state)
openssl rand -hex 24 | gcloud secrets versions add polyrag-api-key --data-file=-
echo -n "$ANTHROPIC_API_KEY" | gcloud secrets versions add anthropic-api-key --data-file=-

# 3. Create everything else
terraform apply
terraform output github_variables
```

Then, in the GitHub repository settings (**Settings → Secrets and variables → Actions →
Variables**), create the variables printed by `terraform output github_variables`, and an
environment named `production`.

Deploy by pushing a tag:

```bash
git tag v0.1.0 && git push origin v0.1.0
```

The workflow builds the image (the index and embedding model are baked in), pushes it, deploys
the revision and checks `/healthz`. Terraform ignores the image field, so infrastructure changes
and deployments do not conflict.

## Costs

With `min_instances = 0` the service scales to zero, so an idle demo costs close to nothing
(Artifact Registry storage and secrets). The main cost is LLM usage per question, reported by
`polyrag eval e2e`.
