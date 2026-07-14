# 🚀 Deployment Guide — GCP Cloud Run

This guide walks through deploying the Haett Churn Prediction API to **Google Cloud Run** — a fully managed serverless container platform.

## Prerequisites

- A **Google Cloud Platform** account ([free tier available](https://cloud.google.com/free))
- A **GitHub** repository with this code pushed to `main`
- `gcloud` CLI ([install guide](https://cloud.google.com/sdk/docs/install))

---

## Step 1: Create a GCP Project

```bash
# Set your project ID (must be globally unique)
export PROJECT_ID="haett-churn-$(whoami)-$(date +%s)"
gcloud projects create $PROJECT_ID --name="Haett Churn Prediction"
gcloud config set project $PROJECT_ID
```

## Step 2: Enable Required APIs

```bash
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  iamcredentials.googleapis.com
```

## Step 3: Create a Service Account for GitHub Actions

```bash
# Create service account
gcloud iam service-accounts create haett-gh-deployer \
  --display-name="GitHub Actions Deployer"

# Grant permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:haett-gh-deployer@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:haett-gh-deployer@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:haett-gh-deployer@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.admin"
```

## Step 4: Create Workload Identity Federation (Recommended)

This lets GitHub Actions authenticate without storing long-lived keys.

```bash
# Create a workload identity pool
gcloud iam workload-identity-pools create "github-pool" \
  --location="global" \
  --display-name="GitHub Actions Pool"

# Get the pool ID
export POOL_ID=$(gcloud iam workload-identity-pools describe "github-pool" \
  --location="global" --format="value(name)")

# Create an OIDC provider for GitHub
gcloud iam workload-identity-pools providers create-oidc "github-actions" \
  --location="global" \
  --workload-identity-pool="github-pool" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.actor=assertion.actor" \
  --attribute-condition="assertion.repository == 'YOUR_GITHUB_USERNAME/Haett_MLOps_Intern_Assessment_'"
```

> **Replace `YOUR_GITHUB_USERNAME`** with your actual GitHub username.

## Step 5: Grant the Service Account Access via Workload Identity

```bash
# Get the WIF provider resource name
export WIF_PROVIDER=$(gcloud iam workload-identity-pools providers describe "github-actions" \
  --location="global" \
  --workload-identity-pool="github-pool" \
  --format="value(name)")

# Allow the service account to be impersonated from GitHub
gcloud iam service-accounts add-iam-policy-binding \
  "haett-gh-deployer@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/$POOL_ID/attribute.repository/YOUR_GITHUB_USERNAME/Haett_MLOps_Intern_Assessment_"
```

## Step 6: Add Secrets to GitHub Repository

Go to your GitHub repo → **Settings** → **Secrets and variables** → **Actions** → Add these secrets:

| Secret | Value |
|--------|-------|
| `GCP_PROJECT_ID` | Your GCP project ID |
| `GCP_WIF_PROVIDER` | Full WIF provider name (from Step 5) |
| `GCP_SERVICE_ACCOUNT` | Email of the deployer SA: `haett-gh-deployer@YOUR_PROJECT_ID.iam.gserviceaccount.com` |
| `GCP_REGION` | (Optional) Default: `us-central1` |

## Step 7: Deploy!

Push a version tag to trigger the CD workflow:

```bash
git tag v1.0.0
git push origin v1.0.0
```

This triggers the `.github/workflows/cd.yml` workflow which will:

1. ✅ Run the full ML pipeline (trains the model)
2. ✅ Build a Docker image and push to Google Container Registry
3. ✅ Deploy to Cloud Run
4. ✅ Run a health check against the deployed service

---

## Manual Deployment (Without GitHub Actions)

If you prefer to deploy directly from your machine:

```bash
# 1. Build and push the image
gcloud builds submit --tag gcr.io/$PROJECT_ID/haett-churn-api

# 2. Deploy to Cloud Run
gcloud run deploy haett-churn-api \
  --image gcr.io/$PROJECT_ID/haett-churn-api \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 10 \
  --timeout 300
```

---

## Post-Deployment

### Test the API

```bash
# Get the service URL
export SERVICE_URL=$(gcloud run services describe haett-churn-api \
  --region us-central1 --format="value(status.url)")

# Health check
curl $SERVICE_URL/health

# Make a prediction
curl -X POST $SERVICE_URL/predict \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "days_since_last_order": 5, "total_orders": 15, "total_spent": 500, "avg_order_value": 33.33, "avg_rating": 4.2, "total_support_tickets": 1, "age": 32, "is_sub_active": true, "subscription_tenure_days": 180, "tenure_days": 200}'
```

### View Logs

```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=haett-churn-api" --limit 20
```

### Run Drift Detection (on the deployed service's data)

```bash
python scripts/check_drift.py --generate
```

---

## Monitoring & Alerts

- **Cloud Monitoring**: Automatic metrics for request count, latency, error rate
- **Cloud Logging**: Automatic log collection with search
- **Uptime Checks**: Set up in Cloud Monitoring to alert if the `/health` endpoint fails

---

## Clean Up

To delete everything and avoid ongoing charges:

```bash
gcloud run services delete haett-churn-api --region us-central1
gcloud container images delete gcr.io/$PROJECT_ID/haett-churn-api --force-delete-tags
gcloud projects delete $PROJECT_ID
```

---

## Architecture

```
┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│  GitHub Repo  │────▶│ GitHub Actions │────▶│  Artifact    │
│  (code push)  │     │  (cd.yml)      │     │  Registry    │
└──────────────┘     └───────────────┘     └──────┬───────┘
                                                   │
                                                   ▼
┌──────────────────────────────────────────────────────────┐
│                 Cloud Run (Serverless)                     │
│  ┌────────────────────────────────────────────────────┐  │
│  │  FastAPI Container                                  │  │
│  │  ├── /health   → Health check                      │  │
│  │  ├── /predict  → Churn prediction                  │  │
│  │  └── /docs     → Swagger UI                        │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Built-in Monitoring                                │  │
│  │  ├── Cloud Logging (logs)                          │  │
│  │  ├── Cloud Monitoring (metrics)                    │  │
│  │  └── Uptime Checks (alerts)                        │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```
