# 🍱 Haett MLOps Churn Prediction System

> **End-to-end machine learning system to predict user churn for the Haett healthy meal delivery platform.**
>
> Built for the **Haett MLOps Intern Assessment**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)]()
[![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)]()
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)]()
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)]()

---

## 📋 Table of Contents

- [🌟 Overview](#-overview)
- [🏗️ Architecture](#-architecture)
- [🎯 Assessment Criteria](#-assessment-criteria)
- [🚀 Quick Start](#-quick-start)
- [📊 Pipeline Steps](#-pipeline-steps)
- [📡 API Reference](#-api-reference)
- [📸 Screenshots](#-screenshots)
- [🧪 Testing](#-testing)
- [🐳 Docker](#-docker)
- [📈 MLOps Practices](#-mlops-practices)
- [📋 Submission Checklist](#-submission-checklist)
- [🔮 Future Improvements](#-future-improvements)

---

## 🌟 Overview

This project predicts which users are likely to **churn (cancel their subscription) within the next 30 days** on the Haett meal delivery platform. The system takes user historical activity as input and returns:

1. **Churn probability** — how likely the user is to churn
2. **Risk level** — Low, Medium, or High
3. **Business recommendation** — actionable retention suggestion for High Risk users

### Key Deliverables

| Deliverable | Status |
|---|---|
| Synthetic dataset (5,000 users) | ✅ Complete |
| 30 features matching assessment criteria | ✅ Complete |
| Multi-model training (LR, RF, XGBoost) | ✅ Complete |
| MLflow experiment tracking | ✅ Complete |
| FastAPI with `POST /predict` endpoint | ✅ Complete |
| Business recommendation engine | ✅ Complete |
| Docker containerization | ✅ Complete |
| CI/CD GitHub Actions | ✅ Bonus |
| SHAP explainability | ✅ Bonus |
| Data drift detection | ✅ Bonus |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                         │
│  ┌──────────┐    ┌────────────┐    ┌──────────────────┐  │
│  │ Generate │ →  │ Preprocess │ →  │ Feature Engineer │  │
│  │  Data    │    │   Data     │    │     ing          │  │
│  └──────────┘    └────────────┘    └────────┬─────────┘  │
│                                              │           │
│                                              ▼           │
│                                      ┌──────────────────┐ │
│                                      │   Train Models    │ │
│                                      │  (MLflow Tracked) │ │
│                                      └────────┬─────────┘ │
│                                               │           │
│                                               ▼           │
│                                      ┌──────────────────┐ │
│                                      │   Best Model     │ │
│                                      │   (joblib)       │ │
│                                      └────────┬─────────┘ │
└──────────────────────────────────────┬────────┘──────────┘
                                       │
                                       ▼
┌──────────────────────────────────────┴──────────────────┐
│                    API SERVICE (FastAPI)                │
│  ┌──────────────────────────────────────────────────┐   │
│  │              POST /predict                       │   │
│  │  User Features → Churn Probability               │   │
│  │                → Risk Level (Low/Medium/High)    │   │
│  │                → Business Recommendation         │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## 🎯 Assessment Criteria

The feature set was designed to match the **7 criteria features** specified in the assessment:

| # | Criteria Feature | Our Feature | Source Module |
|---|---|---|---|
| 1 | Days since last order | `days_since_last_order` | Recency |
| 2 | Orders in the last 30 days | `orders_last_30_days` | Frequency |
| 3 | Average order value | `avg_order_value` | Monetary |
| 4 | Subscription duration | `subscription_tenure_days` | Subscription |
| 5 | **Coupon usage** | `coupon_usage_count`, `coupon_usage_rate` | ⭐ Added |
| 6 | Meal swap frequency | `avg_meals_skipped` | Engagement |
| 7 | Order consistency | `std_days_between_orders` | Frequency |

**Total features: 30** (including one-hot encoded demographics)

### Target Leakage Prevention

Features that could leak future information were **removed**:
- ❌ `is_sub_active` — if subscription already cancelled, user can't churn again
- ❌ `days_since_cancellation` — directly reveals if user already cancelled
- ❌ `total_subscription_days` — 0 means never subscribed, can't churn
- ❌ `cancellation_reason_*` — non-empty reason implies already cancelled

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip

### 1. Clone & Setup

```bash
cd F:\Projects\HaettMLOps
python -m venv venv
.\venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### 2. Run the Full Pipeline (Data → Features → Training)

```bash
python src/run_pipeline.py
```

**What happens:**
1. Generates 5,000 synthetic users with orders, subscriptions, and engagement data
2. Cleans and preprocesses all raw data
3. Engineers **30 features** matching the 7 assessment criteria
4. Trains **3 models** (Logistic Regression, Random Forest, XGBoost) with hyperparameter tuning
5. Tracks all experiments in **MLflow**
6. Selects and saves the **best model**

**Sample output:**
```
[Best] Best model: xgb_classifier
   F1 Score: 0.2835
   ROC-AUC: 0.7587
   PR-AUC:  0.1891
   Features: 30
```

### 3. Start the Prediction API

```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 4. Open in Browser

| Link | What you'll see |
|---|---|
| **http://localhost:8000/docs** | Interactive Swagger API documentation |
| **http://localhost:8000/health** | Health check JSON |
| **http://localhost:5000** | MLflow experiment tracking dashboard |

### 5. Make a Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "days_since_last_order": 60,
    "tenure_days": 75,
    "total_orders": 5,
    "std_days_between_orders": 15,
    "orders_last_30_days": 0,
    "avg_order_value": 28,
    "avg_rating": 2.1,
    "coupon_usage_count": 2,
    "coupon_usage_rate": 0.4,
    "n_plan_changes": 3,
    "monthly_price": 29.99,
    "subscription_tenure_days": 30,
    "avg_app_logins": 0.5,
    "avg_meals_skipped": 3,
    "total_support_tickets": 8,
    "age": 22,
    "age_group_code": 0
  }'
```

**Response:**
```json
{
  "user_id": 1,
  "churn_probability": 0.2269,
  "risk_level": "Low",
  "business_recommendation": "No action needed. User is at low risk of churning."
}
```

### One-Command Runner

```bash
python run.py --all    # Pipeline + API in one command
python run.py --fast   # Quick pipeline (no tuning)
python run.py          # Full pipeline only
```

---

## 📊 Pipeline Steps

| Step | Script | Description | Output |
|---|---|---|---|
| 1. Data Generation | `src/data/generate_data.py` | Creates 5,000 synthetic users with orders, subscriptions, engagement (includes coupon_usage) | `data/raw/*.csv` |
| 2. Preprocessing | `src/data/preprocess.py` | Cleans, validates, and standardizes raw data | `data/processed/*.csv` |
| 3. Feature Engineering | `src/data/feature_engineering.py` | Builds **30 features** matching the 7 assessment criteria | `data/features/*.csv` |
| 4. Model Training | `src/models/train.py` | Trains LR, RF, XGBoost with MLflow tracking + hyperparameter tuning | `models/churn_model.pkl` |

### Feature Groups (30 total)

| Group | Features |
|---|---|
| **Recency** | `days_since_last_order`, `tenure_days` |
| **Frequency** | `total_orders`, `std_days_between_orders` (order consistency), `orders_last_30_days` |
| **Monetary** | `avg_order_value`, `avg_rating`, `coupon_usage_count`, `coupon_usage_rate` |
| **Subscription** | `n_plan_changes`, `monthly_price`, `subscription_tenure_days` |
| **Engagement** | `avg_app_logins`, `avg_meals_skipped` (meal swap frequency), `total_support_tickets` |
| **Demographic** | `age`, `age_group_code`, diet one-hot (6), referral one-hot (7) |

---

## 📡 API Reference

### Health Check

```http
GET /health
```

```json
{"status": "healthy", "model_loaded": true, "version": "1.0.0"}
```

### Predict Churn (Single User)

```http
POST /predict
```

**Request Body (18 fields):**

```json
{
  "user_id": 1,
  "days_since_last_order": 60,
  "tenure_days": 75,
  "total_orders": 5,
  "std_days_between_orders": 15,
  "orders_last_30_days": 0,
  "avg_order_value": 28,
  "avg_rating": 2.1,
  "coupon_usage_count": 2,
  "coupon_usage_rate": 0.4,
  "n_plan_changes": 3,
  "monthly_price": 29.99,
  "subscription_tenure_days": 30,
  "avg_app_logins": 0.5,
  "avg_meals_skipped": 3,
  "total_support_tickets": 8,
  "age": 22,
  "age_group_code": 0
}
```

**Response:**
```json
{
  "user_id": 1,
  "churn_probability": 0.2269,
  "risk_level": "Low",
  "business_recommendation": "No action needed. User is at low risk of churning."
}
```

### With SHAP Explanations

```http
POST /predict?explain=true
```

**Response includes `explanations` array:**
```json
{
  "user_id": 1,
  "churn_probability": 0.061,
  "risk_level": "Low",
  "explanations": [
    {"feature": "tenure_days",         "value": 200.0, "impact": 0.1473},
    {"feature": "subscription_tenure_days", "value": 180.0, "impact": -0.1142},
    {"feature": "avg_meals_skipped",    "value": 1.5,   "impact": -0.0425},
    {"feature": "avg_rating",           "value": 3.2,   "impact": -0.0281},
    {"feature": "total_support_tickets","value": 4.0,   "impact": 0.0261}
  ]
}
```

### Predict Churn (Batch)

```http
POST /predict/batch
```

Accepts up to **1,000 users** in a single request.

---

## 📸 Screenshots

The following screenshots are available in the `screenshots/` directory for submission:

| Screenshot | File | Description |
|---|---|---|
| **Swagger API Documentation** | `screenshots/Haett Churn Prediction API - Swagger UI.pdf` | All 4 API endpoints listed |
| **Prediction Response** | (In Swagger UI) | `/predict` with user data + response |
| **MLflow Experiment Runs** | `screenshots/Runs - Experiment 1 - MLflow.pdf` | Training runs with metrics |
| **MLflow Model Comparison** | `screenshots/Compare Runs - MLflow.pdf` | Side-by-side model comparison |

To take fresh screenshots:
1. Open **http://localhost:8000/docs** → right-click → Save as PDF
2. Open **http://localhost:5000** → click experiment → right-click → Save as PDF
3. In Swagger UI → POST /predict → Try it out → Execute → screenshot the response

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_feature_engineering.py -v
python -m pytest tests/test_api.py -v
```

**Results: 44 passed, 0 failed, 1 skipped**

---

## 🐳 Docker

### Build & Run

```bash
# Build the image (pipeline runs inside during build)
docker build -t haett-churn-api .

# Run the API
docker run -p 8000:8000 haett-churn-api

# Or use docker-compose (includes MLflow)
docker-compose up --build
```

### Docker Compose Services

| Service | Port | Description |
|---|---|---|
| `api` | 8000 | FastAPI prediction service |
| `mlflow` | 5000 | MLflow tracking server |

### Dockerfile Features
- **Multi-stage build** — builder stage trains model, runtime stage is minimal
- **Non-root user** — runs as `appuser` for security
- **HEALTHCHECK** — automatically checks `/health` every 30s
- **Auto-training** — pipeline runs during build, no manual training needed

---

## 📈 MLOps Practices

| Practice | Implementation |
|---|---|
| ✅ **Experiment Tracking** | MLflow logs params, metrics, artifacts per run |
| ✅ **Model Versioning** | Each training run creates a new MLflow run |
| ✅ **Reproducibility** | Fixed random seed (42), requirements.txt, Docker |
| ✅ **Modular Code** | Separate modules: data, models, api, utils |
| ✅ **API Documentation** | Auto-generated Swagger UI at /docs |
| ✅ **Input Validation** | Pydantic schemas with field constraints |
| ✅ **Containerization** | Multi-stage Docker + docker-compose |
| ✅ **Target Leakage Prevention** | Removed is_sub_active and related features |
| ✅ **CI/CD (Bonus)** | GitHub Actions workflows for lint, test, deploy |
| ✅ **SHAP Explainability (Bonus)** | Feature contributions per prediction |
| ✅ **Data Drift Detection (Bonus)** | Evidently AI reference data monitoring |
| ✅ **Cloud Deployment Guide (Bonus)** | DEPLOYMENT.md for GCP Cloud Run |

---

## 📋 Submission Checklist

- [x] Complete source code in `src/`
- [x] README with setup instructions (this file)
- [x] Requirements file (`requirements.txt`)
- [x] Docker configuration (`Dockerfile`, `docker-compose.yml`)
- [x] MLflow experiment tracking (screenshots in `screenshots/`)
- [x] Sample API requests (see [Quick Start](#-quick-start))
- [x] Design assumptions documented (synthetic data generation)
- [x] Feature engineering matches assessment criteria (7 features)
- [x] Target leakage verified and fixed
- [x] Pushed to GitHub: `https://github.com/VijayaKumarchinta/Haett_MLOps_Intern_Assessment_`

---

## 🔮 Future Improvements

- [ ] Deploy to cloud (GCP Cloud Run, AWS ECS)
- [ ] Real data ingestion (replace synthetic generator)
- [ ] Real-time streaming predictions with Kafka
- [ ] A/B testing framework for retention campaigns
- [ ] Grafana monitoring dashboard
- [ ] Automated retraining pipeline
- [ ] Feature store (Feast)

---

## 📝 License

MIT © Vijaya Kumar Chinta

---

<div align="center">
  <sub>Built for the <strong>Haett MLOps Intern Assessment</strong></sub>
</div>
