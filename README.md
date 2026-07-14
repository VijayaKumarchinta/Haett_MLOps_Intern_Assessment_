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
  - [User Scenarios](#-user-scenarios)
  - [SHAP Explainability](#-with-shap-explanations)
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
2. **Risk level** — **Low**, **Medium**, or **High** (using the model's optimal decision threshold)
3. **Business recommendation** — SHAP-driven retention suggestion with risk signals, strengths, and actions

### Key Deliverables

| Deliverable | Status |
|---|---|
| Synthetic dataset (5,000 users) | ✅ Complete |
| 30 features matching assessment criteria | ✅ Complete |
| Multi-model training (LR, RF, XGBoost) | ✅ Complete — XGBoost best (F1=0.28, ROC-AUC=0.76) |
| MLflow experiment tracking | ✅ Complete |
| FastAPI with `POST /predict` endpoint | ✅ Complete |
| **SHAP-driven business recommendations** | ✅ Complete |
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
│  │                → SHAP-driven Recommendation      │   │
│  │                → [Optional] Feature Explanations │   │
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
   Optimal threshold: 0.1620
```

### 3. Start the Prediction API

```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 4. Open in Browser

| Link | What you'll see |
|---|---|
| [**http://localhost:8000/docs**](http://localhost:8000/docs) | Interactive Swagger API documentation |
| [**http://localhost:8000/health**](http://localhost:8000/health) | Health check JSON |
| [**http://localhost:5000**](http://localhost:5000) | MLflow experiment tracking dashboard |

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
  "days_since_last_order": 1,
  "tenure_days": 600,
  "total_orders": 80,
  "std_days_between_orders": 2.5,
  "orders_last_30_days": 10,
  "avg_order_value": 45,
  "avg_rating": 4.8,
  "coupon_usage_count": 0,
  "coupon_usage_rate": 0,
  "n_plan_changes": 0,
  "monthly_price": 99.99,
  "subscription_tenure_days": 580,
  "avg_app_logins": 15,
  "avg_meals_skipped": 0,
  "total_support_tickets": 0,
  "age": 45,
  "age_group_code": 2
}
```

---

### 👤 User Scenarios

#### ✅ Scenario 1: Low Risk — Power User (Loyal Customer)

A loyal, long-term customer with high engagement, no support issues, and excellent ratings.

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "days_since_last_order": 1,
    "tenure_days": 600,
    "total_orders": 80,
    "std_days_between_orders": 2.5,
    "orders_last_30_days": 10,
    "avg_order_value": 45,
    "avg_rating": 4.8,
    "coupon_usage_count": 0,
    "coupon_usage_rate": 0,
    "n_plan_changes": 0,
    "monthly_price": 99.99,
    "subscription_tenure_days": 580,
    "avg_app_logins": 15,
    "avg_meals_skipped": 0,
    "total_support_tickets": 0,
    "age": 45,
    "age_group_code": 2
  }' | python -m json.tool
```

**Response:**
```json
{
    "user_id": 1,
    "churn_probability": 0.0526,
    "risk_level": "Low",
    "business_recommendation": "Low risk (P=5.26%). User is in good standing — no retention action needed. Strength: Long-term subscriber — recognize milestone with a reward."
}
```

---

#### ⚠️ Scenario 2: Medium Risk — Disengaged User (Short History, Low Engagement)

A relatively new user with short account history, declining engagement, poor ratings, and multiple support tickets.

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 2,
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
  }' | python -m json.tool
```

**Response** *(note: the recommendation includes SHAP-identified risk signals and targeted actions)*:
```json
{
    "user_id": 2,
    "churn_probability": 0.2269,
    "risk_level": "Medium",
    "business_recommendation": "At-risk (P=22.69%). Signals: Only 30 days subscribed — early churn risk window (impact: +0.608); Short account history of only 75 days — low loyalty (impact: +0.407).\n\nRecommended actions:\n  1. Strengthen onboarding with guided meal selection and tips\n  2. Welcome series: offer a discounted 4-week trial to build ordering habit"
}
```

---

#### ⚠️ Scenario 3: Medium Risk — Brand New User (Critical Onboarding Window)

A brand new user who signed up only 5 days ago but is already showing warning signs: multiple support tickets, skipped meals, and low app engagement. This captures churn risk during the critical early onboarding phase.

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 3,
    "days_since_last_order": 10,
    "tenure_days": 15,
    "total_orders": 2,
    "std_days_between_orders": 999,
    "orders_last_30_days": 1,
    "avg_order_value": 22,
    "avg_rating": 2.5,
    "coupon_usage_count": 1,
    "coupon_usage_rate": 0.5,
    "n_plan_changes": 1,
    "monthly_price": 19.99,
    "subscription_tenure_days": 5,
    "avg_app_logins": 0.2,
    "avg_meals_skipped": 5,
    "total_support_tickets": 4,
    "age": 20,
    "age_group_code": 0
  }' | python -m json.tool
```

**Response:**
```json
{
    "user_id": 3,
    "churn_probability": 0.2097,
    "risk_level": "Medium",
    "business_recommendation": "At-risk (P=20.97%). Signals: Only 5 days subscribed — early churn risk window (impact: +0.583); Short account history of only 15 days — low loyalty (impact: +0.352).\n\nRecommended actions:\n  1. Strengthen onboarding with guided meal selection and tips\n  2. Welcome series: offer a discounted 4-week trial to build ordering habit"
}
```

---

### 📊 With SHAP Explanations

```http
POST /predict?explain=true
```

Add `?explain=true` to receive the **top 5 feature contributions** via SHAP for every prediction. Each explanation shows:

- **feature**: The feature name
- **value**: The actual feature value for this user
- **impact**: Positive = increases churn risk, Negative = decreases churn risk

```bash
curl -s -X POST 'http://localhost:8000/predict?explain=true' \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 2,
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
    "user_id": 2,
    "churn_probability": 0.2269,
    "risk_level": "Medium",
    "business_recommendation": "At-risk (P=22.69%). Signals: Only 30 days subscribed — early churn risk window (impact: +0.608); Short account history of only 75 days — low loyalty (impact: +0.407).\n\nRecommended actions:\n  1. Strengthen onboarding with guided meal selection and tips\n  2. Welcome series: offer a discounted 4-week trial to build ordering habit",
    "explanations": [
        {"feature": "subscription_tenure_days", "value": 30.0, "impact": 0.6084},
        {"feature": "tenure_days", "value": 75.0, "impact": 0.4074},
        {"feature": "total_support_tickets", "value": 8.0, "impact": 0.1142},
        {"feature": "age", "value": 22.0, "impact": 0.0621},
        {"feature": "avg_meals_skipped", "value": 3.0, "impact": -0.0447}
    ]
}
```

> **💡 Key Insight**: `subscription_tenure_days` (+0.6084) is the strongest risk driver for this user — the model identifies that being only 30 days into the subscription is the #1 churn signal. The recommendation targets this with onboarding improvement.

### Predict Churn (Batch)

```http
POST /predict/batch
```

Accepts up to **1,000 users** in a single request.

---

## 📸 Screenshots

Click the links below to view the screenshots captured from the running system:

| Screenshot | Preview | Description |
|---|---|---|
| **Swagger API Documentation** | [📄 Click to View](./screenshots/Haett%20Churn%20Prediction%20API%20-%20Swagger%20UI.pdf) | All 4 API endpoints (GET /health, GET /, POST /predict, POST /predict/batch) |
| **MLflow Experiment Runs** | [📄 Click to View](./screenshots/Runs%20-%20Experiment%201%20-%20MLflow.pdf) | All training runs with their metrics (F1, ROC-AUC, PR-AUC) |
| **MLflow Model Comparison** | [📄 Click to View](./screenshots/Compare%20Runs%20-%20MLflow.pdf) | Side-by-side comparison of LR, RF, and XGBoost models |

### How to Take Fresh Screenshots

1. **Swagger UI**: Open [http://localhost:8000/docs](http://localhost:8000/docs) → Right-click → **Print** → **Save as PDF**
2. **MLflow Runs**: Open [http://localhost:5000](http://localhost:5000) → Click experiment → Right-click → **Save as PDF**
3. **Prediction Response**: In Swagger UI → `POST /predict` → **Try it out** → Paste user data → **Execute** → Screenshot the response

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_feature_engineering.py -v
python -m pytest tests/test_api.py -v
```

**Results: 45 passed, 0 failed, 0 skipped**

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
| ✅ **Dynamic Risk Thresholds** | Uses model's optimal F1 threshold from training |
| ✅ **SHAP-Driven Recommendations** | Business recommendations use SHAP feature importance |
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
- [x] Sample API requests (see [User Scenarios](#-user-scenarios))
- [x] Design assumptions documented (synthetic data generation)
- [x] Feature engineering matches assessment criteria (7 features)
- [x] Target leakage verified and fixed
- [x] SHAP-driven business recommendations with risk signals
- [x] Dynamic risk thresholds using model's optimal threshold
- [x] Pushed to GitHub: [https://github.com/VijayaKumarchinta/Haett_MLOps_Intern_Assessment_](https://github.com/VijayaKumarchinta/Haett_MLOps_Intern_Assessment_)

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
