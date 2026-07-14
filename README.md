# 🍱 Haett MLOps Churn Prediction System

> **End-to-end machine learning system to predict user churn for the Haett healthy meal delivery platform.**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com/)

---

## 📋 Table of Contents

- [🌟 Overview](#-overview)
- [🏗️ Architecture](#️-architecture)
- [🚀 Quick Start](#-quick-start)
- [📊 Pipeline Steps](#-pipeline-steps)
- [📡 API Reference](#-api-reference)
- [🧪 Testing](#-testing)
- [🐳 Docker](#-docker)
- [📈 MLOps Practices](#-mlops-practices)
- [🔮 Future Improvements](#-future-improvements)

---

## 🌟 Overview

This project builds a production-ready churn prediction system for **Haett**, a healthy meal delivery platform. The goal is to identify users at risk of churning within the next 30 days so the business can take proactive retention measures.

**Key deliverables:**
- ✅ Synthetic dataset generation (5,000 users with orders, subscriptions, and engagement history)
- ✅ Feature engineering pipeline (RFM, behavioral, subscription, and demographic features)
- ✅ Multi-model training with MLflow experiment tracking (XGBoost, Random Forest, Logistic Regression)
- ✅ REST API with FastAPI serving churn predictions + business recommendations
- ✅ Docker containerization for reproducible deployment
- ✅ Business recommendation engine based on risk level

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

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip

### 1. Clone & Setup

```bash
cd F:\Projects\HaettMLOps
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
python src/run_pipeline.py
```

This will:
1. Generate 5,000 synthetic user profiles with realistic behavior patterns
2. Clean and preprocess the data
3. Engineer 40+ predictive features
4. Train XGBoost, Random Forest, and Logistic Regression models
5. Track all experiments with MLflow
6. Save the best model for inference

### 3. Start the API

```bash
uvicorn src.api.main:app --reload
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

### 4. View MLflow Dashboard

```bash
mlflow ui --backend-store-uri mlruns
```

Open **http://localhost:5000** to compare experiments.

---

## 📊 Pipeline Steps

| Step | Script | Description | Output |
|------|--------|-------------|--------|
| 1. Data Generation | `src/data/generate_data.py` | Creates 5,000 synthetic users with orders, subscriptions, engagement metrics | `data/raw/*.csv` |
| 2. Preprocessing | `src/data/preprocess.py` | Cleans, validates, and standardizes raw data | `data/processed/*.csv` |
| 3. Feature Engineering | `src/data/feature_engineering.py` | Builds RFM + behavioral + subscription + demographic features | `data/features/*.csv` |
| 4. Model Training | `src/models/train.py` | Trains XGBoost, RF, LR with MLflow tracking | `models/churn_model.pkl`, MLflow artifacts |

### Feature Groups

| Group | Examples |
|-------|----------|
| **Recency** | Days since last order, days since last high-value order, days since late delivery |
| **Frequency** | Total orders, unique meal plans, order consistency, weekend ratio |
| **Monetary** | Total spent, avg/max/min order value, spending trend, rating average |
| **Subscription** | Plan changes, cancellation reason, tenure, active status, pricing |
| **Engagement** | App logins, recipes viewed, meals skipped, support tickets, decline signals |
| **Demographic** | Age group, dietary preference, referral source |

---

## 📡 API Reference

### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### Predict Churn (Single User)

```http
POST /predict
```

**Request Body:**
```json
{
  "user_id": 42,
  "days_since_last_order": 14,
  "total_orders": 24,
  "total_spent": 720.50,
  "avg_order_value": 30.02,
  "avg_rating": 4.5,
  "total_support_tickets": 0,
  "age": 28,
  "is_sub_active": true,
  "subscription_tenure_days": 200,
  "tenure_days": 210
}
```

**Response:**
```json
{
  "user_id": 42,
  "churn_probability": 0.1234,
  "risk_level": "Low",
  "business_recommendation": "No action needed. User is at low risk of churning."
}
```

### Predict Churn (Batch)

```http
POST /predict/batch
```

Accepts up to 1,000 users in a single request.

---

## 🧪 Testing

```bash
pytest tests/ -v
```

---

## 🐳 Docker

### Build & Run

```bash
# Build the image
docker build -t haett-churn-api .

# Run with Docker
docker run -p 8000:8000 -v $(pwd)/models:/app/models haett-churn-api

# Or use docker-compose (includes MLflow)
docker-compose up --build
```

### Services (docker-compose)

| Service | Port | Description |
|---------|------|-------------|
| `api` | 8000 | FastAPI prediction service |
| `mlflow` | 5000 | MLflow tracking server |

---

## 📈 MLOps Practices

| Practice | Implementation |
|----------|---------------|
| ✅ **Experiment Tracking** | MLflow logs params, metrics, artifacts per run |
| ✅ **Model Versioning** | Each training run creates a new MLflow run |
| ✅ **Reproducibility** | Fixed random seed, requirements.txt, Docker |
| ✅ **Modular Code** | Separate modules for data, models, api, utils |
| ✅ **API Documentation** | Auto-generated Swagger UI at /docs |
| ✅ **Data Versioning Ready** | Data stored in timestamped directories (future) |
| ✅ **Containerization** | Docker & docker-compose for easy deployment |

### Bonus Features (Future)

- 🔄 **CI/CD**: GitHub Actions for automated testing + deployment
- 📊 **Data Drift Detection**: Monitor feature distributions over time
- 🔍 **Model Explainability**: SHAP analysis for prediction explanations
- ☁️ **Cloud Deployment**: Deploy to Cloudflare Workers, AWS, or GCP
- 🏪 **Feature Store**: Centralized feature serving for training and inference

## 🔮 Future Improvements

- [ ] Add **SHAP explainability** to API responses
- [ ] Implement **data drift monitoring** (Evidently AI or similar)
- [ ] Add **CI/CD pipeline** with GitHub Actions
- [ ] Deploy to **cloud platform** (Cloudflare Workers, AWS ECS, GCP Cloud Run)
- [ ] Add **feature store** (Feast or Tecton)
- [ ] Implement **A/B testing framework** for retention campaigns
- [ ] Add **real-time streaming** predictions with Kafka
- [ ] Integrate **retraining pipeline** on new data
- [ ] Add **model monitoring dashboard** (Grafana + Prometheus)

## 📝 License

MIT © Vijaya Kumar Chinta

---

<div align="center">
  <sub>Built for the <strong>Haett MLOps Intern Assessment</strong></sub>
</div>
