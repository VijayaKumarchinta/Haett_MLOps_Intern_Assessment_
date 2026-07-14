"""
Configuration module for Haett MLOps Churn Prediction System.
Centralizes all paths, constants, and hyperparameters.
"""

import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data paths
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"

# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", str(PROJECT_ROOT / "mlruns"))
MLFLOW_EXPERIMENT_NAME = "haett_churn_prediction"

# API
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Data generation
RANDOM_SEED = 42
N_USERS = 2000
N_DAYS_HISTORY = 365
CHURN_LABEL_DAYS = 30
CHURN_RATE = 0.25  # ~25% churn rate

# Feature engineering
FEATURE_GROUPS = [
    "recency",
    "frequency",
    "monetary",
    "subscription",
    "engagement",
    "demographic",
]

# Model hyperparameters
MODEL_PARAMS = {
    "xgb_classifier": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "scale_pos_weight": 3,  # handle class imbalance
        "random_state": RANDOM_SEED,
        "eval_metric": "auc",
        "use_label_encoder": False,
    },
    "rf_classifier": {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "random_state": RANDOM_SEED,
        "class_weight": "balanced",
    },
    "lr_classifier": {
        "C": 1.0,
        "max_iter": 1000,
        "random_state": RANDOM_SEED,
        "class_weight": "balanced",
    },
}

# Evaluation thresholds
CHURN_PROB_THRESHOLDS = {
    "low": 0.3,
    "medium": 0.6,
    "high": 1.0,
}
