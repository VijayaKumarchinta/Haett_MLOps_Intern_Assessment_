"""
Configuration module for Haett MLOps Churn Prediction System.
Centralizes all paths, constants, and hyperparameters.
"""

import os
from datetime import datetime
from pathlib import Path

import pandas as pd

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data paths
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"

# MLflow
# Using SQLite backend (file store deprecated in MLflow 3.x)
_mlflow_db = PROJECT_ROOT / "mlruns" / "mlflow.db"
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{_mlflow_db.as_posix()}",
)
MLFLOW_EXPERIMENT_NAME = "haett_churn_prediction"

# API
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Data generation
RANDOM_SEED = 42
N_USERS = 5000
N_DAYS_HISTORY = 365
CHURN_LABEL_DAYS = 30
CHURN_RATE = 0.25  # Target for data generation tuning; actual label rate depends on simulation

# Snapshot date for temporal consistency
# All features are computed from data up to this date.
# Churn labels predict what happens in the 30 days AFTER this date.
SNAPSHOT_DATE = pd.Timestamp(datetime(2025, 6, 1))

# Feature engineering
FEATURE_GROUPS = [
    "recency",
    "frequency",
    "monetary",
    "subscription",
    "engagement",
    "demographic",
]

# Deterministic age group encoding
AGE_GROUP_MAP = {
    "young_adult": 0,
    "adult": 1,
    "middle_age": 2,
    "senior": 3,
}

# Model hyperparameters (defaults - used if no tuning or as fallback)
MODEL_PARAMS = {
    "xgb_classifier": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "scale_pos_weight": 3,
        "random_state": RANDOM_SEED,
        "eval_metric": "aucpr",
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

# Hyperparameter search spaces for RandomizedSearchCV
# Each defines a distribution (via scipy or plain lists) to sample from
HPARAM_SEARCH_SPACES = {
    "lr_classifier": {
        "C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
        "penalty": ["l2"],
        "solver": ["lbfgs", "liblinear"],
        "class_weight": [None, "balanced"],
        "max_iter": [1000],
    },
    "rf_classifier": {
        "n_estimators": [100, 200, 400, 600, 800],
        "max_depth": [None, 6, 10, 15, 20],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": ["sqrt", "log2", None],
        "class_weight": [None, "balanced"],
    },
    "xgb_classifier": {
        "n_estimators": [100, 200, 400, 600, 800],
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.6, 0.7, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0, 0.1, 0.3],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [1.0, 2.0, 5.0],
        "scale_pos_weight": [1, 2, 3, 5],
    },
}

# Number of random search iterations
N_HPARAM_ITER = 30

# Whether to calibrate model probabilities
CALIBRATE_PROBABILITIES = True
CALIBRATION_METHOD = "sigmoid"  # "sigmoid" or "isotonic"

# Evaluation thresholds
CHURN_PROB_THRESHOLDS = {
    "low": 0.3,
    "medium": 0.6,
    "high": 1.0,
}
