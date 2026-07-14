"""
Model Training Pipeline
Trains multiple classifiers, tracks experiments with MLflow, and saves the best model.
"""

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import (
    FEATURES_DIR,
    MODELS_DIR,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MODEL_PARAMS,
    RANDOM_SEED,
)
from src.utils.metrics import (
    compute_classification_metrics,
    find_optimal_threshold,
)


def load_features() -> tuple:
    """Load the feature matrix and target variable."""
    X = pd.read_csv(FEATURES_DIR / "features_encoded.csv")
    y = pd.read_csv(FEATURES_DIR / "target.csv")["churned"]

    # Separate user_id from features
    user_ids = X.get("user_id", None)
    if user_ids is not None:
        X = X.drop(columns=["user_id"])

    return X, y, user_ids


def train_and_evaluate(X_train, y_train, X_test, y_test, model_name: str, params: dict):
    """Train a single model and log with MLflow. Returns model, metrics, scaler, and threshold info."""
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        mlflow.set_tag("model_type", model_name)

        scaler = None

        # Scale features for logistic regression
        if model_name == "lr_classifier":
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model = LogisticRegression(**params)
            model.fit(X_train_scaled, y_train)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            y_pred = model.predict(X_test_scaled)
            mlflow.sklearn.log_model(model, "model")
        elif model_name == "rf_classifier":
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
            mlflow.sklearn.log_model(model, "model")
        elif model_name == "xgb_classifier":
            model = XGBClassifier(**params)
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:, 1]
            # Use optimal threshold after finding it
            threshold_info = find_optimal_threshold(y_test.values, y_proba)
            y_pred = (y_proba >= threshold_info["optimal_threshold"]).astype(int)
            mlflow.xgboost.log_model(model, "model")
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Compute metrics (for XGBoost, threshold_info is already computed above)
        if model_name != "xgb_classifier":
            threshold_info = find_optimal_threshold(y_test.values, y_proba)

        metrics = compute_classification_metrics(y_test, y_pred, y_proba)
        mlflow.log_metrics(metrics)

        # Log optimal threshold
        mlflow.log_metric("optimal_threshold", threshold_info["optimal_threshold"])
        mlflow.log_metric("max_f1_at_optimal_threshold", threshold_info["max_f1"])

        # Log feature importance
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            for i, col in enumerate(X_train.columns):
                mlflow.log_metric(f"feature_importance_{col}", float(importances[i]))
        elif model_name == "lr_classifier":
            importances = np.abs(model.coef_[0])
            for i, col in enumerate(X_train.columns):
                mlflow.log_metric(f"feature_importance_{col}", float(importances[i]))

        # Log model info
        feature_names = list(X_train.columns)
        mlflow.log_param("n_features", len(feature_names))
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_test_samples", len(X_test))
        mlflow.log_param("class_balance", float(y_train.mean()))

        print(f"\n  ✓ {model_name} trained:")
        for k, v in metrics.items():
            print(f"      {k}: {v}")
        print(f"      optimal_threshold: {threshold_info['optimal_threshold']:.4f}")

        return model, metrics, scaler, threshold_info["optimal_threshold"]


def train_all_models():
    """Train all models and track experiments."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    print("[...] Loading features...")
    X, y, user_ids = load_features()
    print(f"   Feature matrix: {X.shape}")
    print(f"   Positive class: {y.mean():.1%}")

    # Train/test split (stratified)
    print("\n[...] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")

    # Train models
    print("\n[...] Training models...")
    results = {}

    for model_name, params in MODEL_PARAMS.items():
        print(f"\n  -- Training {model_name} --")
        model, metrics, scaler, optimal_threshold = train_and_evaluate(
            X_train, y_train, X_test, y_test, model_name, params
        )
        results[model_name] = {
            "model": model,
            "metrics": metrics,
            "scaler": scaler,
            "optimal_threshold": optimal_threshold,
        }

    # Select best model (by F1 score)
    best_model_name = max(results, key=lambda k: results[k]["metrics"]["f1_score"])
    best_result = results[best_model_name]
    print(f"\n[Best] Best model: {best_model_name}")
    print(f"   F1 Score: {best_result['metrics']['f1_score']:.4f}")
    print(f"   ROC-AUC: {best_result['metrics'].get('roc_auc', 'N/A')}")

    # Save best model
    best_model = best_result["model"]
    model_path = MODELS_DIR / "churn_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"   - Model saved to: {model_path}")

    # Save scaler if the best model used one (Logistic Regression)
    best_scaler = best_result.get("scaler")
    if best_scaler is not None:
        scaler_path = MODELS_DIR / "scaler.pkl"
        joblib.dump(best_scaler, scaler_path)
        print(f"   - Scaler saved to: {scaler_path}")
    else:
        # Remove old scaler if it exists (from a previous LR best model)
        old_scaler = MODELS_DIR / "scaler.pkl"
        if old_scaler.exists():
            old_scaler.unlink()

    # Save optimal threshold for the best model
    optimal_threshold = best_result.get("optimal_threshold", 0.5)
    threshold_path = MODELS_DIR / "optimal_threshold.txt"
    with open(threshold_path, "w") as f:
        f.write(str(optimal_threshold))
    print(f"   - Optimal threshold ({optimal_threshold:.4f}) saved to: {threshold_path}")

    # Save feature names
    feature_names_path = MODELS_DIR / "feature_names.txt"
    with open(feature_names_path, "w") as f:
        f.write("\n".join(X.columns.tolist()))
    print(f"   - Feature names saved to: {feature_names_path}")

    # Save model metadata
    metadata = {
        "best_model_name": best_model_name,
        "features": X.columns.tolist(),
        "metrics": best_result["metrics"],
        "optimal_threshold": optimal_threshold,
        "n_features": X.shape[1],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "class_balance": float(y_train.mean()),
    }
    import json
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"   - Metadata saved to: {MODELS_DIR / 'model_metadata.json'}")

    # Log final results
    print("\n📊 Final Results Summary:")
    print(f"{'Model':<20} {'F1':<8} {'ROC-AUC':<10} {'Precision':<10} {'Recall':<10}")
    print('-' * 58)
    for model_name, result in results.items():
        m = result["metrics"]
        print(
            f"{model_name:<20} {m['f1_score']:<8.4f} "
            f"{m.get('roc_auc', 0):<10.4f} "
            f"{m['precision']:<10.4f} "
            f"{m['recall']:<10.4f}"
        )

    return results, best_model_name


if __name__ == "__main__":
    results, best_model_name = train_all_models()
