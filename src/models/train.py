"""
Model Training Pipeline
Trains multiple classifiers with hyperparameter tuning, tracks experiments with MLflow,
calibrates probabilities, and saves the best model.
"""

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib
import json
import tempfile
from pathlib import Path
import sys
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import (
    FEATURES_DIR,
    MODELS_DIR,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MODEL_PARAMS,
    HPARAM_SEARCH_SPACES,
    N_HPARAM_ITER,
    CALIBRATE_PROBABILITIES,
    CALIBRATION_METHOD,
    RANDOM_SEED,
)
import shap
from src.utils.metrics import (
    compute_classification_metrics,
    find_optimal_threshold,
    compute_lift_at_top_k,
)
from src.monitoring.drift_detection import save_reference_data
from datetime import datetime

# ─── Hyperparameter Tuning ───────────────────────────────────────────────────


def tune_hyperparameters(
    model,
    param_space: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    n_iter: int = N_HPARAM_ITER,
):
    """Run RandomizedSearchCV with stratified 5-fold cross-validation.

    Uses average_precision (PR-AUC) as the scoring metric since it's
    more appropriate for imbalanced binary classification than accuracy.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_space,
        n_iter=n_iter,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        verbose=0,
        random_state=RANDOM_SEED,
        return_train_score=False,
    )

    search.fit(X_train, y_train)

    print(f"      Best CV PR-AUC: {search.best_score_:.4f}")
    print(f"      Best params: {search.best_params_}")

    return search.best_estimator_


# ─── Model Logging Helper ────────────────────────────────────────────────────


def _log_model_to_mlflow(model, artifact_path: str):
    """Log a model to MLflow, with fallback for skops-untrusted types.

    CalibratedClassifierCV wrapping XGBoost triggers skops security warnings
    in newer MLflow versions. We fall back to joblib artifact logging if the
    standard sklearn/xgboost flavor fails.
    """
    try:
        mlflow.sklearn.log_model(model, artifact_path)
    except Exception:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = str(Path(tmpdir) / f"{artifact_path.replace('/', '_')}.pkl")
            joblib.dump(model, tmp_path)
            mlflow.log_artifact(tmp_path, artifact_path)


# ─── Training & Evaluation ───────────────────────────────────────────────────


def train_and_evaluate(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    model_name: str,
    params: dict,
    tune: bool = True,
):
    """Train a single model with optional hyperparameter tuning and MLflow tracking.

    Uses:
    1. Train set for training + cross-validation (if tuning)
    2. Validation set for threshold selection
    3. Test set for final unbiased evaluation
    """
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=model_name):
        scaler = None
        importances = None

        if model_name == "lr_classifier":
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)

            if tune:
                base = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)
                tuned_model = tune_hyperparameters(
                    base,
                    HPARAM_SEARCH_SPACES[model_name],
                    X_train_scaled,
                    y_train,
                    model_name,
                )
            else:
                tuned_model = LogisticRegression(**params)
                tuned_model.fit(X_train_scaled, y_train)

            if CALIBRATE_PROBABILITIES:
                cal_model = CalibratedClassifierCV(
                    tuned_model, method=CALIBRATION_METHOD, cv=3
                )
                cal_model.fit(
                    np.vstack([X_train_scaled, X_val_scaled]),
                    np.hstack([y_train.values, y_val.values]),
                )
                final_model = cal_model
                val_proba = cal_model.predict_proba(X_val_scaled)[:, 1]
            else:
                final_model = tuned_model
                val_proba = tuned_model.predict_proba(X_val_scaled)[:, 1]

            threshold_info = find_optimal_threshold(y_val.values, val_proba)
            y_proba = final_model.predict_proba(X_test_scaled)[:, 1]
            y_pred = (y_proba >= threshold_info["optimal_threshold"]).astype(int)

            # Log model with sklearn flavor (works for both LR and CalibratedClassifierCV)
            _log_model_to_mlflow(final_model, "model")

            # Extract absolute coefficients for feature importance
            if hasattr(tuned_model, "coef_"):
                importances = np.abs(tuned_model.coef_[0])
            elif hasattr(final_model, "coef_"):
                importances = np.abs(final_model.coef_[0])

        elif model_name == "rf_classifier":
            if tune:
                base = RandomForestClassifier(random_state=RANDOM_SEED)
                tuned_model = tune_hyperparameters(
                    base,
                    HPARAM_SEARCH_SPACES[model_name],
                    X_train,
                    y_train,
                    model_name,
                )
            else:
                tuned_model = RandomForestClassifier(**params)
                tuned_model.fit(X_train, y_train)

            if CALIBRATE_PROBABILITIES:
                cal_model = CalibratedClassifierCV(
                    tuned_model, method=CALIBRATION_METHOD, cv=3
                )
                cal_model.fit(
                    np.vstack([X_train, X_val]),
                    np.hstack([y_train.values, y_val.values]),
                )
                final_model = cal_model
                val_proba = cal_model.predict_proba(X_val)[:, 1]
            else:
                final_model = tuned_model
                val_proba = tuned_model.predict_proba(X_val)[:, 1]

            threshold_info = find_optimal_threshold(y_val.values, val_proba)
            y_proba = final_model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= threshold_info["optimal_threshold"]).astype(int)

            _log_model_to_mlflow(final_model, "model")
            importances = (
                tuned_model.feature_importances_
                if hasattr(tuned_model, "feature_importances_")
                else None
            )

        elif model_name == "xgb_classifier":
            if tune:
                base = XGBClassifier(random_state=RANDOM_SEED)
                tuned_model = tune_hyperparameters(
                    base,
                    HPARAM_SEARCH_SPACES[model_name],
                    X_train,
                    y_train,
                    model_name,
                )
            else:
                tuned_model = XGBClassifier(**params)
                tuned_model.fit(
                    X_train, y_train, eval_set=[(X_val, y_val)], verbose=False
                )

            if CALIBRATE_PROBABILITIES:
                cal_model = CalibratedClassifierCV(
                    tuned_model, method=CALIBRATION_METHOD, cv=3
                )
                cal_model.fit(
                    np.vstack([X_train, X_val]),
                    np.hstack([y_train.values, y_val.values]),
                )
                final_model = cal_model
                val_proba = cal_model.predict_proba(X_val)[:, 1]
            else:
                final_model = tuned_model
                val_proba = tuned_model.predict_proba(X_val)[:, 1]

            threshold_info = find_optimal_threshold(y_val.values, val_proba)
            y_proba = final_model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= threshold_info["optimal_threshold"]).astype(int)

            # Use sklearn flavor for calibrated XGB (CalibratedClassifierCV), xgboost for native
            _log_model_to_mlflow(final_model, "model")
            importances = (
                tuned_model.feature_importances_
                if hasattr(tuned_model, "feature_importances_")
                else None
            )

        else:
            raise ValueError(f"Unknown model: {model_name}")

        # ── Compute metrics ──
        metrics = compute_classification_metrics(y_test, y_pred, y_proba)
        metrics["lift_at_10pct"] = compute_lift_at_top_k(y_test, y_proba, 0.1)
        metrics["lift_at_20pct"] = compute_lift_at_top_k(y_test, y_proba, 0.2)

        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        mlflow.log_metric("optimal_threshold", threshold_info["optimal_threshold"])
        mlflow.log_metric("max_f1_at_optimal_threshold", threshold_info["max_f1"])

        # ── Log feature importance as CSV artifact ──
        if importances is not None:
            importance_df = pd.DataFrame(
                {
                    "feature": X_train.columns.tolist(),
                    "importance": importances,
                }
            ).sort_values("importance", ascending=False)
            imp_path = MODELS_DIR / f"feature_importance_{model_name}.csv"
            importance_df.to_csv(imp_path, index=False)
            mlflow.log_artifact(str(imp_path))

        # ── Log model info ──
        mlflow.log_param("n_features", len(X_train.columns))
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_val_samples", len(X_val))
        mlflow.log_param("n_test_samples", len(X_test))
        mlflow.log_param("class_balance", float(y_train.mean()))
        mlflow.log_param("hparams_tuning", tune)
        mlflow.log_param("calibration", CALIBRATE_PROBABILITIES)

        print(f"\n  [OK] {model_name} trained:")
        for k, v in metrics.items():
            print(f"      {k}: {v}")
        print(f"      optimal_threshold: {threshold_info['optimal_threshold']:.4f}")

        # Return both the final model (for inference) and the tuned model (for SHAP)
        return (
            final_model,
            metrics,
            scaler,
            threshold_info["optimal_threshold"],
            tuned_model,
        )


# ─── Pipeline Orchestrator ───────────────────────────────────────────────────


def train_all_models(tune: bool = True):
    """Train all models, tune hyperparameters, and select the best one."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Log tuning config
    with mlflow.start_run(run_name="pipeline_config"):
        mlflow.log_param("n_hparam_iter", N_HPARAM_ITER)
        mlflow.log_param("calibrate_probabilities", CALIBRATE_PROBABILITIES)
        mlflow.log_param("calibration_method", CALIBRATION_METHOD)
        mlflow.log_param("hparams_tuning_enabled", tune)

    print("[...] Loading features...")
    X, y, user_ids = load_features()
    print(f"   Feature matrix: {X.shape}")
    print(f"   Positive class: {y.mean():.1%}")

    # Train/val/test split (60/20/20)
    print("\n[...] Splitting data (60/20/20 train/val/test)...")
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=y_train_full,
    )
    print(f"   Train: {len(X_train)} samples")
    print(f"   Val:   {len(X_val)} samples")
    print(f"   Test:  {len(X_test)} samples")

    # Train models
    print(
        "\n[...] Training models..."
        + (" (with hyperparameter tuning)" if tune else " (default params)")
    )
    results = {}

    for model_name, params in MODEL_PARAMS.items():
        print(f"\n  -- Training {model_name} --")
        model, metrics, scaler, optimal_threshold, tuned_model = train_and_evaluate(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            model_name,
            params,
            tune=tune,
        )
        results[model_name] = {
            "model": model,
            "metrics": metrics,
            "scaler": scaler,
            "optimal_threshold": optimal_threshold,
            "tuned_model": tuned_model,  # raw model before calibration (for SHAP)
        }

    # Select best model (by F1 score on test set)
    best_model_name = max(results, key=lambda k: results[k]["metrics"]["f1_score"])
    best_result = results[best_model_name]
    print(f"\n[Best] Best model: {best_model_name}")
    print(f"   F1 Score: {best_result['metrics']['f1_score']:.4f}")
    print(f"   ROC-AUC: {best_result['metrics'].get('roc_auc', 'N/A')}")
    print(f"   PR-AUC:  {best_result['metrics'].get('pr_auc', 'N/A')}")

    # Save best model to disk
    model_path = MODELS_DIR / "churn_model.pkl"
    joblib.dump(best_result["model"], model_path)
    print(f"   - Model saved to: {model_path}")

    # Save the raw (pre-calibration) tuned model for SHAP explainability
    # (CalibratedClassifierCV wrapping is not SHAP-compatible)
    tuned_model = best_result.get("tuned_model")
    if tuned_model is not None:
        tuned_path = MODELS_DIR / "tuned_model.pkl"
        joblib.dump(tuned_model, tuned_path)
        print(f"   - Tuned model (for SHAP) saved to: {tuned_path}")

    # Save scaler if LR was best
    best_scaler = best_result.get("scaler")
    if best_scaler is not None:
        scaler_path = MODELS_DIR / "scaler.pkl"
        joblib.dump(best_scaler, scaler_path)
        print(f"   - Scaler saved to: {scaler_path}")
    else:
        old_scaler = MODELS_DIR / "scaler.pkl"
        if old_scaler.exists():
            old_scaler.unlink()

    # Save optimal threshold
    optimal_threshold = best_result.get("optimal_threshold", 0.5)
    with open(MODELS_DIR / "optimal_threshold.txt", "w") as f:
        f.write(str(optimal_threshold))
    print(f"   - Optimal threshold ({optimal_threshold:.4f}) saved")

    # Save feature names
    with open(MODELS_DIR / "feature_names.txt", "w") as f:
        f.write("\n".join(X.columns.tolist()))
    print("   - Feature names saved")

    # Save model metadata
    metadata = {
        "best_model_name": best_model_name,
        "features": X.columns.tolist(),
        "metrics": {k: float(v) for k, v in best_result["metrics"].items()},
        "optimal_threshold": float(optimal_threshold),
        "n_features": int(X.shape[1]),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "class_balance": float(y_train.mean()),
        "hparams_tuning": tune,
        "calibration_enabled": CALIBRATE_PROBABILITIES,
        "calibration_method": CALIBRATION_METHOD if CALIBRATE_PROBABILITIES else None,
    }
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("   - Metadata saved")

    # ── Save Reference Data for Drift Monitoring ──
    print("\n[...] Saving reference data for drift monitoring...")
    try:
        # Use the full feature matrix (without user_id) as reference
        ref_X = X.copy()
        save_reference_data(
            features=ref_X,
            predictions=None,
            metadata={
                "model_name": best_model_name,
                "pipeline_run": datetime.now().isoformat(),
            },
        )
        print("   [OK] Reference data saved for drift monitoring")
    except Exception as ref_err:
        print(f"   [!] Reference data saving skipped: {ref_err}")

    # ── SHAP Explainability for Best Model ──
    print(f"\n[...] Computing SHAP explanations for best model ({best_model_name})...")
    try:
        # Use the already-loaded feature matrix X (full dataset) for SHAP
        # Use the tuned/pre-calibrated model for SHAP (avoids CalibratedClassifierCV compatibility issues)
        shap_model = best_result.get("tuned_model", best_result["model"])

        # Use a small sample of the test set for SHAP explanation
        X_explain = X_test.iloc[: min(50, len(X_test))]

        # Use TreeExplainer with default tree_path_dependent algorithm (fast, no background data)
        explainer = shap.TreeExplainer(shap_model)
        shap_values = explainer.shap_values(X_explain.values)

        # Handle different SHAP output formats across versions
        # SHAP returns 3D array (n_samples, n_features, n_classes) for sklearn models
        # Index [..., 1] = positive class (churned)
        shap_val = np.array(shap_values)
        if shap_val.ndim == 3:
            shap_values_class = shap_val[
                :, :, 1
            ]  # positive class: (n_samples, n_features)
        elif shap_val.ndim == 2:
            shap_values_class = shap_val
        elif isinstance(shap_values, (list, tuple)) and len(shap_values) == 2:
            shap_values_class = np.array(shap_values[1])
        else:
            shap_values_class = shap_val

        if shap_values_class.ndim == 1:
            shap_values_class = shap_values_class.reshape(1, -1)

        # ── Global SHAP feature importance (mean absolute value) ──
        shap_importance = np.abs(shap_values_class).mean(axis=0)
        shap_df = pd.DataFrame(
            {
                "feature": X_explain.columns.tolist(),
                "shap_importance": shap_importance,
            }
        ).sort_values("shap_importance", ascending=False)

        shap_csv_path = MODELS_DIR / f"shap_importance_{best_model_name}.csv"
        shap_df.to_csv(shap_csv_path, index=False)
        mlflow.log_artifact(str(shap_csv_path))
        print(f"   - SHAP feature importance saved to: {shap_csv_path}")

        # ── SHAP beeswarm plot (most informative visualization) ──
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            shap.summary_plot(
                shap_values_class,
                X_explain.values,
                feature_names=X_explain.columns.tolist(),
                max_display=15,
                show=False,
            )
            beeswarm_path = MODELS_DIR / f"shap_beeswarm_{best_model_name}.png"
            plt.tight_layout()
            plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact(str(beeswarm_path))
            print(f"   - SHAP beeswarm plot saved to: {beeswarm_path}")
        except Exception as plot_err:
            print(f"   [!] SHAP plot generation skipped: {plot_err}")

        print("[OK] SHAP explainability complete")
    except Exception as shap_err:
        print(f"[!] SHAP analysis skipped: {shap_err}")

    print("\n[Results] Final Model Comparison:\n")
    print(
        f"{'Model':<20} {'F1':<8} {'ROC-AUC':<10} {'PR-AUC':<10} {'Precision':<10} {'Recall':<10} {'Brier':<8} {'Lift@10':<8}"
    )
    print("-" * 84)
    for model_name, result in results.items():
        m = result["metrics"]
        print(
            f"{model_name:<20} {m['f1_score']:<8.4f} "
            f"{m.get('roc_auc', 0):<10.4f} "
            f"{m.get('pr_auc', 0):<10.4f} "
            f"{m['precision']:<10.4f} "
            f"{m['recall']:<10.4f} "
            f"{m.get('brier_score', 0):<8.4f} "
            f"{m.get('lift_at_10pct', 0):<8.4f}"
        )

    return results, best_model_name


# ─── Entry Points ─────────────────────────────────────────────────────────────


def load_features() -> tuple:
    """Load the feature matrix and target variable."""
    X = pd.read_csv(FEATURES_DIR / "features_encoded.csv")
    y = pd.read_csv(FEATURES_DIR / "target.csv")["churned"]

    user_ids = X.get("user_id", None)
    if user_ids is not None:
        X = X.drop(columns=["user_id"])

    return X, y, user_ids


if __name__ == "__main__":
    results, best_model_name = train_all_models(tune=True)
