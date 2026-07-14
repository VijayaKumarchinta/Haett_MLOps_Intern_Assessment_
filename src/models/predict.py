"""
Prediction Module
Handles model loading, single/batch predictions, and SHAP explainability for the FastAPI endpoint.
"""

import logging
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import MODELS_DIR
from src.utils.metrics import assess_risk_level, get_business_recommendation

logger = logging.getLogger(__name__)


class ChurnPredictor:
    """Wrapper around the trained churn prediction model with SHAP explainability."""

    def __init__(self, model_path: Path | None = None):
        self.model_path = model_path or MODELS_DIR / "churn_model.pkl"
        self.tuned_model_path = MODELS_DIR / "tuned_model.pkl"
        self.scaler_path = MODELS_DIR / "scaler.pkl"
        self.feature_names_path = MODELS_DIR / "feature_names.txt"
        self.metadata_path = MODELS_DIR / "model_metadata.json"
        self.threshold_path = MODELS_DIR / "optimal_threshold.txt"
        self.model = None
        self.tuned_model = None  # raw model for SHAP (pre-calibration)
        self.scaler = None
        self.feature_names = None
        self.metadata = None
        self.optimal_threshold = 0.5
        self._shap_explainer = None
        self._load_model()

    def _load_model(self):
        """Load the trained model, tuned model (for SHAP), scaler, and metadata."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. Run train.py first."
            )
        self.model = joblib.load(self.model_path)

        # Load the raw tuned model for SHAP (saved pre-calibration)
        if self.tuned_model_path.exists():
            self.tuned_model = joblib.load(self.tuned_model_path)

        # Load scaler if it exists (used for Logistic Regression)
        if self.scaler_path.exists():
            self.scaler = joblib.load(self.scaler_path)

        # Load optimal threshold
        if self.threshold_path.exists():
            with open(self.threshold_path, "r") as f:
                self.optimal_threshold = float(f.read().strip())

        if self.feature_names_path.exists():
            with open(self.feature_names_path, "r") as f:
                self.feature_names = [line.strip() for line in f.readlines()]

        if self.metadata_path.exists():
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)

    def _init_explainer(self):
        """Lazily initialize the SHAP explainer using the raw tuned model."""
        if self._shap_explainer is not None:
            return

        import shap

        shap_model = self.tuned_model if self.tuned_model is not None else self.model
        model_type = type(shap_model).__name__

        try:
            if "RandomForest" in model_type or "XGB" in model_type or "GradientBoosting" in model_type:
                self._shap_explainer = shap.TreeExplainer(shap_model)
            elif "LogisticRegression" in model_type:
                n_features = len(self.feature_names) if self.feature_names else 10
                self._shap_explainer = shap.LinearExplainer(
                    shap_model, np.zeros((1, n_features))
                )
            else:
                self._shap_explainer = shap.Explainer(
                    shap_model, check_additivity=False
                )
        except Exception:
            logger.warning("SHAP explainer initialization failed", exc_info=True)
            raise

    def prepare_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Align, order, and scale features to match the model's expectations."""
        if self.feature_names:
            for col in self.feature_names:
                if col not in features.columns:
                    features[col] = 0
            features = features[self.feature_names]

        if self.scaler is not None:
            features = pd.DataFrame(
                self.scaler.transform(features),
                columns=features.columns,
                index=features.index,
            )

        return features

    def predict(self, features: pd.DataFrame, explain: bool = False) -> dict:
        """
        Predict churn probability for a single user's feature vector.

        Args:
            features: DataFrame with the same columns as training data.
            explain: If True, include SHAP feature explanations in the result.

        Returns:
            dict with churn_probability, risk_level, business_recommendation,
            and optionally explanations (list of top feature contributions).
        """
        if self.model is None:
            self._load_model()

        features = self.prepare_features(features)

        # Predict probability
        probability = float(self.model.predict_proba(features)[0, 1])

        # Assess risk level
        risk_level = assess_risk_level(probability)

        # Generate business recommendation with actual feature context
        feature_context = features.iloc[0].to_dict()
        recommendation = get_business_recommendation(probability, feature_context)

        result = {
            "churn_probability": round(probability, 4),
            "risk_level": risk_level,
            "business_recommendation": recommendation,
        }

        # Compute SHAP explanations if requested
        if explain:
            explanations = self._compute_explanations(features)
            result["explanations"] = explanations

        return result

    def _compute_explanations(self, features: pd.DataFrame, top_n: int = 5) -> list[dict]:
        """Compute SHAP feature explanations for a single prediction.

        Returns the top_n features by absolute SHAP value, with their
        contribution to the prediction (positive = increases churn risk).
        """
        try:
            self._init_explainer()
        except Exception:
            return []

        try:
            shap_values = self._shap_explainer.shap_values(features.values)
            shap_val = np.array(shap_values)

            # Extract positive class (churned) from various SHAP formats
            if shap_val.ndim == 3:
                # (n_samples, n_features, n_classes) → use class 1
                shap_val = shap_val[:, :, 1]
            elif isinstance(shap_values, (list, tuple)) and len(shap_values) == 2:
                shap_val = np.array(shap_values[1])
            elif shap_val.ndim == 1:
                shap_val = shap_val.reshape(1, -1)

            contributions = []
            for i, col in enumerate(features.columns):
                impact = float(shap_val[0, i]) if shap_val.shape[0] > 0 else 0.0
                contributions.append({
                    "feature": col,
                    "value": float(features.iloc[0, i]),
                    "impact": round(impact, 4),
                })

            contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)
            return contributions[:top_n]

        except Exception:
            logger.warning("SHAP computation failed", exc_info=True)
            return []

    def predict_batch(self, features_batch: pd.DataFrame) -> list[dict]:
        """Predict churn for multiple users."""
        results = []
        for idx in range(len(features_batch)):
            row = features_batch.iloc[[idx]]
            result = self.predict(row)
            results.append(result)
        return results


# Singleton instance for use across the API
_predictor: ChurnPredictor | None = None


def get_predictor() -> ChurnPredictor:
    """Get or create the global predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = ChurnPredictor()
    return _predictor
