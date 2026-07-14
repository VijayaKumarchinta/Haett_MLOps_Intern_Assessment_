"""
Prediction Module
Handles model loading and single/batch predictions for the FastAPI endpoint.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import MODELS_DIR
from src.utils.metrics import assess_risk_level, get_business_recommendation


class ChurnPredictor:
    """Wrapper around the trained churn prediction model."""

    def __init__(self, model_path: Path | None = None):
        self.model_path = model_path or MODELS_DIR / "churn_model.pkl"
        self.scaler_path = MODELS_DIR / "scaler.pkl"
        self.feature_names_path = MODELS_DIR / "feature_names.txt"
        self.metadata_path = MODELS_DIR / "model_metadata.json"
        self.threshold_path = MODELS_DIR / "optimal_threshold.txt"
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.metadata = None
        self.optimal_threshold = 0.5
        self._load_model()

    def _load_model(self):
        """Load the trained model, scaler, and metadata."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. Run train.py first."
            )
        self.model = joblib.load(self.model_path)

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

    def predict(self, features: pd.DataFrame) -> dict:
        """
        Predict churn probability for a single user's feature vector.

        Args:
            features: DataFrame with the same columns as training data.

        Returns:
            dict with churn_probability, risk_level, and business_recommendation.
        """
        if self.model is None:
            self._load_model()

        # Ensure features are in the right order
        if self.feature_names:
            missing_cols = set(self.feature_names) - set(features.columns)
            if missing_cols:
                # Add missing columns with default value 0
                for col in missing_cols:
                    features[col] = 0

            features = features[self.feature_names]

        # Apply scaler if available (for Logistic Regression models)
        if self.scaler is not None:
            features = pd.DataFrame(
                self.scaler.transform(features),
                columns=features.columns,
                index=features.index,
            )

        # Predict probability
        probability = float(self.model.predict_proba(features)[0, 1])

        # Assess risk level
        risk_level = assess_risk_level(probability)

        # Generate business recommendation
        recommendation = get_business_recommendation(probability, features=None)

        return {
            "churn_probability": round(probability, 4),
            "risk_level": risk_level,
            "business_recommendation": recommendation,
        }

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
