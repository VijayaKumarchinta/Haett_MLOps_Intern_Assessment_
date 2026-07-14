"""
Prediction module for the Haett Churn Prediction System.

Responsibilities:
- Load model artifacts once.
- Align and transform inference features.
- Produce single and vectorized batch predictions.
- Calculate optional SHAP explanations.
- Generate business retention recommendations.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.utils.config import MODELS_DIR
from src.utils.metrics import (
    assess_risk_level,
    get_business_recommendation,
)

logger = logging.getLogger(__name__)


class ChurnPredictor:
    """
    Wrapper around the trained churn model.

    The deployed model is used for probabilities. The saved raw tuned model
    is used for explainability when available.
    """

    def __init__(
        self,
        model_path: Path | None = None,
    ) -> None:
        self.model_path = (
            Path(model_path)
            if model_path is not None
            else MODELS_DIR / "churn_model.pkl"
        )

        self.tuned_model_path = MODELS_DIR / "tuned_model.pkl"
        self.scaler_path = MODELS_DIR / "scaler.pkl"
        self.feature_names_path = MODELS_DIR / "feature_names.txt"
        self.metadata_path = MODELS_DIR / "model_metadata.json"
        self.threshold_path = MODELS_DIR / "optimal_threshold.txt"

        self.model: Any | None = None
        self.tuned_model: Any | None = None
        self.scaler: Any | None = None
        self.feature_names: list[str] | None = None
        self.metadata: dict[str, Any] | None = None
        self.optimal_threshold = 0.5

        self._shap_explainer: Any | None = None
        self._explainer_lock = threading.Lock()

        self._load_model()

    def _load_model(self) -> None:
        """Load all available inference artifacts."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {self.model_path}. "
                "Run the training pipeline before starting the API."
            )

        self.model = joblib.load(self.model_path)

        if self.tuned_model_path.exists():
            self.tuned_model = joblib.load(self.tuned_model_path)

        if self.scaler_path.exists():
            self.scaler = joblib.load(self.scaler_path)

        if self.threshold_path.exists():
            threshold_text = self.threshold_path.read_text(encoding="utf-8").strip()

            threshold = float(threshold_text)

            if not 0 < threshold < 1:
                raise ValueError(
                    "The stored optimal threshold must be between 0 and 1."
                )

            self.optimal_threshold = threshold

        if self.feature_names_path.exists():
            feature_names = [
                line.strip()
                for line in self.feature_names_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]

            self.feature_names = feature_names or None

        if self.metadata_path.exists():
            with self.metadata_path.open(
                "r",
                encoding="utf-8",
            ) as metadata_file:
                self.metadata = json.load(metadata_file)

        logger.info(
            "Loaded churn model from %s.",
            self.model_path,
        )

    def _model_requires_external_scaling(self) -> bool:
        """
        Determine whether the separately saved scaler must be applied.

        Tree-based models must not be scaled merely because a stale scaler
        artifact exists. Scaling is enabled only when the deployed/raw model
        is identified as Logistic Regression.
        """
        if self.scaler is None:
            return False

        # Avoid scaling twice when a complete sklearn Pipeline was saved.
        if hasattr(self.model, "named_steps"):
            return False

        candidate_models = [
            self.tuned_model,
            self.model,
            getattr(self.model, "estimator", None),
            getattr(self.model, "base_estimator", None),
        ]

        for candidate in candidate_models:
            if candidate is None:
                continue

            model_type = type(candidate).__name__.lower()

            if "logisticregression" in model_type:
                return True

        if self.metadata:
            model_name = str(
                self.metadata.get(
                    "best_model",
                    self.metadata.get(
                        "model_name",
                        self.metadata.get("model_type", ""),
                    ),
                )
            ).lower()

            if "logistic" in model_name or model_name == "lr_classifier":
                return True

        return False

    def align_features(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Align input columns with the exact training feature order.

        Missing model features are initialized to zero. Extra API fields are
        removed when stored feature names are available.
        """
        if features.empty:
            raise ValueError("At least one feature row is required.")

        aligned = features.copy()

        if self.feature_names:
            aligned = aligned.reindex(
                columns=self.feature_names,
                fill_value=0,
            )

        try:
            aligned = aligned.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("All prediction features must be numeric.") from exc

        return aligned

    def transform_features(
        self,
        aligned_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply the saved scaler only when required by the model."""
        if self.scaler is None or not self._model_requires_external_scaling():
            return aligned_features

        transformed_values = self.scaler.transform(aligned_features)

        return pd.DataFrame(
            transformed_values,
            columns=aligned_features.columns,
            index=aligned_features.index,
        )

    def prepare_features(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Align and transform inference features."""
        aligned = self.align_features(features)
        return self.transform_features(aligned)

    def predict(
        self,
        features: pd.DataFrame,
        explain: bool = False,
    ) -> dict[str, Any]:
        """
        Predict churn probability for one user.

        SHAP is computed only when:
        - explain=True, or
        - the user is classified as High Risk.

        This avoids unnecessary inference latency for routine Low- and
        Medium-Risk requests while preserving evidence-based actions for
        High-Risk users.
        """
        if len(features) != 1:
            raise ValueError("predict() accepts exactly one feature row.")

        if self.model is None:
            self._load_model()

        aligned_features = self.align_features(features)
        model_features = self.transform_features(aligned_features)

        probability = float(self.model.predict_proba(model_features)[0, 1])

        risk_level = assess_risk_level(
            probability,
            self.optimal_threshold,
        )

        should_explain = explain or risk_level == "High"

        shap_explanations = None

        if should_explain:
            shap_explanations = self._compute_explanations(
                model_features=model_features,
                display_features=aligned_features,
            )

        recommendation = get_business_recommendation(
            probability=probability,
            risk_level=risk_level,
            optimal_threshold=self.optimal_threshold,
            shap_explanations=shap_explanations,
        )

        result: dict[str, Any] = {
            "churn_probability": round(probability, 4),
            "risk_level": risk_level,
            "business_recommendation": recommendation,
        }

        if explain and shap_explanations:
            result["explanations"] = shap_explanations

        return result

    def predict_batch(
        self,
        features_batch: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """
        Predict churn for multiple users using vectorized probability scoring.

        Batch requests deliberately skip SHAP computation to avoid running an
        expensive explanation process hundreds of times.
        """
        if features_batch.empty:
            raise ValueError("At least one feature row is required.")

        if self.model is None:
            self._load_model()

        aligned_features = self.align_features(features_batch)
        model_features = self.transform_features(aligned_features)

        probabilities = np.asarray(
            self.model.predict_proba(model_features)[:, 1],
            dtype=float,
        )

        results: list[dict[str, Any]] = []

        for probability in probabilities:
            probability_value = float(probability)

            risk_level = assess_risk_level(
                probability_value,
                self.optimal_threshold,
            )

            recommendation = get_business_recommendation(
                probability=probability_value,
                risk_level=risk_level,
                optimal_threshold=self.optimal_threshold,
                shap_explanations=None,
            )

            results.append(
                {
                    "churn_probability": round(
                        probability_value,
                        4,
                    ),
                    "risk_level": risk_level,
                    "business_recommendation": recommendation,
                }
            )

        return results

    def _init_explainer(self) -> None:
        """Lazily and safely initialize the appropriate SHAP explainer."""
        if self._shap_explainer is not None:
            return

        with self._explainer_lock:
            if self._shap_explainer is not None:
                return

            import shap

            shap_model = (
                self.tuned_model if self.tuned_model is not None else self.model
            )

            if shap_model is None:
                raise RuntimeError("Cannot initialize SHAP without a model.")

            model_type = type(shap_model).__name__.lower()
            feature_count = (
                len(self.feature_names)
                if self.feature_names
                else int(
                    getattr(
                        shap_model,
                        "n_features_in_",
                        1,
                    )
                )
            )

            background = np.zeros(
                (1, feature_count),
                dtype=float,
            )

            try:
                if any(
                    model_name in model_type
                    for model_name in (
                        "randomforest",
                        "xgb",
                        "gradientboosting",
                        "decisiontree",
                    )
                ):
                    self._shap_explainer = shap.TreeExplainer(shap_model)

                elif "logisticregression" in model_type:
                    self._shap_explainer = shap.LinearExplainer(
                        shap_model,
                        background,
                    )

                elif hasattr(shap_model, "predict_proba"):
                    self._shap_explainer = shap.Explainer(
                        shap_model.predict_proba,
                        background,
                    )

                else:
                    self._shap_explainer = shap.Explainer(
                        shap_model,
                        background,
                    )

            except Exception:
                logger.warning(
                    "SHAP explainer initialization failed.",
                    exc_info=True,
                )
                raise

    def _extract_shap_row(
        self,
        shap_output: Any,
    ) -> np.ndarray:
        """
        Normalize different SHAP return formats to one feature vector.

        Supported formats include:
        - shap.Explanation
        - ndarray
        - list of class-specific ndarrays
        - binary-class three-dimensional outputs
        """
        values = getattr(
            shap_output,
            "values",
            shap_output,
        )

        if isinstance(values, (list, tuple)):
            if not values:
                raise ValueError("SHAP returned an empty value list.")

            # For binary classification, class index 1 represents churn.
            values = values[1] if len(values) > 1 else values[0]

        values_array = np.asarray(values)

        if values_array.ndim == 3:
            # Common format: samples × features × classes.
            if values_array.shape[-1] > 1:
                values_array = values_array[:, :, 1]
            else:
                values_array = values_array[:, :, 0]

        if values_array.ndim == 2:
            return np.asarray(
                values_array[0],
                dtype=float,
            )

        if values_array.ndim == 1:
            return np.asarray(
                values_array,
                dtype=float,
            )

        raise ValueError("Unsupported SHAP output shape: " f"{values_array.shape}")

    def _compute_explanations(
        self,
        model_features: pd.DataFrame,
        display_features: pd.DataFrame,
        top_n: int = 5,
    ) -> list[dict[str, Any]] | None:
        """
        Compute the largest absolute SHAP contributions.

        model_features contains the transformed values consumed by the model.
        display_features contains the original aligned values returned to the
        API so explanations remain understandable.
        """
        try:
            self._init_explainer()
        except Exception:
            return None

        try:
            try:
                shap_output = self._shap_explainer(model_features)
            except (TypeError, AttributeError):
                # Compatibility fallback for older SHAP explainer APIs.
                shap_output = self._shap_explainer.shap_values(model_features)

            shap_row = self._extract_shap_row(shap_output)

            if len(shap_row) != len(model_features.columns):
                raise ValueError(
                    "SHAP feature count does not match "
                    "the model input feature count."
                )

            contributions: list[dict[str, Any]] = []

            for index, column in enumerate(model_features.columns):
                contributions.append(
                    {
                        "feature": column,
                        "value": float(display_features.iloc[0, index]),
                        "impact": round(
                            float(shap_row[index]),
                            4,
                        ),
                    }
                )

            contributions.sort(
                key=lambda item: abs(item["impact"]),
                reverse=True,
            )

            return contributions[:top_n]

        except Exception:
            logger.warning(
                "SHAP computation failed.",
                exc_info=True,
            )
            return None


_predictor: ChurnPredictor | None = None
_predictor_lock = threading.Lock()


def get_predictor() -> ChurnPredictor:
    """Get or create the process-wide predictor instance."""
    global _predictor

    if _predictor is not None:
        return _predictor

    with _predictor_lock:
        if _predictor is None:
            _predictor = ChurnPredictor()

    return _predictor


def reset_predictor() -> None:
    """
    Reset the singleton.

    Intended for automated tests that replace model artifacts or monkeypatch
    predictor behavior.
    """
    global _predictor

    with _predictor_lock:
        _predictor = None
