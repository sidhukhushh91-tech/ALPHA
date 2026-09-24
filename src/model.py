"""
Machine learning inference module.
Manages model artifact loading, feature scaling, probabilistic inference,
and confidence thresholding for the 3-state classification rule:
(Normal, Abnormal, Uncertain).
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import joblib

from src.config import MODEL_PATH, SCALER_PATH, CONFIDENCE_THRESHOLD


class ModelInferenceError(Exception):
    """Custom exception raised during model loading or inference failures."""
    def __init__(self, message: str, error_code: str = "ERR-007", status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code


class MachineHealthClassifier:
    """Encapsulates the pre-trained classifier and feature scaler."""

    def __init__(self, model_path: Path = MODEL_PATH, scaler_path: Path = SCALER_PATH):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.is_loaded = False
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads serialized model and scaler from disk."""
        if not self.model_path.exists() or not self.scaler_path.exists():
            # Will be caught or handled during startup/training
            self.is_loaded = False
            return

        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.is_loaded = True
        except Exception as e:
            raise ModelInferenceError(
                f"Failed to deserialize model artifacts: {str(e)}",
                error_code="ERR-006",
                status_code=503
            )

    def predict(self, feature_vector: np.ndarray) -> Dict[str, Any]:
        """
        Executes model inference on a 46-dimensional feature vector.
        Applies confidence thresholding: if max(P) < tau, marks as 'uncertain'.
        """
        if not self.is_loaded or self.model is None or self.scaler is None:
            # Re-attempt loading in case it was just trained
            self._load_artifacts()
            if not self.is_loaded:
                raise ModelInferenceError(
                    "Model artifact not found. Please train the model before running inference.",
                    error_code="ERR-006",
                    status_code=503
                )

        if np.isnan(feature_vector).any() or np.isinf(feature_vector).any():
            raise ModelInferenceError(
                "Extracted features contain NaN or infinite values.",
                error_code="ERR-007",
                status_code=422
            )

        try:
            X = feature_vector.reshape(1, -1)
            X_scaled = self.scaler.transform(X)

            # Retrieve probability distribution
            probabilities = self.model.predict_proba(X_scaled)[0]
            classes = list(self.model.classes_)  # e.g., ['abnormal', 'normal']

            prob_map = {cls: float(prob) for cls, prob in zip(classes, probabilities)}
            p_normal = prob_map.get("normal", 0.0)
            p_abnormal = prob_map.get("abnormal", 0.0)

            # Determine dominant predicted class and confidence
            dominant_class = "normal" if p_normal >= p_abnormal else "abnormal"
            confidence = max(p_normal, p_abnormal)

            # Apply uncertainty rule
            if confidence < CONFIDENCE_THRESHOLD:
                final_class = "uncertain"
                summary = (
                    f"Acoustic confidence ({confidence * 100:.1f}%) is below the {CONFIDENCE_THRESHOLD * 100:.0f}% "
                    f"threshold. Sound pattern is ambiguous; inspection recommended."
                )
            elif final_class_raw := dominant_class:
                final_class = final_class_raw
                if final_class == "normal":
                    summary = f"Normal acoustic profile confirmed with {confidence * 100:.1f}% confidence."
                else:
                    summary = f"Abnormal acoustic anomalies detected with {confidence * 100:.1f}% confidence. Immediate inspection advised."

            return {
                "predicted_class": final_class,
                "confidence": round(float(confidence), 4),
                "probabilities": {
                    "normal": round(float(p_normal), 4),
                    "abnormal": round(float(p_abnormal), 4)
                },
                "raw_dominant_class": dominant_class,
                "result_summary": summary
            }

        except ModelInferenceError:
            raise
        except Exception as e:
            raise ModelInferenceError(
                f"Inference computation error: {str(e)}",
                error_code="ERR-007",
                status_code=500
            )


# Global singleton instance for re-use across FastAPI requests
_classifier_instance: Optional[MachineHealthClassifier] = None


def get_classifier() -> MachineHealthClassifier:
    """Returns or initializes the global singleton classifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = MachineHealthClassifier()
    return _classifier_instance
