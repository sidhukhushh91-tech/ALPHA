"""
Unit tests for the model inference engine and decision logic (src/model.py).
Tests constraints corresponding to FR-007, FR-008, FR-009, and TC-ML-001/002.
"""

import pytest
import numpy as np
from src.model import get_classifier, MachineHealthClassifier, ModelInferenceError
from src.config import CONFIDENCE_THRESHOLD


def test_classifier_is_loaded():
    """Verifies that the pre-trained classifier and scaler are properly deserialized."""
    clf = get_classifier()
    assert clf.is_loaded is True
    assert clf.model is not None
    assert clf.scaler is not None


def test_prediction_output_structure():
    """Verifies standard prediction response format and probability bounds."""
    clf = get_classifier()
    dummy_feat = np.ones(46, dtype=np.float32) * 0.1

    result = clf.predict(dummy_feat)
    assert "predicted_class" in result
    assert result["predicted_class"] in ["normal", "abnormal", "uncertain"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert "probabilities" in result
    assert "normal" in result["probabilities"]
    assert "abnormal" in result["probabilities"]
    assert "result_summary" in result


def test_uncertainty_thresholding_logic():
    """
    Verifies that when confidence is artificially forced below the tau=0.65 threshold,
    the system outputs 'uncertain' rather than forcing a binary label.
    """
    clf = get_classifier()

    # Create a mock wrapper to simulate borderline probabilities (e.g. 0.55 Normal vs 0.45 Abnormal)
    class MockModel:
        classes_ = ["abnormal", "normal"]
        def predict_proba(self, X):
            return np.array([[0.45, 0.55]])

    original_model = clf.model
    clf.model = MockModel()

    try:
        dummy_feat = np.zeros(46, dtype=np.float32)
        res = clf.predict(dummy_feat)
        assert res["confidence"] == 0.55
        assert res["confidence"] < CONFIDENCE_THRESHOLD
        assert res["predicted_class"] == "uncertain"
        assert "below the" in res["result_summary"]
    finally:
        clf.model = original_model


def test_nan_feature_error_handling():
    """Verifies that features containing NaN values raise a clean ModelInferenceError (ERR-007)."""
    clf = get_classifier()
    bad_features = np.ones(46, dtype=np.float32)
    bad_features[5] = np.nan

    with pytest.raises(ModelInferenceError) as exc:
        clf.predict(bad_features)
    assert exc.value.error_code == "ERR-007"
    assert exc.value.status_code == 422
