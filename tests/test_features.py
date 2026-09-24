"""
Unit tests for the acoustic feature extraction module (src/features.py).
Tests constraints corresponding to FR-006 and TC-I-002.
"""

import numpy as np
from src.features import extract_features, compute_mel_spectrogram, extract_waveform_summary
from src.config import TARGET_SAMPLE_RATE


def test_feature_vector_dimension():
    """Verifies that extracted acoustic vectors have the required 46 dimensions."""
    # Synthesize 2.0s of sound
    t = np.linspace(0, 2.0, int(TARGET_SAMPLE_RATE * 2.0), endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    features = extract_features(y, TARGET_SAMPLE_RATE)
    assert isinstance(features, np.ndarray)
    assert features.shape == (46,)
    assert not np.isnan(features).any()
    assert not np.isinf(features).any()


def test_feature_extraction_deterministic():
    """Ensures deterministic behavior given identical acoustic waveforms."""
    rng = np.random.default_rng(1234)
    y = rng.normal(0, 0.2, int(TARGET_SAMPLE_RATE * 1.5)).astype(np.float32)

    feat1 = extract_features(y, TARGET_SAMPLE_RATE)
    feat2 = extract_features(y, TARGET_SAMPLE_RATE)
    np.testing.assert_allclose(feat1, feat2, rtol=1e-5)


def test_mel_spectrogram_computation():
    """Validates 2D Mel-spectrogram tensor generation."""
    t = np.linspace(0, 2.0, int(TARGET_SAMPLE_RATE * 2.0), endpoint=False)
    y = (0.3 * np.sin(2 * np.pi * 500 * t)).astype(np.float32)

    s_mel = compute_mel_spectrogram(y, TARGET_SAMPLE_RATE)
    assert s_mel.ndim == 2
    assert s_mel.shape[0] == 128  # N_MELS
    assert s_mel.shape[1] > 0


def test_waveform_summary():
    """Validates downsampled waveform array for UI."""
    y = np.sin(np.linspace(0, 100, 16000)).astype(np.float32)
    summary = extract_waveform_summary(y, num_points=100)
    assert len(summary) == 100
    assert all(isinstance(v, float) for v in summary)
