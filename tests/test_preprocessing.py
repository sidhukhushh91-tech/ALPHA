"""
Unit tests for the audio validation and preprocessing module (src/preprocessing.py).
Tests constraints corresponding to FR-003, FR-004, FR-005, and ERR-001 through ERR-004.
"""

import pytest
import numpy as np
from src.preprocessing import (
    preprocess_audio,
    validate_file_metadata,
    check_audio_silence,
    check_audio_duration,
    AudioValidationError
)
from src.config import TARGET_SAMPLE_RATE


def test_valid_audio_preprocessing(valid_audio_bytes):
    """TC-F-001 / TC-I-001: Verifies successful preprocessing of a valid audio stream."""
    y, sr = preprocess_audio(valid_audio_bytes, filename="test_motor.wav")
    assert sr == TARGET_SAMPLE_RATE
    assert isinstance(y, np.ndarray)
    assert y.ndim == 1
    assert len(y) >= int(TARGET_SAMPLE_RATE * 1.0)
    assert np.max(np.abs(y)) <= 1.0


def test_empty_audio_rejection():
    """TC-N-001: Rejection of zero-byte file payload (ERR-002)."""
    with pytest.raises(AudioValidationError) as exc:
        validate_file_metadata("empty.wav", 0)
    assert exc.value.error_code == "ERR-002"
    assert exc.value.status_code == 400


def test_unsupported_format_rejection():
    """TC-N-001: Rejection of unsupported extension (ERR-001)."""
    with pytest.raises(AudioValidationError) as exc:
        validate_file_metadata("malicious.exe", 1024)
    assert exc.value.error_code == "ERR-001"
    assert exc.value.status_code == 415


def test_oversized_file_rejection():
    """TC-N-002: Rejection of files exceeding 10 MB limit (ERR-004)."""
    with pytest.raises(AudioValidationError) as exc:
        validate_file_metadata("huge.wav", 15 * 1024 * 1024)
    assert exc.value.error_code == "ERR-004"
    assert exc.value.status_code == 413


def test_silent_audio_rejection(silent_audio_bytes):
    """TC-N-002: Rejection of clips containing silence below threshold (ERR-002)."""
    with pytest.raises(AudioValidationError) as exc:
        preprocess_audio(silent_audio_bytes, filename="silence.wav")
    assert exc.value.error_code == "ERR-002"
    assert exc.value.status_code == 422


def test_short_audio_rejection(short_audio_bytes):
    """TC-N-002: Rejection of clips under 1.0 second minimum duration (ERR-004)."""
    with pytest.raises(AudioValidationError) as exc:
        preprocess_audio(short_audio_bytes, filename="short.wav")
    assert exc.value.error_code == "ERR-004"
    assert exc.value.status_code == 422
    assert "below the minimum" in str(exc.value.message)


def test_long_audio_rejection(long_audio_bytes):
    """TC-N-002: Rejection of clips exceeding 10.0 seconds limit (ERR-004)."""
    with pytest.raises(AudioValidationError) as exc:
        preprocess_audio(long_audio_bytes, filename="long.wav")
    assert exc.value.error_code == "ERR-004"
    assert exc.value.status_code == 422
    assert "exceeds the maximum" in str(exc.value.message)


def test_corrupted_audio_rejection(corrupted_audio_bytes):
    """TC-N-003: Rejection of corrupted/malformed audio streams (ERR-003)."""
    with pytest.raises(AudioValidationError) as exc:
        preprocess_audio(corrupted_audio_bytes, filename="corrupt.wav")
    assert exc.value.error_code == "ERR-003"
    assert exc.value.status_code == 422
