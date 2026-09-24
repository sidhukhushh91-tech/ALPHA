"""
Pytest fixtures and mock audio data generators for testing.
"""

import io
import pytest
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from src.app import app
from src.config import TARGET_SAMPLE_RATE


@pytest.fixture
def client():
    """Provides a TestClient instance for testing FastAPI endpoints."""
    with TestClient(app) as test_client:
        yield test_client


def make_wav_bytes(duration: float = 3.0, sr: int = TARGET_SAMPLE_RATE, freq: float = 440.0, amplitude: float = 0.5) -> bytes:
    """Generates an in-memory WAV byte stream."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = amplitude * np.sin(2 * np.pi * freq * t)
    buf = io.BytesIO()
    sf.write(buf, y.astype(np.float32), sr, format="WAV")
    return buf.getvalue()


@pytest.fixture
def valid_audio_bytes():
    """Valid 3.0-second machine audio clip."""
    return make_wav_bytes(duration=3.0, freq=300.0, amplitude=0.6)


@pytest.fixture
def short_audio_bytes():
    """Audio clip below minimum 1.0-second limit (0.5s)."""
    return make_wav_bytes(duration=0.5, freq=440.0)


@pytest.fixture
def long_audio_bytes():
    """Audio clip exceeding maximum 10.0-second limit (12.0s)."""
    return make_wav_bytes(duration=12.0, freq=440.0)


@pytest.fixture
def silent_audio_bytes():
    """Audio clip containing purely near-zero noise / silence."""
    t = np.linspace(0, 3.0, int(TARGET_SAMPLE_RATE * 3.0), endpoint=False)
    y = np.zeros_like(t, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, y, TARGET_SAMPLE_RATE, format="WAV")
    return buf.getvalue()


@pytest.fixture
def corrupted_audio_bytes():
    """Malformed pseudo-audio byte stream."""
    return b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00CORRUPTED_PAYLOAD_NOT_READABLE"
