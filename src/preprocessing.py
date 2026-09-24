"""
Audio validation and preprocessing pipeline.
Handles file format verification, universal decoding (WAV, MP3, WebM, OGG),
silence detection, resampling to 16 kHz mono, and amplitude normalization.
"""

import io
import os
import shutil
from pathlib import Path
from typing import Tuple, Union
import numpy as np
import soundfile as sf
import librosa

# Auto-detect winget-installed FFmpeg if not in active process PATH before importing pydub
if not shutil.which("ffmpeg"):
    winget_packages = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
    for candidate in winget_packages.glob("**/ffmpeg.exe"):
        ffmpeg_bin_dir = str(candidate.parent)
        os.environ["PATH"] = ffmpeg_bin_dir + os.pathsep + os.environ.get("PATH", "")
        break

from pydub import AudioSegment

from src.config import (
    TARGET_SAMPLE_RATE,
    MIN_DURATION_SEC,
    MAX_DURATION_SEC,
    MAX_FILE_SIZE_BYTES,
    SILENCE_DBFS_THRESHOLD,
    TOP_DB_TRIM,
    SUPPORTED_EXTENSIONS,
)


class AudioValidationError(Exception):
    """Custom exception raised during audio validation failures with an error code."""
    def __init__(self, message: str, error_code: str = "ERR-001", status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code


def validate_file_metadata(filename: str, file_size: int):
    """
    Validates file extension and size before reading audio data into memory.
    """
    if file_size == 0:
        raise AudioValidationError(
            "Uploaded audio file is empty (0 bytes).",
            error_code="ERR-002",
            status_code=400
        )

    if file_size > MAX_FILE_SIZE_BYTES:
        raise AudioValidationError(
            f"File size ({file_size / (1024*1024):.2f} MB) exceeds maximum allowed limit of 10 MB.",
            error_code="ERR-004",
            status_code=413
        )

    ext = Path(filename).suffix.lower()
    if ext and ext not in SUPPORTED_EXTENSIONS:
        raise AudioValidationError(
            f"Unsupported file format '{ext}'. Accepted formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}.",
            error_code="ERR-001",
            status_code=415
        )


def load_and_decode_audio(file_bytes: bytes, filename: str = "") -> Tuple[np.ndarray, int]:
    """
    Decodes audio byte stream into a floating-point NumPy array and sample rate.
    Uses soundfile for WAV/FLAC/OGG and falls back to pydub/FFmpeg for WebM/MP3.
    """
    # 1. Try decoding with soundfile (fastest for WAV/FLAC/OGG)
    try:
        data, sr = sf.read(io.BytesIO(file_bytes), dtype="float32", always_2d=False)
        return data, sr
    except Exception:
        pass

    # 2. Try decoding with pydub (handles WebM from MediaRecorder, MP3, etc. via FFmpeg)
    try:
        format_hint = Path(filename).suffix.lower().replace(".", "") if filename else None
        audio_segment = AudioSegment.from_file(
            io.BytesIO(file_bytes),
            format=format_hint if format_hint in ["webm", "ogg", "mp3", "wav"] else None
        )
        # Convert to mono, 16-bit PCM numpy array
        audio_segment = audio_segment.set_channels(1)
        sr = audio_segment.frame_rate
        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        # Normalize 16-bit integer scale to [-1.0, 1.0]
        max_val = float(1 << (8 * audio_segment.sample_width - 1))
        samples = samples / max_val
        return samples, sr
    except Exception as e:
        raise AudioValidationError(
            f"Unable to read audio file. Stream is corrupted or unreadable: {str(e)}",
            error_code="ERR-003",
            status_code=422
        )


def check_audio_silence(y: np.ndarray):
    """
    Computes RMS energy and ensures audio is not completely silent or muted.
    """
    rms = np.sqrt(np.mean(y**2)) if len(y) > 0 else 0.0
    dbfs = 20.0 * np.log10(rms + 1e-9)

    if dbfs < SILENCE_DBFS_THRESHOLD or rms < 1e-4:
        raise AudioValidationError(
            f"Audio appears silent or empty (signal level: {dbfs:.1f} dBFS). Please record or upload a clear machine sound.",
            error_code="ERR-002",
            status_code=422
        )


def check_audio_duration(duration_sec: float):
    """
    Enforces minimum and maximum allowable duration constraints.
    """
    if duration_sec < MIN_DURATION_SEC:
        raise AudioValidationError(
            f"Audio duration ({duration_sec:.2f}s) is below the minimum required duration of {MIN_DURATION_SEC:.1f} second.",
            error_code="ERR-004",
            status_code=422
        )

    if duration_sec > MAX_DURATION_SEC:
        raise AudioValidationError(
            f"Audio duration ({duration_sec:.2f}s) exceeds the maximum allowed duration of {MAX_DURATION_SEC:.1f} seconds. Please trim the clip.",
            error_code="ERR-004",
            status_code=422
        )


def preprocess_audio(file_bytes: bytes, filename: str = "") -> Tuple[np.ndarray, int]:
    """
    Full end-to-end preprocessing pipeline:
    1. Metadata validation (size, extension)
    2. Universal decoding
    3. Mono downmixing
    4. Duration validation
    5. Silence detection
    6. Silence trimming (edges)
    7. Resampling to target rate (16 kHz)
    8. Peak normalization
    """
    validate_file_metadata(filename, len(file_bytes))
    y, sr = load_and_decode_audio(file_bytes, filename)

    # Convert multi-channel to mono
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    initial_duration = len(y) / float(sr)
    check_audio_duration(initial_duration)
    check_audio_silence(y)

    # Resample to target rate (16 kHz) if needed
    if sr != TARGET_SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
        sr = TARGET_SAMPLE_RATE

    # Trim leading and trailing silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=TOP_DB_TRIM)
    if len(y_trimmed) > int(MIN_DURATION_SEC * sr * 0.5):
        y = y_trimmed

    # Re-check duration on trimmed signal
    trimmed_duration = len(y) / float(sr)
    check_audio_duration(trimmed_duration)

    # Peak amplitude normalization to -1.0 dBFS (approx 0.90 amplitude)
    max_val = np.max(np.abs(y))
    if max_val > 1e-6:
        y = y / max_val * 0.90

    return y, sr
