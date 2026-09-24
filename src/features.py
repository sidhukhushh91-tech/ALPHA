"""
Acoustic feature extraction pipeline.
Computes deterministic, fixed-dimensional statistical vectors from preprocessed audio:
- 20 Mel-Frequency Cepstral Coefficients (MFCCs) [Mean + Std = 40 features]
- Spectral Centroid [Mean + Std = 2 features]
- Spectral Rolloff [Mean + Std = 2 features]
- Zero Crossing Rate [Mean + Std = 2 features]
Total feature dimension: 46 features.
"""

from typing import Tuple, Dict, Any
import numpy as np
import librosa
from src.config import N_MFCC, N_FFT, HOP_LENGTH, N_MELS, TARGET_SAMPLE_RATE


def extract_features(y: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """
    Extracts a fixed 46-dimensional statistical acoustic feature vector from an audio waveform.
    """
    # 1. 20 Mel-Frequency Cepstral Coefficients (MFCCs)
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # 2. Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    sc_mean = np.mean(spectral_centroid)
    sc_std = np.std(spectral_centroid)

    # 3. Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    ro_mean = np.mean(spectral_rolloff)
    ro_std = np.std(spectral_rolloff)

    # 4. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)

    # Concatenate all into a single 1D vector (46 dimensions)
    feature_vector = np.hstack([
        mfcc_mean,       # 20
        mfcc_std,        # 20
        sc_mean, sc_std, # 2
        ro_mean, ro_std, # 2
        zcr_mean, zcr_std # 2
    ])

    return feature_vector.astype(np.float32)


def compute_mel_spectrogram(y: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """
    Computes a 2D Log-Mel Spectrogram in decibels for visualization and advanced modeling.
    """
    s_mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    s_db = librosa.power_to_db(s_mel, ref=np.max)
    return s_db


def extract_waveform_summary(y: np.ndarray, num_points: int = 150) -> list:
    """
    Downsamples the raw waveform into a small array of peak amplitudes
    for fast, lightweight interactive client-side rendering.
    """
    if len(y) == 0:
        return []
    step = max(1, len(y) // num_points)
    sampled = y[::step][:num_points]
    return [round(float(v), 3) for v in sampled]
