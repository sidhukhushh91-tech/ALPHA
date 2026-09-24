"""
Audio visualization engine.
Generates server-side base64 PNG images for Mel-spectrograms and waveforms
for rich, responsive dashboard visualization.
"""

import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import librosa
import librosa.display

from src.config import TARGET_SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH


def generate_spectrogram_base64(y: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> str:
    """
    Renders a Mel-spectrogram in decibels and encodes it as a base64 PNG data URL.
    """
    fig, ax = plt.subplots(figsize=(6.5, 2.5), dpi=100)
    fig.patch.set_facecolor("#1e293b")  # Dark theme Slate-800
    ax.set_facecolor("#0f172a")        # Slate-900

    s_mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    s_db = librosa.power_to_db(s_mel, ref=np.max)

    img = librosa.display.specshow(
        s_db, sr=sr, hop_length=HOP_LENGTH, x_axis="time", y_axis="mel", ax=ax, cmap="magma"
    )

    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.xaxis.label.set_color("#94a3b8")
    ax.yaxis.label.set_color("#94a3b8")
    ax.set_title("Acoustic Mel-Spectrogram (Frequency vs. Time)", color="#f8fafc", fontsize=10, pad=8)
    for spine in ax.spines.values():
        spine.set_color("#334155")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def generate_waveform_base64(y: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> str:
    """
    Renders a clean amplitude waveform and encodes it as a base64 PNG data URL.
    """
    fig, ax = plt.subplots(figsize=(6.5, 1.8), dpi=100)
    fig.patch.set_facecolor("#1e293b")
    ax.set_facecolor("#0f172a")

    time_axis = np.linspace(0, len(y) / sr, len(y))
    ax.plot(time_axis, y, color="#38bdf8", linewidth=0.8, alpha=0.9)
    ax.set_ylim(-1.05, 1.05)

    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.xaxis.label.set_color("#94a3b8")
    ax.yaxis.label.set_color("#94a3b8")
    ax.set_title("Time-Domain Waveform (Normalized Amplitude)", color="#f8fafc", fontsize=10, pad=8)
    for spine in ax.spines.values():
        spine.set_color("#334155")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"
