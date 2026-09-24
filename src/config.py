"""
Global configuration and hyperparameters for Sound-Based Machine Health Monitor.
Governs audio preprocessing parameters, threshold limits, model paths, and database settings.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SAMPLES_DIR = DATA_DIR / "samples"
DATABASE_PATH = BASE_DIR / "machine_health.db"

# Audio Pipeline Constraints
TARGET_SAMPLE_RATE = 16000  # 16 kHz standard (matches MIMII benchmark)
CHANNELS = 1                # Mono audio
MIN_DURATION_SEC = 1.0      # Minimum valid clip length
MAX_DURATION_SEC = 10.0     # Maximum valid clip length
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB maximum payload

# Silence & Quality Checks
SILENCE_DBFS_THRESHOLD = -45.0  # RMS decibels relative to full scale
TOP_DB_TRIM = 20.0              # Trim leading/trailing silence > 20 dB below peak

# Feature Extraction Parameters
N_MFCC = 20
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

# Machine Learning & Decision Rules
CONFIDENCE_THRESHOLD = 0.65     # tau = 0.65; max probability < 0.65 yields "uncertain"
MODEL_PATH = MODELS_DIR / "classifier.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"

# Supported Input Containers
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".webm", ".ogg", ".flac"}
SUPPORTED_MIME_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/mpeg",
    "audio/mp3",
    "audio/webm",
    "audio/ogg",
    "audio/flac",
    "application/octet-stream"  # Often sent by browsers for binary audio blobs
}

# Machine Categories
VALID_MACHINE_CATEGORIES = ["fan", "pump", "valve", "slider", "unspecified"]
DEFAULT_MACHINE_CATEGORY = "unspecified"
