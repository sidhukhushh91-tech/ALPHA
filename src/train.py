"""
Model training, evaluation, and benchmark generator script.
Simulates realistic acoustic condition monitoring profiles (harmonically rich machine hum,
bearing flutter, cavitation impulses, and mechanical friction) to train and evaluate
the baseline Random Forest classifier, producing serialized models and demo audio samples.
"""

from pathlib import Path
import numpy as np
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

from src.config import (
    MODELS_DIR,
    SAMPLES_DIR,
    MODEL_PATH,
    SCALER_PATH,
    TARGET_SAMPLE_RATE
)
from src.features import extract_features


def generate_synthetic_machine_sound(
    duration: float = 3.0,
    sr: int = TARGET_SAMPLE_RATE,
    is_abnormal: bool = False,
    machine_type: str = "fan"
) -> np.ndarray:
    """
    Synthesizes physically grounded acoustic signatures for industrial machinery.
    - Normal: Steady fundamental motor hum (60/120 Hz) + blade pass frequency + smooth airflow noise.
    - Abnormal: Bearing flutter, tonal screech (1.8 kHz - 3.2 kHz harmonics), impulsive impact spikes, cavitation.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    rng = np.random.default_rng()

    if machine_type == "fan":
        # Base motor rotation (60 Hz fundamental + harmonics)
        base_hum = (
            0.30 * np.sin(2 * np.pi * 60 * t) +
            0.15 * np.sin(2 * np.pi * 120 * t) +
            0.08 * np.sin(2 * np.pi * 240 * t)
        )
        # Broadband aerodynamic airflow noise
        noise = rng.normal(0, 0.05, size=len(t))
        sound = base_hum + noise

        if is_abnormal:
            # Mechanical fault: Blade imbalance vibration + high-frequency bearing squeal
            screech_freq = 2400.0 + 150.0 * np.sin(2 * np.pi * 3.5 * t)
            fault_screech = 0.25 * np.sin(2 * np.pi * screech_freq * t)
            # Periodic impact clicks (bearing fault every 0.25 sec)
            impacts = np.zeros_like(t)
            impact_indices = np.arange(0, len(t), int(sr * 0.25))
            for idx in impact_indices:
                end = min(idx + int(sr * 0.02), len(t))
                decay = np.exp(-np.linspace(0, 5, end - idx))
                impacts[idx:end] += rng.normal(0, 0.40, size=end - idx) * decay
            sound += fault_screech + impacts

    else:  # Pump
        # Hydraulic cyclic hum
        base_hum = (
            0.25 * np.sin(2 * np.pi * 50 * t) +
            0.18 * np.sin(2 * np.pi * 100 * t) +
            0.10 * np.sin(2 * np.pi * 300 * t)
        )
        noise = rng.normal(0, 0.08, size=len(t))
        sound = base_hum + noise

        if is_abnormal:
            # Cavitation: violent high-energy acoustic bursts and friction modulation
            cavitation = rng.normal(0, 0.25, size=len(t)) * (np.sin(2 * np.pi * 8 * t) > 0.3)
            friction = 0.20 * np.sin(2 * np.pi * 1800 * t + rng.normal(0, 0.5, size=len(t)))
            sound += cavitation + friction

    # Normalize amplitude
    sound = sound / (np.max(np.abs(sound)) + 1e-6) * 0.85
    return sound.astype(np.float32)


def generate_demo_samples():
    """Generates authentic demo audio clips for UI testing and evaluation presentation."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    samples = [
        ("normal_fan_sample.wav", False, "fan"),
        ("abnormal_fan_sample.wav", True, "fan"),
        ("normal_pump_sample.wav", False, "pump"),
        ("abnormal_pump_sample.wav", True, "pump"),
    ]

    for fname, is_abnormal, mtype in samples:
        path = SAMPLES_DIR / fname
        audio = generate_synthetic_machine_sound(duration=4.0, is_abnormal=is_abnormal, machine_type=mtype)
        sf.write(str(path), audio, TARGET_SAMPLE_RATE)
        print(f"Generated sample audio: {path.name}")


def train_and_evaluate(num_samples_per_class: int = 250):
    """
    Builds the dataset, extracts 46-dimensional features, trains the Random Forest classifier,
    evaluates performance on a held-out test split, and exports the serialized model.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("STARTING MACHINE HEALTH MODEL TRAINING PIPELINE")
    print("=" * 60)

    features = []
    labels = []

    print(f"Synthesizing {num_samples_per_class * 2} training clips across Fan & Pump categories...")
    for _ in range(num_samples_per_class):
        # Normal samples
        mtype = "fan" if np.random.rand() > 0.5 else "pump"
        y_norm = generate_synthetic_machine_sound(duration=3.0, is_abnormal=False, machine_type=mtype)
        feat_norm = extract_features(y_norm, TARGET_SAMPLE_RATE)
        features.append(feat_norm)
        labels.append("normal")

        # Abnormal samples
        y_abnorm = generate_synthetic_machine_sound(duration=3.0, is_abnormal=True, machine_type=mtype)
        feat_abnorm = extract_features(y_abnorm, TARGET_SAMPLE_RATE)
        features.append(feat_abnorm)
        labels.append("abnormal")

    X = np.array(features)
    y = np.array(labels)
    print(f"Dataset generated. Feature matrix shape: {X.shape}")

    # Stratified Train/Test Split (75% Train, 25% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Standardize Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Baseline Random Forest
    print("Training Random Forest Classifier (100 estimators)...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        class_weight="balanced"
    )
    clf.fit(X_train_scaled, y_train)

    # Evaluate on Held-Out Test Set
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=["normal", "abnormal"])

    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS ON HELD-OUT TEST SET (Accuracy: {acc * 100:.2f}%)")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=["normal", "abnormal"]))
    print("Confusion Matrix:")
    print(f"                Predicted Normal   Predicted Abnormal")
    print(f"Actual Normal        {cm[0][0]:<18} {cm[0][1]}")
    print(f"Actual Abnormal      {cm[1][0]:<18} {cm[1][1]}")
    print("=" * 60)

    # Serialize Artifacts
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved classifier: {MODEL_PATH}")
    print(f"Saved scaler:     {SCALER_PATH}")

    # Generate preloaded sample audio files
    generate_demo_samples()
    print("Model training and sample asset generation complete.\n")


if __name__ == "__main__":
    train_and_evaluate()
