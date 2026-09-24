# Sound-Based Machine Health Monitor

A prototype acoustic condition monitoring system combining audio signal processing and machine learning to classify machine operating sounds as **Normal**, **Abnormal**, or **Uncertain**.

---

## 📌 Project Overview

- **Problem:** Industrial machinery downtime is costly. Traditional time-based maintenance is rigid, while continuous manual auditory inspection does not scale.
- **Approach:** Analyzes sound emissions (1–10 seconds) from rotating machinery (fans, pumps, valves) to identify acoustic anomalies such as bearing flutter, cavitation, and friction.
- **Decision Engine:** Evaluates maximum class probability against threshold $\tau = 0.65$; predictions below 65% confidence are classified as **Uncertain** rather than forced into binary outputs.
- **Architecture:** Decoupled FastAPI backend, `librosa` acoustic preprocessing (16 kHz mono), Random Forest classifier on 46-dimensional statistical features, SQLite persistence, and responsive browser dashboard with live recording and spectrogram visualization.

---

## 🚀 Quick Start Guide

### 1. Activate Environment
Open PowerShell in the project directory:
```powershell
cd C:\Users\lovep\.gemini\antigravity\scratch\sound_machine_health_monitor
.\venv\Scripts\Activate.ps1
```

### 2. Launch the Application
```powershell
python run_server.py
```
Open your browser and navigate to:
- **Interactive Web Dashboard:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger API Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🧪 Running Automated Test Suite

Execute the comprehensive test suite (unit, integration, and negative boundary tests):
```powershell
python -m pytest -v
```

### Test Coverage Highlights:
- `tests/test_preprocessing.py`: Validation rules, empty file rejection (`ERR-002`), oversized file limits (`ERR-004`), silence detection below -45 dBFS (`ERR-002`), duration bounds (1.0s–10.0s), and corrupted stream recovery (`ERR-003`).
- `tests/test_features.py`: Exact 46-dimensional vector verification, numerical integrity, deterministic extraction, and Mel-spectrogram tensor validation.
- `tests/test_model.py`: Artifact deserialization, probability distribution bounds, uncertainty thresholding logic ($\tau = 0.65$), and NaN exception safety.
- `tests/test_api.py`: FastAPI endpoints (`/api/health`, `/api/analyze`, `/api/history`, `/api/samples`), SQLite logging, and error handling.

---

## 📁 Project Structure

```
sound_machine_health_monitor/
├── data/
│   └── samples/              # Preloaded demo audio clips (Normal & Abnormal)
├── docs/
│   ├── PRD.md                # Product Requirements Document
│   └── SRS.pdf               # Software Requirements Specification
├── models/
│   ├── classifier.joblib     # Serialized Random Forest model
│   └── scaler.joblib         # Serialized feature scaler
├── src/
│   ├── config.py             # Global constants & threshold parameters
│   ├── database.py           # SQLite persistence & query module
│   ├── features.py           # 46-dim MFCC & spectral feature extractor
│   ├── model.py              # ML inference & uncertainty logic
│   ├── preprocessing.py      # Audio validation, decoding, & resampling
│   ├── train.py              # Model training & synthetic data benchmark
│   ├── visualization.py      # Base64 spectrogram & waveform rendering
│   └── app.py                # FastAPI REST API & routes
├── static/
│   ├── index.html            # Web dashboard
│   ├── css/style.css         # Dark-slate responsive theme
│   └── js/app.js             # Client upload, recording & charting
├── tests/
│   ├── conftest.py           # Fixtures & mock audio generators
│   ├── test_api.py           # API integration tests
│   ├── test_features.py      # Acoustic feature tests
│   ├── test_model.py         # Classifier & uncertainty tests
│   └── test_preprocessing.py # Input validation tests
├── pytest.ini                # Pytest configuration
├── requirements.txt          # Pinned project dependencies
└── run_server.py             # Server launcher script
```

---

## 📊 REST API Specification

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Service status, model readiness, and supported categories |
| `POST` | `/api/analyze` | Ingests audio (`file`, `machine_category`), returns prediction, confidence, & plots |
| `GET` | `/api/history` | Fetches recent analysis records stored in SQLite |
| `GET` | `/api/samples` | Lists available preloaded demonstration audio clips |
| `GET` | `/api/samples/{name}` | Downloads or streams a specific demo audio clip |
| `GET` | `/` | Serves the interactive health monitor dashboard |
