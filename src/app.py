"""
FastAPI application service for Sound-Based Machine Health Monitor.
Exposes REST endpoints for audio ingestion, health diagnostics, history retrieval,
and serves the web dashboard UI.
"""

import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from src.config import (
    BASE_DIR,
    SAMPLES_DIR,
    DEFAULT_MACHINE_CATEGORY,
    VALID_MACHINE_CATEGORIES
)
from src.database import init_db, log_analysis, get_history
from src.preprocessing import preprocess_audio, AudioValidationError
from src.features import extract_features, extract_waveform_summary
from src.model import get_classifier, ModelInferenceError
from src.visualization import generate_spectrogram_base64, generate_waveform_base64


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle startup and shutdown handler."""
    init_db()
    # Eagerly load model
    try:
        classifier = get_classifier()
        print(f"Classifier initialized (loaded={classifier.is_loaded})")
    except Exception as e:
        print(f"Warning: Classifier initialization deferred: {e}")
    yield


app = FastAPI(
    title="Sound-Based Machine Health Monitor API",
    description="Acoustic condition monitoring API for industrial predictive maintenance",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local client development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    """System health and diagnostic status endpoint."""
    classifier = get_classifier()
    return {
        "status": "online",
        "service": "Sound-Based Machine Health Monitor",
        "model_loaded": classifier.is_loaded,
        "database": "sqlite_ready",
        "supported_categories": VALID_MACHINE_CATEGORIES
    }


@app.post("/api/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    machine_category: str = Form(DEFAULT_MACHINE_CATEGORY)
):
    """
    Main audio analysis endpoint.
    Accepts an uploaded or recorded audio clip, runs the complete
    signal processing and ML inference pipeline, logs to SQLite,
    and returns classification results with acoustic visualizations.
    """
    start_time = time.perf_counter()
    filename = file.filename or "recording.webm"
    category = machine_category.lower() if machine_category in VALID_MACHINE_CATEGORIES else DEFAULT_MACHINE_CATEGORY

    # 1. Ingestion
    try:
        file_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read upload payload: {str(e)}")

    # 2. Audio Preprocessing & Validation
    try:
        y, sr = preprocess_audio(file_bytes, filename)
    except AudioValidationError as ave:
        return JSONResponse(
            status_code=ave.status_code,
            content={
                "status": "error",
                "error_code": ave.error_code,
                "detail": ave.message
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error_code": "ERR-003",
                "detail": f"Audio processing error: {str(e)}"
            }
        )

    # 3. Acoustic Feature Extraction
    try:
        feature_vector = extract_features(y, sr)
        waveform_summary = extract_waveform_summary(y)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error_code": "ERR-007",
                "detail": f"Feature extraction failure: {str(e)}"
            }
        )

    # 4. Machine Learning Inference & Decision Rule
    classifier = get_classifier()
    try:
        prediction = classifier.predict(feature_vector)
    except ModelInferenceError as mie:
        return JSONResponse(
            status_code=mie.status_code,
            content={
                "status": "error",
                "error_code": mie.error_code,
                "detail": mie.message
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error_code": "ERR-007",
                "detail": f"Model inference execution failure: {str(e)}"
            }
        )

    # 5. Visualizations Generation
    waveform_b64 = generate_waveform_base64(y, sr)
    spectrogram_b64 = generate_spectrogram_base64(y, sr)

    elapsed_ms = int((time.perf_counter() - start_time) * 1000)

    # 6. Database Logging
    record = None
    try:
        record = log_analysis(
            audio_reference=filename,
            predicted_class=prediction["predicted_class"],
            confidence=prediction["confidence"],
            machine_category=category,
            processing_status="success",
            execution_time_ms=elapsed_ms,
            result_summary=prediction["result_summary"]
        )
    except Exception as db_err:
        print(f"Warning: Database logging failed: {db_err}")

    return {
        "status": "success",
        "analysis_id": record.analysis_id if record else None,
        "timestamp": record.timestamp.isoformat() if record else None,
        "machine_category": category,
        "audio_reference": filename,
        "predicted_class": prediction["predicted_class"],
        "confidence": prediction["confidence"],
        "probabilities": prediction["probabilities"],
        "result_summary": prediction["result_summary"],
        "execution_time_ms": elapsed_ms,
        "visualizations": {
            "waveform_image": waveform_b64,
            "spectrogram_image": spectrogram_b64,
            "waveform_summary": waveform_summary
        }
    }


@app.get("/api/history")
async def get_analysis_history(limit: int = 50):
    """Fetches recent past analyses stored in the SQLite database."""
    try:
        records = get_history(limit=limit)
        return {"status": "success", "count": len(records), "records": records}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error_code": "ERR-008", "detail": f"Database read failure: {str(e)}"}
        )


@app.get("/api/samples")
async def list_demo_samples():
    """Lists preloaded demo audio clips available for testing."""
    if not SAMPLES_DIR.exists():
        return {"samples": []}
    files = [f.name for f in SAMPLES_DIR.glob("*.wav")]
    return {"status": "success", "samples": files}


@app.get("/api/samples/{sample_name}")
async def get_demo_sample(sample_name: str):
    """Serves a specific demo audio clip for playback or analysis."""
    sample_path = SAMPLES_DIR / sample_name
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Demo sample not found")
    return FileResponse(sample_path, media_type="audio/wav")


# Mount static assets and frontend UI
STATIC_DIR = BASE_DIR / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_index():
    """Serves the main application dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Sound-Based Machine Health Monitor API is running. UI dashboard at /static/index.html"}
