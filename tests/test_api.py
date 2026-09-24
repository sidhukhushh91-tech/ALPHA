"""
Integration and API endpoint tests (src/app.py).
Tests end-to-end request/response contracts, HTTP status codes, and error responses
corresponding to FR-001, FR-003, FR-004, FR-007..FR-012, FR-014, and ERR-001..ERR-008.
"""

def test_health_endpoint(client):
    """Verifies system diagnostics status endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["model_loaded"] is True
    assert data["database"] == "sqlite_ready"


def test_analyze_valid_audio(client, valid_audio_bytes):
    """Verifies end-to-end analysis of a valid machine sound upload."""
    files = {"file": ("motor_sample.wav", valid_audio_bytes, "audio/wav")}
    data = {"machine_category": "fan"}

    response = client.post("/api/analyze", files=files, data=data)
    assert response.status_code == 200
    res = response.json()

    assert res["status"] == "success"
    assert res["predicted_class"] in ["normal", "abnormal", "uncertain"]
    assert 0.0 <= res["confidence"] <= 1.0
    assert res["machine_category"] == "fan"
    assert "visualizations" in res
    assert res["visualizations"]["waveform_image"].startswith("data:image/png;base64,")
    assert res["visualizations"]["spectrogram_image"].startswith("data:image/png;base64,")
    assert res["execution_time_ms"] > 0


def test_analyze_silent_audio_rejection(client, silent_audio_bytes):
    """TC-N-002: Submitting silent audio returns HTTP 422 and ERR-002."""
    files = {"file": ("silence.wav", silent_audio_bytes, "audio/wav")}
    response = client.post("/api/analyze", files=files)
    assert response.status_code == 422
    res = response.json()
    assert res["status"] == "error"
    assert res["error_code"] == "ERR-002"
    assert "silent or empty" in res["detail"]


def test_analyze_short_audio_rejection(client, short_audio_bytes):
    """TC-N-002: Submitting audio under 1.0s returns HTTP 422 and ERR-004."""
    files = {"file": ("short.wav", short_audio_bytes, "audio/wav")}
    response = client.post("/api/analyze", files=files)
    assert response.status_code == 422
    res = response.json()
    assert res["status"] == "error"
    assert res["error_code"] == "ERR-004"
    assert "below the minimum" in res["detail"]


def test_analyze_unsupported_format(client):
    """TC-N-001: Submitting an unsupported file format returns HTTP 415 and ERR-001."""
    files = {"file": ("script.py", b"print('hello world')", "text/plain")}
    response = client.post("/api/analyze", files=files)
    assert response.status_code == 415
    res = response.json()
    assert res["status"] == "error"
    assert res["error_code"] == "ERR-001"
    assert "Unsupported file format" in res["detail"]


def test_analyze_corrupted_audio(client, corrupted_audio_bytes):
    """TC-N-003: Submitting corrupted audio payload returns HTTP 422 and ERR-003."""
    files = {"file": ("corrupt.wav", corrupted_audio_bytes, "audio/wav")}
    response = client.post("/api/analyze", files=files)
    assert response.status_code == 422
    res = response.json()
    assert res["status"] == "error"
    assert res["error_code"] == "ERR-003"


def test_history_logging_and_retrieval(client, valid_audio_bytes):
    """Verifies that analyses are persisted to SQLite and retrieved via /api/history."""
    # Run an analysis to ensure at least one record exists
    files = {"file": ("history_test.wav", valid_audio_bytes, "audio/wav")}
    client.post("/api/analyze", files=files, data={"machine_category": "pump"})

    response = client.get("/api/history")
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"
    assert res["count"] >= 1
    assert any(r["audio_reference"] == "history_test.wav" for r in res["records"])


def test_demo_samples_endpoints(client):
    """Verifies listing and downloading preloaded demo samples."""
    # List samples
    list_res = client.get("/api/samples")
    assert list_res.status_code == 200
    samples = list_res.json()["samples"]
    assert len(samples) > 0
    assert "normal_fan_sample.wav" in samples

    # Download specific sample
    sample_res = client.get("/api/samples/normal_fan_sample.wav")
    assert sample_res.status_code == 200
    assert sample_res.headers["content-type"].startswith("audio/")


def test_root_dashboard_serving(client):
    """Verifies that the web dashboard HTML is served at the root URL."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Sound-Based Machine Health Monitor" in response.text
