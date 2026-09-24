"""
Server launcher for Sound-Based Machine Health Monitor.
Starts the FastAPI application with Uvicorn.
"""

import sys
import uvicorn

if __name__ == "__main__":
    print("=" * 65)
    print("  SOUND-BASED MACHINE HEALTH MONITOR")
    print("  Acoustic Condition Monitoring & Anomaly Detection Prototype")
    print("=" * 65)
    print("  Web Dashboard: http://127.0.0.1:8000")
    print("  API Docs (Swagger): http://127.0.0.1:8000/docs")
    print("=" * 65)
    uvicorn.run("src.app:app", host="127.0.0.1", port=8000, reload=True)
