"""
Database management module using SQLAlchemy with SQLite.
Manages schema definition, connection pooling, and historical analysis logs.
"""

from datetime import datetime
from typing import List, Optional
from sqlalchemy import Column, Integer, String, Float, DateTime, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from src.config import DATABASE_PATH

DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class AnalysisRecord(Base):
    """Represents a persisted machine audio analysis record."""
    __tablename__ = "analyses"

    analysis_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    machine_category = Column(String(50), default="unspecified", nullable=False)
    audio_reference = Column(String(255), nullable=False)
    predicted_class = Column(String(20), nullable=False)
    confidence = Column(Float, nullable=False)
    processing_status = Column(String(20), default="success", nullable=False)
    execution_time_ms = Column(Integer, default=0, nullable=False)
    result_summary = Column(String(500), nullable=True)

    def to_dict(self):
        """Serializes the ORM object into a dictionary."""
        return {
            "analysis_id": self.analysis_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "machine_category": self.machine_category,
            "audio_reference": self.audio_reference,
            "predicted_class": self.predicted_class,
            "confidence": round(self.confidence, 4),
            "processing_status": self.processing_status,
            "execution_time_ms": self.execution_time_ms,
            "result_summary": self.result_summary,
        }


def init_db():
    """Initializes tables in the SQLite database."""
    Base.metadata.create_all(bind=engine)


def log_analysis(
    audio_reference: str,
    predicted_class: str,
    confidence: float,
    machine_category: str = "unspecified",
    processing_status: str = "success",
    execution_time_ms: int = 0,
    result_summary: Optional[str] = None
) -> AnalysisRecord:
    """Inserts a new analysis entry into the database."""
    db = SessionLocal()
    try:
        record = AnalysisRecord(
            audio_reference=audio_reference,
            predicted_class=predicted_class,
            confidence=confidence,
            machine_category=machine_category,
            processing_status=processing_status,
            execution_time_ms=execution_time_ms,
            result_summary=result_summary
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record
    finally:
        db.close()


def get_history(limit: int = 50) -> List[dict]:
    """Retrieves recent analysis records sorted by most recent first."""
    db = SessionLocal()
    try:
        records = db.query(AnalysisRecord).order_by(AnalysisRecord.timestamp.desc()).limit(limit).all()
        return [r.to_dict() for r in records]
    finally:
        db.close()


def get_analysis_by_id(analysis_id: int) -> Optional[dict]:
    """Retrieves a single record by its analysis ID."""
    db = SessionLocal()
    try:
        record = db.query(AnalysisRecord).filter(AnalysisRecord.analysis_id == analysis_id).first()
        return record.to_dict() if record else None
    finally:
        db.close()
