"""
Database Models and Connection for PhishNet
============================================
PostgreSQL database for storing scan history and user feedback.
Used for model improvement through human-in-the-loop learning.
"""

import os
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    Text,
    ForeignKey,
    JSON,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Database URL from environment or default
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://phishnet_admin:PhishNet2024Secure@phishnet-db.c83quikqw26n.us-east-1.rds.amazonaws.com:5432/phishnet",
)

# Create engine with connection pooling
# Note: Connection may fail locally if RDS is in private VPC
try:
    engine = create_engine(
        DATABASE_URL,
        pool_size=5,
        max_overflow=10,
        pool_timeout=10,  # Shorter timeout for faster failure
        pool_recycle=1800,  # Recycle connections after 30 minutes
        connect_args={"connect_timeout": 5},  # 5 second connect timeout
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    DB_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Database connection not available: {e}")
    engine = None
    SessionLocal = None
    DB_AVAILABLE = False

Base = declarative_base()


class Scan(Base):
    """
    Stores every URL scan with prediction details.
    Used for analytics and feeding corrections back into training.
    """

    __tablename__ = "scans"

    id = Column(Integer, primary_key=True, index=True)
    url = Column(Text, nullable=False, index=True)

    # Final ensemble prediction
    prediction = Column(Integer)  # 0=legitimate, 1=phishing
    confidence = Column(Float)  # Ensemble confidence score

    # Individual model scores (3-model ensemble)
    url_model_score = Column(Float)
    dns_model_score = Column(Float)
    whois_model_score = Column(Float)

    # LLM explanation
    explanation = Column(Text)

    # Metadata
    model_version = Column(String(50))
    source = Column(String(50))  # 'extension', 'api', 'web'
    scanned_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationship to feedback
    feedback = relationship("Feedback", back_populates="scan", uselist=False)


class Feedback(Base):
    """
    Stores user corrections for model improvement.
    When a user says "this prediction was wrong", we store the correction
    and use it in the next training cycle.
    """

    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    scan_id = Column(Integer, ForeignKey("scans.id"), nullable=False, index=True)

    # Prediction correction
    correct_label = Column(Integer)  # User's correction: 0=legitimate, 1=phishing

    # Explanation feedback
    explanation_helpful = Column(Boolean)  # Was the explanation useful?
    explanation_comment = Column(Text)  # Optional user comment

    # Metadata
    source = Column(String(50))  # 'extension', 'api', 'web'
    submitted_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationship to scan
    scan = relationship("Scan", back_populates="feedback")


class ModelMetrics(Base):
    """
    Tracks model performance over time.
    Updated during training runs with accuracy, F1, etc.
    """

    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    model_type = Column(String(20))  # 'url', 'dns', 'whois', 'ensemble'
    model_name = Column(String(100))

    # Performance metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    auc_roc = Column(Float)

    # Training details
    training_samples = Column(Integer)
    feedback_corrections = Column(Integer)  # How many user corrections used

    # Metadata
    trained_at = Column(DateTime, default=datetime.utcnow, index=True)
    deployed = Column(Boolean, default=False)


def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created/verified")


def get_db():
    """
    Dependency for FastAPI endpoints.
    Yields a database session and ensures cleanup.
    """
    if SessionLocal is None:
        raise Exception(
            "Database not available - RDS may not be accessible from this environment"
        )
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Pydantic schemas for API
from pydantic import BaseModel
from typing import Optional


class ScanCreate(BaseModel):
    """Schema for creating a scan record."""

    url: str
    prediction: int
    confidence: float
    url_model_score: Optional[float] = None
    dns_model_score: Optional[float] = None
    whois_model_score: Optional[float] = None
    explanation: Optional[str] = None
    model_version: Optional[str] = None
    source: str = "api"


class ScanResponse(BaseModel):
    """Schema for scan response."""

    id: int
    url: str
    prediction: int
    confidence: float
    explanation: Optional[str] = None
    scanned_at: datetime

    class Config:
        from_attributes = True


class FeedbackCreate(BaseModel):
    """Schema for submitting feedback."""

    scan_id: int
    correct_label: Optional[int] = None  # 0 or 1
    explanation_helpful: Optional[bool] = None
    explanation_comment: Optional[str] = None
    source: str = "api"


class FeedbackResponse(BaseModel):
    """Schema for feedback response."""

    id: int
    scan_id: int
    correct_label: Optional[int] = None
    explanation_helpful: Optional[bool] = None
    submitted_at: datetime

    class Config:
        from_attributes = True
