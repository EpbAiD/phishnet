# ===============================================================
# src/api/schemas.py
# Pydantic models for phishing API (URL and WHOIS)
# ===============================================================

from pydantic import BaseModel
from typing import Optional


class URLPredictRequest(BaseModel):
    """Request schema for URL prediction."""

    url: str  # allow any string; we'll sanitize inside


class URLPredictResponse(BaseModel):
    """Response schema for URL prediction."""

    url: str
    risk_score: float  # P(phishing) in [0, 1]
    is_phishing: bool  # risk_score >= threshold
    threshold: float  # decision threshold used
    model_name: str  # e.g., "lgbm_url_v1"
    latency_ms: float  # end-to-end model latency
    explanation: str  # LLM-generated human-readable explanation (REQUIRED for trust)
    verdict: str  # "safe" or "suspicious" (user-friendly)
    confidence: str  # "high", "medium", or "low"
    debug: Optional[dict] = None  # optional small debug info


class WHOISPredictRequest(BaseModel):
    """Request schema for WHOIS prediction."""

    url: str  # allow any string; we'll sanitize inside


class WHOISPredictResponse(BaseModel):
    """Response schema for WHOIS prediction."""

    url: str
    risk_score: float  # P(phishing) in [0, 1]
    is_phishing: bool  # risk_score >= threshold
    threshold: float  # decision threshold used
    model_name: str  # e.g., "lgbm_whois_v1"
    latency_ms: float  # end-to-end model latency
    explanation: str  # LLM-generated human-readable explanation (REQUIRED for trust)
    verdict: str  # "safe" or "suspicious" (user-friendly)
    confidence: str  # "high", "medium", or "low"
    debug: Optional[dict] = None  # optional small debug info


class DNSPredictRequest(BaseModel):
    """Request schema for DNS prediction."""

    url: str  # allow any string; we'll sanitize inside


class DNSPredictResponse(BaseModel):
    """Response schema for DNS prediction."""

    url: str
    risk_score: float  # P(phishing) in [0, 1]
    is_phishing: bool  # risk_score >= threshold
    threshold: float  # decision threshold used
    model_name: str  # e.g., "catboost_dns_v1"
    latency_ms: float  # end-to-end model latency
    explanation: str  # LLM-generated human-readable explanation (REQUIRED for trust)
    verdict: str  # "safe" or "suspicious" (user-friendly)
    confidence: str  # "high", "medium", or "low"
    debug: Optional[dict] = None  # optional small debug info


class EnsemblePredictRequest(BaseModel):
    """Request schema for ensemble prediction."""

    url: str  # allow any string; we'll sanitize inside


class EnsemblePredictResponse(BaseModel):
    """Response schema for ensemble prediction."""

    url: str
    risk_score: float  # Ensemble P(phishing) in [0, 1]
    is_phishing: bool  # risk_score >= threshold
    threshold: float  # decision threshold used
    model_name: str  # e.g., "ensemble_url+whois_v1"
    latency_ms: float  # total end-to-end latency
    explanation: str  # LLM-generated human-readable explanation (REQUIRED for trust)
    verdict: str  # "safe" or "suspicious" (user-friendly)
    confidence: str  # "high", "medium", or "low"
    debug: Optional[dict] = None  # includes individual model predictions


class ExplainRequest(BaseModel):
    """Request schema for explanation generation."""

    url: str
    include_shap: bool = False  # Whether to include SHAP feature importance


class ExplainResponse(BaseModel):
    """Response schema for explanation generation."""

    url: str
    explanation: str  # Layman explanation (simple, specific)
    technical_explanation: Optional[str] = None  # Technical SHAP analysis for engineers
    predictions: dict  # URL/DNS/WHOIS/Ensemble predictions
    verdict: str  # "phishing" or "legit"
    confidence: str  # "high", "medium", or "low"
    latency_ms: float  # Total latency including LLM generation
    shap_features: Optional[dict] = None  # Optional SHAP top features
