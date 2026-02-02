# ===============================================================
# src/api/app.py
# FastAPI app for URL-only phishing prediction
# ---------------------------------------------------------------
# - Normalizes incomplete URLs (e.g., "youtube.com" → "https://youtube.com")
# - /health : simple health check with model version info
# - /predict/url : takes URL, returns P(phishing) + label
# - /feedback : submit user corrections for model improvement
# - HOT RELOAD: Automatically reloads models from S3 when updated
# ===============================================================
import sys, os
import asyncio
import json
import logging
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import urlparse
from sqlalchemy.orm import Session
from typing import List, Optional
import boto3
from botocore.exceptions import ClientError

from src.api.schemas import (
    URLPredictRequest,
    URLPredictResponse,
    WHOISPredictRequest,
    WHOISPredictResponse,
    DNSPredictRequest,
    DNSPredictResponse,
    EnsemblePredictRequest,
    EnsemblePredictResponse,
    ExplainRequest,
    ExplainResponse,
)
from src.api.predict_utils import (
    predict_url_risk,
    predict_whois_risk,
    predict_dns_risk,
    predict_ensemble_risk,
)
from src.api.model_loader import load_url_model, load_whois_model, load_dns_model, clear_model_cache
from src.api.llm_explainer import generate_explanation
from src.api.database import (
    get_db,
    init_db,
    Scan,
    Feedback,
    ScanCreate,
    ScanResponse,
    FeedbackCreate,
    FeedbackResponse,
    DB_AVAILABLE,
    SessionLocal,
)
import time
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===============================================================
# MODEL HOT RELOAD CONFIGURATION
# ===============================================================
S3_BUCKET = os.environ.get('S3_BUCKET', 'phishnet-data')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
MODEL_CHECK_INTERVAL = int(os.environ.get('MODEL_CHECK_INTERVAL', 300))  # 5 minutes
ENABLE_HOT_RELOAD = os.environ.get('ENABLE_HOT_RELOAD', 'true').lower() == 'true'

# Global state for model versioning
model_state = {
    'version': None,
    'last_check': None,
    'last_reload': None,
    'reload_count': 0,
    'enabled': ENABLE_HOT_RELOAD,
}

app = FastAPI(
    title="Phishing Detection API",
    version="4.0.0",
    description="Phishing risk scoring with URL, WHOIS, and DNS models. Includes feedback for model improvement.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


# ---------------- Normalize Incoming URLs ------------------
def normalize_input_url(raw_url: str) -> str:
    """Ensure URL has a scheme; prepend https:// if missing."""
    raw_url = raw_url.strip()
    parsed = urlparse(raw_url)

    # If scheme missing, prepend https
    if not parsed.scheme:
        return "https://" + raw_url
    return raw_url


# ===============================================================
# MODEL HOT RELOAD FUNCTIONS
# ===============================================================

def get_s3_model_version() -> Optional[str]:
    """Check S3 for the current model version (last_trained timestamp)."""
    try:
        s3 = boto3.client('s3', region_name=AWS_REGION)
        response = s3.get_object(Bucket=S3_BUCKET, Key='models/production_metadata.json')
        metadata = json.loads(response['Body'].read().decode('utf-8'))
        return metadata.get('last_trained')
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logger.warning("No production_metadata.json found in S3")
            return None
        logger.error(f"Error checking S3 model version: {e}")
        return None
    except Exception as e:
        logger.error(f"Error checking S3 model version: {e}")
        return None


def download_models_from_s3() -> bool:
    """Download latest models from S3 to local models/ directory."""
    try:
        s3 = boto3.client('s3', region_name=AWS_REGION)
        models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
        os.makedirs(models_dir, exist_ok=True)

        # List all model files in S3
        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix='models/')

        if 'Contents' not in response:
            logger.warning("No models found in S3")
            return False

        downloaded = 0
        for obj in response['Contents']:
            key = obj['Key']
            if key.endswith('.pkl') or key.endswith('.json'):
                filename = os.path.basename(key)
                local_path = os.path.join(models_dir, filename)
                s3.download_file(S3_BUCKET, key, local_path)
                downloaded += 1

        logger.info(f"Downloaded {downloaded} model files from S3")
        return downloaded > 0
    except Exception as e:
        logger.error(f"Error downloading models from S3: {e}")
        return False


def reload_models():
    """Clear model cache and reload models."""
    try:
        # Clear the model cache to force reload
        clear_model_cache()

        # Reload models
        model, cols, thr = load_url_model()
        logger.info(f"Reloaded URL model: {type(model).__name__} | {len(cols)} features")

        model, cols, thr = load_whois_model()
        logger.info(f"Reloaded WHOIS model: {type(model).__name__} | {len(cols)} features")

        return True
    except Exception as e:
        logger.error(f"Error reloading models: {e}")
        return False


async def model_hot_reload_task():
    """Background task to check for and load new models from S3."""
    global model_state

    logger.info(f"Starting model hot reload task (interval: {MODEL_CHECK_INTERVAL}s)")

    while True:
        try:
            if not model_state['enabled']:
                await asyncio.sleep(MODEL_CHECK_INTERVAL)
                continue

            model_state['last_check'] = datetime.now()

            # Check S3 for new model version
            s3_version = get_s3_model_version()

            if s3_version and s3_version != model_state['version']:
                logger.info(f"New model version detected: {model_state['version']} -> {s3_version}")

                # Download new models
                if download_models_from_s3():
                    # Reload models into memory
                    if reload_models():
                        model_state['version'] = s3_version
                        model_state['last_reload'] = datetime.now()
                        model_state['reload_count'] += 1
                        logger.info(f"Models reloaded successfully (version: {s3_version})")
                    else:
                        logger.error("Failed to reload models after download")
                else:
                    logger.error("Failed to download models from S3")
            else:
                logger.debug(f"No new models (current: {model_state['version']})")

        except Exception as e:
            logger.error(f"Error in hot reload task: {e}")

        await asyncio.sleep(MODEL_CHECK_INTERVAL)


@app.on_event("startup")
async def _startup_event():
    """Warm model cache on server boot and start hot reload task."""
    global model_state

    # Load URL model
    try:
        model, cols, thr = load_url_model()
        print(
            f"🔐 [startup] Loaded URL model: {type(model).__name__} | "
            f"{len(cols)} features | threshold={thr}"
        )
    except Exception as e:
        print(f"🔥 [startup] ERROR loading URL model: {e}")

    # Load WHOIS model
    try:
        model, cols, thr = load_whois_model()
        print(
            f"🔐 [startup] Loaded WHOIS model: {type(model).__name__} | "
            f"{len(cols)} features | threshold={thr}"
        )
    except Exception as e:
        print(f"🔥 [startup] ERROR loading WHOIS model: {e}")

    # Get initial model version from S3
    try:
        s3_version = get_s3_model_version()
        if s3_version:
            model_state['version'] = s3_version
            model_state['last_reload'] = datetime.now()
            logger.info(f"Initial model version: {s3_version}")
    except Exception as e:
        logger.warning(f"Could not get initial model version from S3: {e}")

    # Start hot reload background task
    if ENABLE_HOT_RELOAD:
        asyncio.create_task(model_hot_reload_task())
        logger.info("Model hot reload task started")
    else:
        logger.info("Model hot reload disabled")

    # Load DNS model - COMMENTED OUT (using API from VM for live DNS data)
    # try:
    #     model, cols, thr = load_dns_model()
    #     print(
    #         f"🔐 [startup] Loaded DNS model: {type(model).__name__} | "
    #         f"{len(cols)} features | threshold={thr}"
    #     )
    # except Exception as e:
    #     print(f"🔥 [startup] ERROR loading DNS model: {e}")


@app.get("/health")
def health_check():
    """Health check with model version info."""
    return {
        "status": "ok",
        "model_version": model_state['version'],
        "last_reload": model_state['last_reload'].isoformat() if model_state['last_reload'] else None,
        "reload_count": model_state['reload_count'],
        "hot_reload_enabled": model_state['enabled'],
    }


def save_scan_to_db(
    url: str,
    prediction: int,
    confidence: float,
    url_score: float = None,
    dns_score: float = None,
    whois_score: float = None,
    explanation: str = None,
    source: str = "api"
) -> Optional[int]:
    """
    Save a scan to the database and return the scan_id.
    Returns None if database is unavailable.
    """
    if not DB_AVAILABLE or SessionLocal is None:
        return None

    try:
        db = SessionLocal()
        db_scan = Scan(
            url=url,
            prediction=prediction,
            confidence=confidence,
            url_model_score=url_score,
            dns_model_score=dns_score,
            whois_model_score=whois_score,
            explanation=explanation,
            model_version=model_state.get('version'),
            source=source,
        )
        db.add(db_scan)
        db.commit()
        db.refresh(db_scan)
        scan_id = db_scan.id
        db.close()
        return scan_id
    except Exception as e:
        logger.warning(f"Failed to save scan to database: {e}")
        return None


@app.post("/predict/url", response_model=URLPredictResponse)
def predict_url(request: URLPredictRequest):
    """Predict phishing risk using URL features only."""
    t0 = time.time()
    url = normalize_input_url(request.url)

    prob, latency_ms, model_name, debug = predict_url_risk(url)
    _, _, threshold = load_url_model()
    is_phishing = prob >= threshold

    # Generate LLM explanation (REQUIRED for trust-building)
    from src.api.unified_explainer import (
        generate_unified_explanation,
        extract_features_for_explanation,
    )

    # Extract SHAP features for evidence-based explanation
    top_features = extract_features_for_explanation(url, model_type="url")

    # Generate explanation
    explanation, verdict, confidence = generate_unified_explanation(
        url=url,
        risk_score=prob,
        model_type="url",
        threshold=threshold,
        top_features=top_features,
    )

    total_latency_ms = (time.time() - t0) * 1000

    # Save scan to database for feedback collection
    scan_id = save_scan_to_db(
        url=url,
        prediction=1 if is_phishing else 0,
        confidence=prob,
        url_score=prob,
        explanation=explanation,
        source="api"
    )

    return URLPredictResponse(
        url=url,
        risk_score=prob,
        is_phishing=is_phishing,
        threshold=threshold,
        model_name=model_name,
        latency_ms=total_latency_ms,
        explanation=explanation,
        verdict=verdict,
        confidence=confidence,
        scan_id=scan_id,
        debug=debug,
    )


@app.post("/predict/whois", response_model=WHOISPredictResponse)
def predict_whois(request: WHOISPredictRequest):
    """Predict phishing risk using WHOIS features only."""
    t0 = time.time()
    url = normalize_input_url(request.url)

    prob, latency_ms, model_name, debug = predict_whois_risk(url)
    _, _, threshold = load_whois_model()
    is_phishing = prob >= threshold

    # Generate LLM explanation (REQUIRED for trust-building)
    from src.api.unified_explainer import (
        generate_unified_explanation,
        extract_features_for_explanation,
    )

    # Extract SHAP features for evidence-based explanation
    top_features = extract_features_for_explanation(url, model_type="whois")

    # Generate explanation
    explanation, verdict, confidence = generate_unified_explanation(
        url=url,
        risk_score=prob,
        model_type="whois",
        threshold=threshold,
        top_features=top_features,
    )

    total_latency_ms = (time.time() - t0) * 1000

    return WHOISPredictResponse(
        url=url,
        risk_score=prob,
        is_phishing=is_phishing,
        threshold=threshold,
        model_name=model_name,
        latency_ms=total_latency_ms,
        explanation=explanation,
        verdict=verdict,
        confidence=confidence,
        debug=debug,
    )


@app.post("/predict/dns", response_model=DNSPredictResponse)
def predict_dns(request: DNSPredictRequest):
    """Predict phishing risk using DNS features only."""
    t0 = time.time()
    url = normalize_input_url(request.url)

    prob, _, model_name, debug = predict_dns_risk(url)
    _, _, threshold = load_dns_model()
    is_phishing = prob >= threshold

    # Generate LLM explanation (REQUIRED for trust-building)
    from src.api.unified_explainer import (
        generate_unified_explanation,
        extract_features_for_explanation,
    )

    # Extract SHAP features for evidence-based explanation
    top_features = extract_features_for_explanation(url, model_type="dns")

    # Generate explanation
    explanation, verdict, confidence = generate_unified_explanation(
        url=url,
        risk_score=prob,
        model_type="dns",
        threshold=threshold,
        top_features=top_features,
    )

    total_latency_ms = (time.time() - t0) * 1000

    return DNSPredictResponse(
        url=url,
        risk_score=prob,
        is_phishing=is_phishing,
        threshold=threshold,
        model_name=model_name,
        latency_ms=total_latency_ms,
        explanation=explanation,
        verdict=verdict,
        confidence=confidence,
        debug=debug,
    )


@app.post("/predict/ensemble", response_model=EnsemblePredictResponse)
def predict_ensemble(request: EnsemblePredictRequest):
    """Predict phishing risk using ensemble of URL + WHOIS + DNS models."""
    t0 = time.time()
    url = normalize_input_url(request.url)

    prob, _, model_name, debug = predict_ensemble_risk(url)

    # Use URL model threshold as default for ensemble
    _, _, threshold = load_url_model()
    is_phishing = prob >= threshold

    # Generate LLM explanation (REQUIRED for trust-building)
    # For ensemble, we extract features from all three models for comprehensive explanation
    from src.api.unified_explainer import (
        generate_unified_explanation,
        extract_features_for_explanation,
    )

    # Extract SHAP features from URL model (primary signal)
    # For ensemble, we could extract from all models but URL is the strongest signal
    top_features = extract_features_for_explanation(url, model_type="url")

    # Generate explanation
    explanation, verdict, confidence = generate_unified_explanation(
        url=url,
        risk_score=prob,
        model_type="ensemble",
        threshold=threshold,
        top_features=top_features,
    )

    total_latency_ms = (time.time() - t0) * 1000

    # Save scan to database for feedback collection
    # Extract individual model scores from debug if available
    url_score = debug.get('url_prob') if debug else None
    dns_score = debug.get('dns_prob') if debug else None
    whois_score = debug.get('whois_prob') if debug else None

    scan_id = save_scan_to_db(
        url=url,
        prediction=1 if is_phishing else 0,
        confidence=prob,
        url_score=url_score,
        dns_score=dns_score,
        whois_score=whois_score,
        explanation=explanation,
        source="api"
    )

    return EnsemblePredictResponse(
        url=url,
        risk_score=prob,
        is_phishing=is_phishing,
        threshold=threshold,
        model_name=model_name,
        latency_ms=total_latency_ms,
        explanation=explanation,
        verdict=verdict,
        confidence=confidence,
        scan_id=scan_id,
        debug=debug,
    )


@app.post("/explain", response_model=ExplainResponse)
def explain_prediction(request: ExplainRequest):
    """
    Generate human-readable explanation for phishing detection result.
    Uses LLM (fine-tuned Phi-3) to explain model predictions.
    Extracts SHAP-based feature contributions from all models for evidence-based explanations.
    """
    t0 = time.time()
    url = normalize_input_url(request.url)
    parsed = urlparse(url)
    domain = parsed.netloc or parsed.path

    # Run all model predictions
    url_prob, _, _, _ = predict_url_risk(url)
    # DNS prediction COMMENTED OUT (using VM API for live DNS data)
    # dns_prob, _, _, _ = predict_dns_risk(url)
    dns_prob = None  # Will be fetched from VM API
    whois_prob, _, _, _ = predict_whois_risk(url)
    ensemble_prob, _, _, debug_info = predict_ensemble_risk(url)

    # Calculate confidence level
    ensemble_prob_val = ensemble_prob if not np.isnan(ensemble_prob) else 0.5
    confidence_score = abs(ensemble_prob_val - 0.5)
    if confidence_score > 0.3:
        confidence = "high"
    elif confidence_score > 0.15:
        confidence = "medium"
    else:
        confidence = "low"

    # Determine verdict
    _, _, threshold = load_url_model()
    verdict = "phishing" if ensemble_prob_val >= threshold else "legit"

    # User-friendly verdict for display
    user_friendly_verdict = "suspicious" if verdict == "phishing" else "safe"

    # Prepare predictions dict
    predictions = {
        "url_prob": float(url_prob) if not np.isnan(url_prob) else None,
        "dns_prob": (
            float(dns_prob) if dns_prob is not None and not np.isnan(dns_prob) else None
        ),  # DNS disabled (using VM API)
        "whois_prob": float(whois_prob) if not np.isnan(whois_prob) else None,
        "ensemble_prob": float(ensemble_prob_val),
        "verdict": verdict,  # Technical verdict for internal use
        "user_verdict": user_friendly_verdict,  # User-friendly verdict for display
    }

    # Extract per-request SHAP contributions for evidence-based explanations
    shap_features = None
    top_features_for_llm = None

    if request.include_shap:
        try:
            from src.api.shap_explainer_realtime import extract_shap_features_realtime
            from src.api.predict_utils import _features_dict_to_dataframe
            from src.features.url_features import extract_single_url_features
            from src.features.dns_ipwhois import extract_single_domain_features
            from src.features.whois import extract_single_whois_features
            from src.data_prep.dataset_builder import preprocess_features_for_inference
            import socket

            shap_features = {}
            top_features_for_llm = {}

            # --- URL Model SHAP Contributions ---
            try:
                url_model, url_cols, _ = load_url_model()
                # Extract estimator if wrapped in dict
                if isinstance(url_model, dict) and "model" in url_model:
                    url_model_obj = url_model["model"]
                else:
                    url_model_obj = url_model

                url_feats = extract_single_url_features(url)

                # Use SAME inference pipeline as predict_utils.py
                features = preprocess_features_for_inference(url_features=url_feats)
                X = _features_dict_to_dataframe(features, url_cols)

                url_important = extract_shap_features_realtime(
                    url_model_obj, X, "url", top_n=5
                )
                shap_features["url"] = url_important
                top_features_for_llm["url"] = url_important
            except Exception as e:
                print(f"⚠️ URL SHAP extraction failed: {e}")
                import traceback

                traceback.print_exc()
                shap_features["url"] = []
                top_features_for_llm["url"] = []

            # --- DNS Model Feature Importance --- COMMENTED OUT (using VM API)
            # try:
            #     dns_model, dns_cols, _ = load_dns_model()
            #     # Extract estimator if wrapped in dict
            #     if isinstance(dns_model, dict) and "model" in dns_model:
            #         dns_model_obj = dns_model["model"]
            #     else:
            #         dns_model_obj = dns_model
            #
            #     # Check if domain is resolvable
            #     try:
            #         socket.gethostbyname(domain)
            #         dnsf = extract_single_domain_features(domain)
            #
            #         # Use SAME inference pipeline as predict_utils.py
            #         features = preprocess_features_for_inference(url_features={}, dns_features=dnsf)
            #         X = _features_dict_to_dataframe(features, dns_cols)
            #
            #         dns_important = extract_important_features_with_values(dns_model_obj, X, "dns", top_n=5)
            #         shap_features["dns"] = dns_important
            #         top_features_for_llm["dns"] = dns_important
            #     except socket.gaierror:
            #         shap_features["dns"] = []
            #         top_features_for_llm["dns"] = []
            # except Exception as e:
            #     print(f"⚠️ DNS feature importance extraction failed: {e}")
            #     import traceback
            #     traceback.print_exc()
            #     shap_features["dns"] = []
            #     top_features_for_llm["dns"] = []

            # Set DNS features to empty for now (will be fetched from VM API)
            shap_features["dns"] = []
            top_features_for_llm["dns"] = []

            # --- WHOIS Model SHAP Contributions ---
            try:
                whois_model, whois_cols, _ = load_whois_model()
                # Extract estimator if wrapped in dict
                if isinstance(whois_model, dict) and "model" in whois_model:
                    whois_model_obj = whois_model["model"]
                else:
                    whois_model_obj = whois_model

                whof = extract_single_whois_features(domain, live_lookup=True)

                # Use SAME inference pipeline as predict_utils.py
                features = preprocess_features_for_inference(
                    url_features={}, whois_features=whof
                )
                X = _features_dict_to_dataframe(features, whois_cols)

                whois_important = extract_shap_features_realtime(
                    whois_model_obj, X, "whois", top_n=5
                )
                shap_features["whois"] = whois_important
                top_features_for_llm["whois"] = whois_important
            except Exception as e:
                print(f"⚠️ WHOIS SHAP extraction failed: {e}")
                import traceback

                traceback.print_exc()
                shap_features["whois"] = []
                top_features_for_llm["whois"] = []

        except Exception as e:
            print(f"⚠️ SHAP extraction failed: {e}")
            shap_features = None
            top_features_for_llm = None

    # Generate dual explanations (layman + technical)
    try:
        from src.api.dual_explainer import generate_dual_explanations

        explanations = generate_dual_explanations(
            url=url,
            domain=domain,
            predictions=predictions,
            top_features=top_features_for_llm,
        )
        layman_explanation = explanations["layman"]
        technical_explanation = explanations["technical"]
    except Exception as e:
        print(f"⚠️ Dual explanation generation failed, using fallback: {e}")
        import traceback

        traceback.print_exc()

        # Fallback to simple explanations
        if verdict == "phishing":
            layman_explanation = f"⚠️ This website is likely a phishing attempt ({ensemble_prob_val*100:.1f}% risk). Do not enter any personal information."
            technical_explanation = (
                f"Ensemble score: {ensemble_prob_val:.4f} (threshold: {threshold})"
            )
        else:
            layman_explanation = f"✅ This website appears legitimate ({(1-ensemble_prob_val)*100:.1f}% confidence). Exercise normal caution."
            technical_explanation = (
                f"Ensemble score: {ensemble_prob_val:.4f} (threshold: {threshold})"
            )

    total_latency_ms = (time.time() - t0) * 1000

    # Save scan to database for feedback collection
    scan_id = save_scan_to_db(
        url=url,
        prediction=1 if verdict == "phishing" else 0,
        confidence=ensemble_prob_val,
        url_score=url_prob if not np.isnan(url_prob) else None,
        dns_score=dns_prob if dns_prob is not None and not np.isnan(dns_prob) else None,
        whois_score=whois_prob if not np.isnan(whois_prob) else None,
        explanation=layman_explanation,
        source="api"
    )

    return ExplainResponse(
        url=url,
        explanation=layman_explanation,
        technical_explanation=technical_explanation,
        predictions=predictions,
        verdict=verdict,
        confidence=confidence,
        latency_ms=total_latency_ms,
        scan_id=scan_id,
        shap_features=shap_features,
    )


# ===============================================================
# FEEDBACK ENDPOINTS - For Human-in-the-Loop Model Improvement
# ===============================================================

@app.post("/scan", response_model=ScanResponse)
def create_scan(scan: ScanCreate, db: Session = Depends(get_db)):
    """
    Record a scan in the database.
    Called automatically after predictions to enable feedback collection.
    """
    db_scan = Scan(
        url=scan.url,
        prediction=scan.prediction,
        confidence=scan.confidence,
        url_model_score=scan.url_model_score,
        dns_model_score=scan.dns_model_score,
        whois_model_score=scan.whois_model_score,
        explanation=scan.explanation,
        model_version=scan.model_version,
        source=scan.source,
    )
    db.add(db_scan)
    db.commit()
    db.refresh(db_scan)
    return db_scan


@app.get("/scan/{scan_id}", response_model=ScanResponse)
def get_scan(scan_id: int, db: Session = Depends(get_db)):
    """Get a specific scan by ID."""
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    return scan


@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(feedback: FeedbackCreate, db: Session = Depends(get_db)):
    """
    Submit user feedback for a scan.

    Use cases:
    - Correct a wrong prediction (set correct_label to 0 or 1)
    - Rate explanation helpfulness (set explanation_helpful to true/false)
    - Add a comment about the explanation

    This feedback is used to improve the model in the next training cycle.
    """
    # Verify scan exists
    scan = db.query(Scan).filter(Scan.id == feedback.scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    # Check if feedback already exists for this scan
    existing = db.query(Feedback).filter(Feedback.scan_id == feedback.scan_id).first()
    if existing:
        # Update existing feedback
        if feedback.correct_label is not None:
            existing.correct_label = feedback.correct_label
        if feedback.explanation_helpful is not None:
            existing.explanation_helpful = feedback.explanation_helpful
        if feedback.explanation_comment:
            existing.explanation_comment = feedback.explanation_comment
        existing.source = feedback.source
        db.commit()
        db.refresh(existing)
        return existing

    # Create new feedback
    db_feedback = Feedback(
        scan_id=feedback.scan_id,
        correct_label=feedback.correct_label,
        explanation_helpful=feedback.explanation_helpful,
        explanation_comment=feedback.explanation_comment,
        source=feedback.source,
    )
    db.add(db_feedback)
    db.commit()
    db.refresh(db_feedback)
    return db_feedback


@app.get("/feedback/corrections", response_model=List[dict])
def get_corrections(
    limit: int = 100,
    since_days: int = 30,
    db: Session = Depends(get_db)
):
    """
    Get user corrections for model retraining.

    Returns scans where the user provided a different label than the model predicted.
    Used by the training pipeline to incorporate human feedback.
    """
    from datetime import datetime, timedelta

    cutoff = datetime.utcnow() - timedelta(days=since_days)

    # Query scans with corrections
    results = (
        db.query(Scan, Feedback)
        .join(Feedback, Scan.id == Feedback.scan_id)
        .filter(Feedback.correct_label.isnot(None))
        .filter(Feedback.submitted_at >= cutoff)
        .order_by(Feedback.submitted_at.desc())
        .limit(limit)
        .all()
    )

    corrections = []
    for scan, feedback in results:
        if feedback.correct_label != scan.prediction:
            corrections.append({
                "url": scan.url,
                "model_prediction": scan.prediction,
                "correct_label": feedback.correct_label,
                "confidence": scan.confidence,
                "submitted_at": feedback.submitted_at.isoformat(),
            })

    return corrections


@app.get("/feedback/stats")
def get_feedback_stats(db: Session = Depends(get_db)):
    """
    Get feedback statistics for monitoring model performance.
    """
    from sqlalchemy import func

    total_scans = db.query(func.count(Scan.id)).scalar()
    total_feedback = db.query(func.count(Feedback.id)).scalar()

    # Corrections (where user disagreed with model)
    corrections = (
        db.query(func.count(Feedback.id))
        .join(Scan, Scan.id == Feedback.scan_id)
        .filter(Feedback.correct_label.isnot(None))
        .filter(Feedback.correct_label != Scan.prediction)
        .scalar()
    )

    # Explanation feedback
    helpful_count = (
        db.query(func.count(Feedback.id))
        .filter(Feedback.explanation_helpful == True)
        .scalar()
    )
    unhelpful_count = (
        db.query(func.count(Feedback.id))
        .filter(Feedback.explanation_helpful == False)
        .scalar()
    )

    return {
        "total_scans": total_scans,
        "total_feedback": total_feedback,
        "corrections": corrections,
        "correction_rate": corrections / total_feedback if total_feedback > 0 else 0,
        "explanation_helpful": helpful_count,
        "explanation_unhelpful": unhelpful_count,
    }


@app.on_event("startup")
def startup_init_db():
    """Initialize database tables on startup."""
    try:
        init_db()
        print("✅ Database initialized")
    except Exception as e:
        print(f"⚠️ Database initialization failed (may not be available): {e}")
