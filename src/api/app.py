# ===============================================================
# src/api/app.py
# FastAPI app for URL-only phishing prediction
# ---------------------------------------------------------------
# - Normalizes incomplete URLs (e.g., "youtube.com" → "https://youtube.com")
# - /health : simple health check
# - /predict/url : takes URL, returns P(phishing) + label
# ===============================================================
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import urlparse

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
from src.api.model_loader import load_url_model, load_whois_model, load_dns_model
from src.api.llm_explainer import generate_explanation
import time
import numpy as np

app = FastAPI(
    title="Phishing Detection API",
    version="3.0.0",
    description="Phishing risk scoring with URL, WHOIS, and DNS models. Ensemble combines all three.",
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


@app.on_event("startup")
def _startup_event():
    """Warm model cache on server boot."""
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
    return {"status": "ok"}


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

    return ExplainResponse(
        url=url,
        explanation=layman_explanation,
        technical_explanation=technical_explanation,
        predictions=predictions,
        verdict=verdict,
        confidence=confidence,
        latency_ms=total_latency_ms,
        shap_features=shap_features,
    )
