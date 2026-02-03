# ===============================================================
# unified_explainer.py
# ---------------------------------------------------------------
# Unified explanation generation for ALL prediction endpoints
# Automatically generates LLM explanations for trust-building
# Priority: Groq API -> Local LLM -> Rule-based fallback
# ===============================================================

import numpy as np
from typing import Dict, Tuple, Optional
from urllib.parse import urlparse


def generate_unified_explanation(
    url: str,
    risk_score: float,
    model_type: str,  # "url", "whois", "dns", or "ensemble"
    threshold: float,
    top_features: Optional[Dict] = None,
) -> Tuple[str, str, str]:
    """
    Generate unified explanation for any model prediction.

    Priority order:
    1. Groq API (fast, high-quality, cloud-based)
    2. Local LLM (Qwen2.5-0.5B, requires torch/transformers)
    3. Rule-based fallback (always available)

    Args:
        url: The URL being analyzed
        risk_score: Probability of phishing [0, 1]
        model_type: Type of model ("url", "whois", "dns", "ensemble")
        threshold: Decision threshold
        top_features: Optional SHAP feature contributions

    Returns:
        Tuple of (explanation, verdict, confidence)
        - explanation: Human-readable LLM-generated explanation
        - verdict: "safe" or "suspicious"
        - confidence: "high", "medium", or "low"
    """
    # Calculate confidence
    confidence_score = abs(risk_score - 0.5)
    if confidence_score > 0.3:
        confidence = "high"
    elif confidence_score > 0.15:
        confidence = "medium"
    else:
        confidence = "low"

    # Determine verdict
    is_phishing = risk_score >= threshold
    verdict = "suspicious" if is_phishing else "safe"

    # Prepare predictions dict for LLM
    predictions = {
        f"{model_type}_prob": float(risk_score),
        "ensemble_prob": float(risk_score),  # Use risk_score as ensemble
        "verdict": (
            "phishing" if is_phishing else "legit"
        ),  # LLM expects "phishing"/"legit"
        "user_verdict": verdict,  # User-facing verdict
    }

    # Extract domain from URL
    parsed = urlparse(url)
    domain = parsed.netloc or parsed.path

    # Priority 1: Try Groq API (fast, cloud-based)
    try:
        from src.api.groq_explainer import is_groq_available, generate_groq_explanation

        if is_groq_available():
            explanation = generate_groq_explanation(
                url=url, domain=domain, predictions=predictions, top_features=top_features
            )
            if explanation:
                return explanation, verdict, confidence
            print("Groq returned empty response, trying fallback...")
    except Exception as e:
        print(f"Groq explanation failed: {e}")

    # Priority 2: Try local LLM (requires torch/transformers)
    try:
        from src.api.llm_explainer import generate_explanation, TORCH_AVAILABLE

        # Only try local LLM if torch is actually available
        if TORCH_AVAILABLE:
            explanation = generate_explanation(
                url=url, domain=domain, predictions=predictions, top_features=top_features
            )
            # Check if we got a real explanation (not the "not available" message)
            if explanation and "not available" not in explanation.lower():
                return explanation, verdict, confidence

        # If torch not available or explanation failed, use fallback
        print("Local LLM not available, using rule-based fallback")

    except Exception as e:
        print(f"Local LLM explanation failed, using fallback: {e}")

    # Priority 3: Rule-based fallback (always available)
    return (
        _generate_fallback_explanation(
            url, risk_score, verdict, confidence, model_type
        ),
        verdict,
        confidence,
    )


def _generate_fallback_explanation(
    url: str, risk_score: float, verdict: str, confidence: str, model_type: str
) -> str:
    """
    Generate rule-based fallback explanation when LLM is unavailable.

    Args:
        url: The URL being analyzed
        risk_score: Probability of phishing [0, 1]
        verdict: "safe" or "suspicious"
        confidence: "high", "medium", or "low"
        model_type: Type of model used

    Returns:
        Human-readable explanation string
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    domain = parsed.netloc or parsed.path

    # Analyze URL for specific red flags
    red_flags = []
    trust_signals = []

    domain_lower = domain.lower()

    # Check for brand impersonation
    brand_names = [
        "paypal",
        "amazon",
        "google",
        "microsoft",
        "apple",
        "facebook",
        "netflix",
        "bank",
        "ebay",
    ]
    subdomain_parts = domain.split(".")[:-2] if domain.count(".") >= 2 else []
    has_brand_in_subdomain = any(
        brand in "-".join(subdomain_parts).lower() for brand in brand_names
    )

    if has_brand_in_subdomain and verdict == "suspicious":
        brand_found = next(
            (brand for brand in brand_names if brand in domain_lower), "known brand"
        )
        red_flags.append(
            f"the URL contains '{brand_found}' but isn't the official {brand_found.title()} website"
        )

    # Check TLD
    tld = domain.split(".")[-1]
    suspicious_tlds = [
        "xyz",
        "top",
        "tk",
        "ml",
        "ga",
        "cf",
        "gq",
        "work",
        "click",
        "link",
    ]
    if tld in suspicious_tlds:
        red_flags.append(f"it uses a '.{tld}' domain extension often used by scammers")
    elif tld in ["com", "org", "net", "edu", "gov"]:
        trust_signals.append(f"it uses a standard '.{tld}' domain extension")

    # Check domain length
    if len(domain) > 30:
        red_flags.append(
            f"the domain name is unusually long ({len(domain)} characters)"
        )
    elif len(domain) < 15 and verdict == "safe":
        trust_signals.append("the domain name is concise and straightforward")

    # Check hyphens
    hyphen_count = domain.count("-")
    if hyphen_count >= 2:
        red_flags.append(
            f"the domain has multiple hyphens ({hyphen_count}), which is uncommon for legitimate sites"
        )
    elif hyphen_count == 0 and verdict == "safe":
        trust_signals.append("the domain doesn't use suspicious hyphens")

    # Check for IP address
    if parsed.hostname and parsed.hostname.replace(".", "").isdigit():
        red_flags.append("it uses a raw IP address instead of a proper domain name")

    # Check for HTTPS
    if parsed.scheme == "https":
        trust_signals.append("it uses secure HTTPS encryption")
    elif parsed.scheme == "http":
        red_flags.append("it doesn't use secure HTTPS encryption")

    # Check for suspicious patterns
    suspicious_keywords = [
        "verify",
        "secure",
        "account",
        "update",
        "confirm",
        "login",
        "signin",
    ]
    if (
        any(kw in domain_lower for kw in suspicious_keywords)
        and verdict == "suspicious"
    ):
        red_flags.append(
            "the domain uses suspicious keywords often seen in phishing attempts"
        )

    # Build explanation
    if verdict == "suspicious":
        emoji = "🚨"
        status = "SUSPICIOUS"
        action = "Do not click this link or enter any personal information."

        if red_flags:
            reasons = " and ".join(red_flags[:3])
            details = f"We detected {reasons}."
        else:
            details = f"Our {model_type.upper()} model flagged several warning signs in how it's set up."

        confidence_text = f" (confidence: {confidence})" if confidence != "high" else ""

        explanation = (
            f"{emoji} This website is {status}{confidence_text}. {details} {action}"
        )

    else:  # safe
        emoji = "✅"
        status = "SAFE"
        action = "You can proceed, but always be cautious with personal information."

        if trust_signals:
            reasons = " and ".join(trust_signals[:3])
            details = f"We found {reasons}."
        else:
            details = f"Our {model_type.upper()} model found no major red flags."

        confidence_text = f" (confidence: {confidence})" if confidence != "high" else ""

        explanation = f"{emoji} This website appears {status}{confidence_text}. {details} {action}"

    return explanation


# ===============================================================
# Feature extraction wrapper for explanations
# ===============================================================


def extract_features_for_explanation(url: str, model_type: str) -> Optional[Dict]:
    """
    Extract SHAP features for explanation generation.

    Args:
        url: The URL to analyze
        model_type: Type of model ("url", "whois", "dns")

    Returns:
        Dict with top SHAP features or None if extraction fails
    """
    try:
        from src.api.shap_explainer_realtime import extract_shap_features_realtime
        from src.api.predict_utils import _features_dict_to_dataframe
        from src.api.model_loader import (
            load_url_model,
            load_whois_model,
            load_dns_model,
        )
        from src.features.url_features import extract_single_url_features
        from src.features.dns_ipwhois import extract_single_domain_features
        from src.features.whois import extract_single_whois_features
        from src.data_prep.dataset_builder import preprocess_features_for_inference
        from urllib.parse import urlparse

        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path

        if model_type == "url":
            model, cols, _ = load_url_model()
            if isinstance(model, dict) and "model" in model:
                model = model["model"]

            url_feats = extract_single_url_features(url)
            features = preprocess_features_for_inference(url_features=url_feats)
            X = _features_dict_to_dataframe(features, cols)

            top_features = extract_shap_features_realtime(model, X, "url", top_n=5)
            return {"url": top_features}

        elif model_type == "whois":
            model, cols, _ = load_whois_model()
            if isinstance(model, dict) and "model" in model:
                model = model["model"]

            whois_feats = extract_single_whois_features(domain, live_lookup=True)
            features = preprocess_features_for_inference(
                url_features={}, whois_features=whois_feats
            )
            X = _features_dict_to_dataframe(features, cols)

            top_features = extract_shap_features_realtime(model, X, "whois", top_n=5)
            return {"whois": top_features}

        elif model_type == "dns":
            model, cols, _ = load_dns_model()
            if isinstance(model, dict) and "model" in model:
                model = model["model"]

            dns_feats = extract_single_domain_features(url)
            features = preprocess_features_for_inference(
                url_features={}, dns_features=dns_feats
            )
            X = _features_dict_to_dataframe(features, cols)

            top_features = extract_shap_features_realtime(model, X, "dns", top_n=5)
            return {"dns": top_features}

        return None

    except Exception as e:
        print(f"⚠️ Feature extraction failed for {model_type}: {e}")
        return None
