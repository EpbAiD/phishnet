# ===============================================================
# groq_explainer.py
# ---------------------------------------------------------------
# Groq API-based explanation generator for phishing detection
# Uses llama-3.3-70b-versatile or mixtral-8x7b-32768 models
# Falls back gracefully if GROQ_API_KEY is not set
# Integrates with unified_explainer.py flow
# ===============================================================

import os
from typing import Dict, Optional
from urllib.parse import urlparse

# Check if groq library is installed
try:
    from groq import Groq

    GROQ_LIBRARY_AVAILABLE = True
except ImportError:
    Groq = None
    GROQ_LIBRARY_AVAILABLE = False

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
# Primary model: llama-3.3-70b-versatile (best quality)
# Fallback model: mixtral-8x7b-32768 (faster, still good)
PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "mixtral-8x7b-32768"

# Generation parameters
MAX_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9

# ---------------------------------------------------------------
# Global client (lazy loaded)
# ---------------------------------------------------------------
_client = None


def get_groq_client():
    """
    Get or create the Groq client.
    Returns None if GROQ_API_KEY is not set.
    Checks environment variable at runtime (not import time) for hot reload support.
    """
    global _client

    if not GROQ_LIBRARY_AVAILABLE:
        return None

    # Check API key at runtime (allows hot reload of env vars)
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None

    if _client is None:
        _client = Groq(api_key=api_key)

    return _client


def is_groq_available() -> bool:
    """
    Check if Groq API is available and configured.
    Checks at runtime for hot reload support.
    """
    if not GROQ_LIBRARY_AVAILABLE:
        return False
    api_key = os.environ.get("GROQ_API_KEY")
    return api_key is not None and len(api_key) > 0


def generate_groq_explanation(
    url: str,
    domain: str,
    predictions: Dict,
    top_features: Optional[Dict] = None,
    ground_truth: Optional[str] = None,
    model: str = None,
) -> Optional[str]:
    """
    Generate a human-readable explanation using Groq API.

    Args:
        url: The URL being analyzed
        domain: The domain extracted from the URL
        predictions: Dict containing model predictions:
            - url_prob: URL model probability
            - dns_prob: DNS model probability
            - whois_prob: WHOIS model probability
            - ensemble_prob: Ensemble probability
            - verdict: Final verdict ('phishing' or 'legit')
        top_features: Optional dict containing top features for each model
            Format: {
                "url": [{"feature": "...", "plain_language": "...", "shap_contribution": ...}, ...],
                "dns": [...],
                "whois": [...]
            }
        ground_truth: Optional ground truth label for evaluation
        model: Optional model override (defaults to PRIMARY_MODEL)

    Returns:
        Human-readable explanation string, or None if Groq is unavailable
    """
    if not is_groq_available():
        return None

    client = get_groq_client()
    if client is None:
        return None

    try:
        # Extract key info for cleaner prompt
        verdict = predictions.get("verdict", "unknown")
        ensemble_prob = predictions.get("ensemble_prob", 0.0)
        url_prob = predictions.get("url_prob")
        dns_prob = predictions.get("dns_prob")
        whois_prob = predictions.get("whois_prob")

        # Determine confidence
        confidence = (
            "HIGH"
            if abs(ensemble_prob - 0.5) > 0.3
            else "MEDIUM" if abs(ensemble_prob - 0.5) > 0.15 else "LOW"
        )

        # Extract evidence from Feature Importance + Feature Values
        red_flags = []
        trust_signals = []

        if top_features:
            for model_type in ["url", "dns", "whois"]:
                if model_type in top_features and top_features[model_type]:
                    for feat in top_features[model_type][:3]:
                        plain_text = feat.get("plain_language", "")

                        if not plain_text or "none detected" in plain_text.lower():
                            continue

                        # Classify based on common patterns
                        is_negative = any(
                            keyword in plain_text.lower()
                            for keyword in [
                                "suspicious",
                                "recently registered",
                                "short",
                                "random",
                                "trick",
                                "multiple",
                                "unusual",
                                "brand name",
                                "imperson",
                                "scam",
                                "ip address",
                                "privacy protection",
                                "abuse",
                                "free registration",
                                "many",
                            ]
                        )

                        is_positive = any(
                            keyword in plain_text.lower()
                            for keyword in [
                                "https",
                                "several years",
                                "normal",
                                "standard",
                                "legitimate",
                                "does not",
                                "low risk",
                            ]
                        )

                        if is_negative:
                            red_flags.append(plain_text)
                        elif is_positive:
                            trust_signals.append(plain_text)
                        else:
                            if verdict == "phishing":
                                red_flags.append(plain_text)
                            else:
                                trust_signals.append(plain_text)

        # Fallback to basic URL analysis if no SHAP features
        if not red_flags and not trust_signals:
            red_flags, trust_signals = _extract_basic_url_features(
                url, domain, verdict, url_prob, whois_prob
            )

        # Build prompt
        system_message = """You are a cybersecurity expert helping everyday people understand website safety.

Your job:
1. Explain if a website is SAFE or SUSPICIOUS in clear, simple language
2. Mention specific red flags or trust signals you found (be concrete!)
3. Give actionable advice: Should they click it? What should they avoid?
4. NO technical terms (DNS, WHOIS, probability, ensemble, model, algorithm)
5. Write 2-4 natural sentences - conversational and reassuring
6. Focus on WHY it's safe or dangerous, not just WHAT features it has

IMPORTANT: Start your response with the appropriate emoji:
- Use a warning emoji for SUSPICIOUS websites
- Use a checkmark emoji for SAFE websites"""

        # Prepare evidence summary
        if verdict == "phishing":
            evidence = f"Red flags detected: {', '.join(red_flags[:3]) if red_flags else 'suspicious patterns in URL structure'}"
            verdict_label = "SUSPICIOUS"
        else:
            evidence = f"Trust signals: {', '.join(trust_signals[:3]) if trust_signals else 'no major red flags detected'}"
            verdict_label = "SAFE"

        user_message = f"""Analyze this website and explain if it's safe:

URL: {url}
Domain: {domain}
Analysis Result: {verdict_label} (confidence: {confidence})
Risk Score: {ensemble_prob:.1%}
{evidence}

Provide a brief, clear explanation for a non-technical user."""

        # Try primary model first, fall back to secondary
        model_to_use = model or PRIMARY_MODEL

        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                model=model_to_use,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
        except Exception as e:
            # Try fallback model
            if model_to_use == PRIMARY_MODEL:
                print(f"Primary model failed, trying fallback: {e}")
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message},
                    ],
                    model=FALLBACK_MODEL,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                )
            else:
                raise

        response = chat_completion.choices[0].message.content.strip()
        return response

    except Exception as e:
        print(f"Groq explanation generation failed: {e}")
        return None


def _extract_basic_url_features(
    url: str,
    domain: str,
    verdict: str,
    url_prob: Optional[float],
    whois_prob: Optional[float],
) -> tuple:
    """
    Extract basic URL features as fallback when SHAP features are unavailable.

    Returns:
        Tuple of (red_flags, trust_signals)
    """
    red_flags = []
    trust_signals = []

    parsed = urlparse(url)
    domain_lower = domain.lower()

    # Brand impersonation check
    brand_names = [
        "paypal",
        "amazon",
        "google",
        "microsoft",
        "apple",
        "facebook",
        "netflix",
        "bank",
    ]
    subdomain_parts = domain.split(".")[:-2] if domain.count(".") >= 2 else []
    has_brand_in_subdomain = any(
        brand in "-".join(subdomain_parts).lower() for brand in brand_names
    )

    if has_brand_in_subdomain and url_prob and url_prob > 0.5:
        brand_found = next(
            (brand for brand in brand_names if brand in domain_lower), "brand"
        )
        red_flags.append(
            f"the URL contains '{brand_found}' but isn't the official {brand_found.title()} website"
        )

    # TLD check
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

    # Domain length check
    if len(domain) > 30:
        red_flags.append(
            f"the domain name is unusually long ({len(domain)} characters)"
        )

    # Hyphen check
    hyphen_count = domain.count("-")
    if hyphen_count >= 2:
        red_flags.append(
            f"the domain has multiple hyphens ({hyphen_count}), which is uncommon for legitimate sites"
        )

    # IP address check
    if parsed.hostname and parsed.hostname.replace(".", "").isdigit():
        red_flags.append("it uses a raw IP address instead of a proper domain name")

    # Positive signals
    if not red_flags and url_prob and url_prob < 0.3:
        trust_signals.append("the website address follows normal naming patterns")

    if url_prob and url_prob > 0.5 and not red_flags:
        red_flags.append("the URL structure has suspicious characteristics")

    if whois_prob and whois_prob < 0.3:
        trust_signals.append("the domain has been registered for a while")
    elif whois_prob and whois_prob > 0.7:
        red_flags.append("the domain was registered very recently")

    return red_flags, trust_signals


def generate_groq_dual_explanations(
    url: str,
    domain: str,
    predictions: Dict,
    top_features: Optional[Dict] = None,
) -> Optional[Dict[str, str]]:
    """
    Generate both layman and technical explanations using Groq.

    Returns:
        Dict with 'layman' and 'technical' keys, or None if Groq unavailable
    """
    if not is_groq_available():
        return None

    client = get_groq_client()
    if client is None:
        return None

    try:
        verdict = predictions.get("verdict", "legit")
        ensemble_prob = predictions.get("ensemble_prob", 0.5)

        # Generate layman explanation
        layman = generate_groq_explanation(
            url=url,
            domain=domain,
            predictions=predictions,
            top_features=top_features,
        )

        if layman is None:
            return None

        # Generate technical explanation with different prompt
        technical_prompt = f"""You are a security analyst writing a technical report.

Analyze this URL and provide a technical assessment:

URL: {url}
Domain: {domain}
Model Predictions:
- URL Model: {predictions.get('url_prob', 'N/A')}
- WHOIS Model: {predictions.get('whois_prob', 'N/A')}
- DNS Model: {predictions.get('dns_prob', 'N/A')}
- Ensemble Score: {ensemble_prob:.4f}
- Verdict: {verdict.upper()}

Top Features: {top_features if top_features else 'Not available'}

Write a brief technical summary (3-5 sentences) covering:
1. Key risk indicators or trust signals
2. Model agreement/disagreement
3. Confidence assessment
4. Recommended action"""

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a cybersecurity analyst providing technical assessments.",
                },
                {"role": "user", "content": technical_prompt},
            ],
            model=PRIMARY_MODEL,
            max_tokens=MAX_TOKENS,
            temperature=0.5,  # Lower temperature for technical content
            top_p=TOP_P,
        )

        technical = chat_completion.choices[0].message.content.strip()

        return {
            "layman": layman,
            "technical": technical,
        }

    except Exception as e:
        print(f"Groq dual explanation generation failed: {e}")
        return None


# ---------------------------------------------------------------
# Testing
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("Testing Groq Explainer\n")

    if not is_groq_available():
        print("Groq API is not available. Set GROQ_API_KEY environment variable.")
        print("Example: export GROQ_API_KEY='your-api-key'")
        exit(1)

    # Test input
    test_url = "http://paypal-verify.suspicious-domain.xyz/login"
    test_domain = "paypal-verify.suspicious-domain.xyz"
    test_predictions = {
        "url_prob": 0.92,
        "dns_prob": 0.85,
        "whois_prob": 0.78,
        "ensemble_prob": 0.85,
        "verdict": "phishing",
    }

    test_features = {
        "url": [
            {
                "feature": "suspicious_tld",
                "plain_language": "uses a suspicious domain extension (.xyz)",
                "shap_contribution": 0.32,
            },
            {
                "feature": "brand_in_subdomain",
                "plain_language": "contains 'paypal' in a suspicious way",
                "shap_contribution": 0.28,
            },
        ],
        "dns": [
            {
                "feature": "short_ttl",
                "plain_language": "DNS records change frequently",
                "shap_contribution": 0.15,
            }
        ],
        "whois": [
            {
                "feature": "domain_age_days",
                "plain_language": "domain was recently registered (5 days ago)",
                "shap_contribution": 0.25,
            }
        ],
    }

    print("Generating explanation...\n")
    explanation = generate_groq_explanation(
        test_url, test_domain, test_predictions, test_features, ground_truth="phishing"
    )

    print("=" * 80)
    print("EXPLANATION:")
    print("=" * 80)
    print(explanation)
    print("=" * 80)

    print("\nGenerating dual explanations...\n")
    dual = generate_groq_dual_explanations(
        test_url, test_domain, test_predictions, test_features
    )

    if dual:
        print("=" * 80)
        print("LAYMAN EXPLANATION:")
        print("=" * 80)
        print(dual["layman"])
        print("=" * 80)
        print("\nTECHNICAL EXPLANATION:")
        print("=" * 80)
        print(dual["technical"])
        print("=" * 80)
