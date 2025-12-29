# ===============================================================
# llm_explainer.py
# ---------------------------------------------------------------
# ✅ LLM-based explanation generator for phishing detection
# ✅ Uses Qwen2.5-0.5B-Instruct with prompt engineering (no fine-tuning)
# ✅ Generates human-readable explanations for predictions
# ✅ Integrates with FastAPI endpoints
# ✅ Supports lazy loading for fast API startup
# ===============================================================

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Optional
import numpy as np

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
# Using base Qwen2.5-0.5B with prompt engineering (no fine-tuning needed!)
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Generation parameters
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
DO_SAMPLE = True

# ---------------------------------------------------------------
# Global model and tokenizer (lazy loaded)
# ---------------------------------------------------------------
_model = None
_tokenizer = None


def load_explainer_model():
    """
    Load the LLM model and tokenizer.
    Uses Qwen2.5-0.5B-Instruct with prompt engineering (no fine-tuning).
    This is called lazily on first use to avoid slowing API startup.
    """
    global _model, _tokenizer

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    print("⚙️ Loading LLM explainer model...")

    # Load tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
        _tokenizer.pad_token_id = _tokenizer.eos_token_id

    # Load base model with prompt engineering (no adapters)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    print("✅ Qwen2.5-0.5B loaded successfully (using prompt engineering)")

    # Set to eval mode
    _model.eval()

    return _model, _tokenizer


def generate_explanation(
    url: str,
    domain: str,
    predictions: Dict,
    top_features: Optional[Dict] = None,
    ground_truth: Optional[str] = None,
) -> str:
    """
    Generate a human-readable explanation for a phishing detection result.

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

    Returns:
        Human-readable explanation string
    """
    try:
        model, tokenizer = load_explainer_model()

        # Prepare input data
        input_data = {
            "url": url,
            "domain": domain,
            "predictions": predictions,
        }

        if top_features:
            input_data["top_features"] = top_features

        if ground_truth:
            input_data["ground_truth"] = ground_truth

        # Enhanced prompt engineering for cybersecurity analysis
        system_message = (
            "You are a cybersecurity expert helping everyday people understand website safety. "
            "Your job:\n"
            "1. Explain if a website is SAFE or SUSPICIOUS in clear, simple language\n"
            "2. Mention specific red flags or trust signals you found (be concrete!)\n"
            "3. Give actionable advice: Should they click it? What should they avoid?\n"
            "4. NO technical terms (DNS, WHOIS, probability, ensemble, model, algorithm)\n"
            "5. Write 2-4 natural sentences - conversational and reassuring\n"
            "6. Focus on WHY it's safe or dangerous, not just WHAT features it has"
        )

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

        # Extract evidence from Feature Importance + Feature Values (from ALL THREE models)
        red_flags = []
        trust_signals = []

        if top_features:
            # Process features from all three models
            for model_type in ["url", "dns", "whois"]:
                if model_type in top_features and top_features[model_type]:
                    for feat in top_features[model_type][:3]:  # Top 3 from each model
                        plain_text = feat.get("plain_language", "")

                        # Skip if no meaningful text
                        if not plain_text or "none detected" in plain_text.lower():
                            continue

                        # Determine if this is a red flag or trust signal based on predictions
                        # If verdict is phishing and this is an important feature, it's likely a red flag
                        # If verdict is legit and this is an important feature, it's likely a trust signal

                        # For now, classify based on common patterns
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
                            # Default: if phishing verdict, treat as red flag, else trust signal
                            if verdict == "phishing":
                                red_flags.append(plain_text)
                            else:
                                trust_signals.append(plain_text)

        # Fallback to basic URL analysis if no SHAP features available
        if not red_flags and not trust_signals:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            domain_lower = domain.lower()

            # Basic URL-only checks as fallback
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
                    brand for brand in brand_names if brand in domain_lower
                )
                red_flags.append(
                    f"the URL contains '{brand_found}' but isn't the official {brand_found.title()} website"
                )

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
                red_flags.append(
                    f"it uses a '.{tld}' domain extension often used by scammers"
                )

            if len(domain) > 30:
                red_flags.append(
                    f"the domain name is unusually long ({len(domain)} characters)"
                )

            hyphen_count = domain.count("-")
            if hyphen_count >= 2:
                red_flags.append(
                    f"the domain has multiple hyphens ({hyphen_count}), which is uncommon for legitimate sites"
                )

            if parsed.hostname and parsed.hostname.replace(".", "").isdigit():
                red_flags.append(
                    "it uses a raw IP address instead of a proper domain name"
                )

            if not red_flags and url_prob and url_prob < 0.3:
                trust_signals.append(
                    "the website address follows normal naming patterns"
                )

            if url_prob and url_prob > 0.5 and not red_flags:
                red_flags.append("the URL structure has suspicious characteristics")

            if whois_prob and whois_prob < 0.3:
                trust_signals.append("the domain has been registered for a while")
            elif whois_prob and whois_prob > 0.7:
                red_flags.append("the domain was registered very recently")

        # Create specific explanation based on verdict - use SAFE/SUSPICIOUS instead of legit/phishing
        user_verdict = "SUSPICIOUS" if verdict == "phishing" else "SAFE"

        if verdict == "phishing":
            reasons = (
                " and ".join(red_flags[:3])
                if red_flags
                else "several warning signs in how it's set up"
            )
            example_start = f"🚨 This website is SUSPICIOUS. We detected {reasons}. Do not click this link or enter any personal information."
        else:
            reasons = (
                " and ".join(trust_signals[:3])
                if trust_signals
                else "no major red flags"
            )
            example_start = f"✅ This website appears SAFE. We found {reasons}. You can proceed, but always be cautious with personal information."

        prompt = f"""<|im_start|>system
{system_message}

Example SAFE response: "This website is SAFE to visit. The domain has been registered for many years, uses secure HTTPS encryption, and has a clean reputation. You can proceed confidently, but never share passwords via email or suspicious forms."

Example SUSPICIOUS response: "This website is SUSPICIOUS and likely a scam. The web address imitates PayPal but uses a fake domain that was registered just 2 days ago. Do not click this link or enter any credit card information - it's designed to steal your data."

Example SAFE response: "This website looks SAFE. It's an established domain with proper security certificates and normal website structure. Feel free to browse, but remember to always verify you're on the correct site before logging in."<|im_end|>
<|im_start|>user
Is this website safe? {url}<|im_end|>
<|im_start|>assistant
{example_start}"""

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=DO_SAMPLE,
                top_p=TOP_P,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode with special tokens first to find the assistant's response
        response_with_tokens = tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Extract only the assistant's response
        if "<|im_start|>assistant" in response_with_tokens:
            response = response_with_tokens.split("<|im_start|>assistant")[-1].strip()
            if "<|im_end|>" in response:
                response = response.split("<|im_end|>")[0].strip()
        else:
            # Fallback: decode without special tokens and try to extract
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the input prompt (everything before "This URL has been classified")
            if "This URL has been classified" in response:
                response = (
                    "This URL has been classified"
                    + response.split("This URL has been classified")[-1]
                )

        return response

    except Exception as e:
        print(f"❌ LLM explanation generation failed: {e}")
        # Fallback to simple explanation
        return generate_fallback_explanation(url, predictions)


def generate_fallback_explanation(url: str, predictions: Dict) -> str:
    """
    Generate a simple rule-based explanation as fallback.
    Used when LLM is not available or fails.
    """
    verdict = predictions.get("verdict", "unknown")
    ensemble_prob = predictions.get("ensemble_prob", 0.0)

    confidence = (
        "high"
        if abs(ensemble_prob - 0.5) > 0.3
        else "medium" if abs(ensemble_prob - 0.5) > 0.15 else "low"
    )

    explanation = f"Analysis of URL: {url}\n\n"
    explanation += f"Verdict: {verdict.upper()} (confidence: {confidence}, score: {ensemble_prob:.3f})\n\n"

    explanation += "Model Predictions:\n"
    for model_type in ["url", "dns", "whois"]:
        prob = predictions.get(f"{model_type}_prob")
        if prob is not None and not (isinstance(prob, float) and np.isnan(prob)):
            explanation += f"- {model_type.upper()} model: {prob:.3f} ({'phishing' if prob > 0.5 else 'legitimate'})\n"

    explanation += "\nConclusion: "
    if confidence == "high":
        explanation += f"The models show strong agreement that this URL is {verdict}. "
    elif confidence == "low":
        explanation += f"The models show uncertainty in classifying this URL. "

    explanation += f"The ensemble score of {ensemble_prob:.3f} indicates a {verdict} classification."

    return explanation


def unload_explainer_model():
    """
    Unload the LLM model to free memory.
    Useful for testing or when model is not needed.
    """
    global _model, _tokenizer

    if _model is not None:
        del _model
        _model = None

    if _tokenizer is not None:
        del _tokenizer
        _tokenizer = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("✅ LLM explainer model unloaded")


# ---------------------------------------------------------------
# Testing
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("🧪 Testing LLM Explainer\n")

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
            {"feature": "suspicious_tld", "value": 1.0, "shap_contribution": 0.32},
            {"feature": "brand_in_subdomain", "value": 1.0, "shap_contribution": 0.28},
        ],
        "dns": [{"feature": "short_ttl", "value": 300.0, "shap_contribution": 0.15}],
        "whois": [
            {"feature": "domain_age_days", "value": 5.0, "shap_contribution": 0.25}
        ],
    }

    print("Generating explanation...\n")
    explanation = generate_explanation(
        test_url, test_domain, test_predictions, test_features, ground_truth="phishing"
    )

    print("=" * 80)
    print("EXPLANATION:")
    print("=" * 80)
    print(explanation)
    print("=" * 80)
