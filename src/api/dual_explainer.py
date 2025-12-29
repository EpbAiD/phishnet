# ===============================================================
# dual_explainer.py
# ---------------------------------------------------------------
# ✅ Dual explanation system: Layman + Technical explanations
# ✅ Layman: Simple, specific WHY/HOW explanation
# ✅ Technical: Detailed SHAP analysis for engineers
# ✅ No generic LLM - uses rule-based templates with SHAP data
# ===============================================================

from typing import Dict, List, Optional


def generate_layman_explanation(
    url: str,
    verdict: str,
    risk_score: float,
    top_features: Optional[Dict] = None
) -> str:
    """
    Generate a clear, specific explanation for non-technical users.
    Explains WHY this URL is suspicious/safe and HOW to stay safe.

    Args:
        url: The URL being analyzed
        verdict: 'phishing' or 'legit'
        risk_score: Ensemble probability (0-1)
        top_features: SHAP features from all models

    Returns:
        Layman-friendly explanation
    """
    is_suspicious = verdict == "phishing"

    # Extract key suspicious indicators
    suspicious_reasons = []
    protective_factors = []

    if top_features:
        # Collect suspicious factors (positive SHAP)
        for model_type in ["url", "whois", "dns"]:
            if model_type in top_features:
                for feat in top_features[model_type][:3]:  # Top 3 per model
                    shap_val = feat.get('shap_contribution', feat.get('importance', 0))
                    if shap_val > 0.5:  # Strong positive contribution
                        suspicious_reasons.append(feat['plain_language'])
                    elif shap_val < -0.5:  # Strong negative contribution (protective)
                        protective_factors.append(feat['plain_language'])

    # Build explanation
    if is_suspicious:
        # SUSPICIOUS URL explanation
        explanation = f"⚠️ **WARNING: This website is likely a PHISHING ATTEMPT**\n\n"
        explanation += f"**Risk Level:** {risk_score*100:.1f}% likelihood of being malicious\n\n"

        explanation += "**Why we flagged this:**\n"
        if suspicious_reasons:
            for i, reason in enumerate(suspicious_reasons[:5], 1):
                explanation += f"{i}. The URL {reason}\n"
        else:
            explanation += "• Multiple suspicious patterns detected in the URL structure\n"
            explanation += "• Domain characteristics match known phishing techniques\n"

        explanation += "\n**What this means:**\n"
        explanation += "This website is trying to impersonate a legitimate service to steal your personal information, "
        explanation += "passwords, credit card details, or other sensitive data.\n\n"

        explanation += "**How to stay safe:**\n"
        explanation += "• ❌ DO NOT enter any personal information\n"
        explanation += "• ❌ DO NOT click any links on this page\n"
        explanation += "• ❌ DO NOT download anything from this site\n"
        explanation += "• ✅ Close this page immediately\n"
        explanation += "• ✅ If you received this link via email/message, report it as phishing\n"
        explanation += "• ✅ Check your accounts for suspicious activity if you already entered information\n"

    else:
        # SAFE URL explanation
        explanation = f"✅ **This website appears to be LEGITIMATE**\n\n"
        explanation += f"**Confidence Level:** {(1-risk_score)*100:.1f}% confidence this is safe\n\n"

        explanation += "**Why we trust this site:**\n"
        if protective_factors:
            for i, factor in enumerate(protective_factors[:5], 1):
                explanation += f"{i}. The URL {factor}\n"
        else:
            explanation += "• The URL structure follows normal patterns\n"
            explanation += "• The domain has a clean reputation\n"
            explanation += "• No suspicious characteristics detected\n"

        explanation += "\n**What this means:**\n"
        explanation += "Our analysis indicates this is likely a legitimate website. However, you should still "
        explanation += "exercise caution when sharing personal information online.\n\n"

        explanation += "**Best practices:**\n"
        explanation += "• ✅ Verify you're on the correct website (check the URL carefully)\n"
        explanation += "• ✅ Look for HTTPS and a padlock icon in your browser\n"
        explanation += "• ✅ Be cautious with sensitive information even on trusted sites\n"
        explanation += "• ✅ Use strong, unique passwords for each account\n"
        explanation += "• ✅ Enable two-factor authentication when available\n"

    return explanation


def generate_technical_explanation(
    url: str,
    predictions: Dict,
    top_features: Optional[Dict] = None
) -> str:
    """
    Generate detailed technical analysis for engineers and security analysts.
    Includes SHAP values, feature contributions, and model-level insights.

    Args:
        url: The URL being analyzed
        predictions: Model predictions dict
        top_features: SHAP features from all models

    Returns:
        Technical analysis report
    """
    explanation = "## Technical Analysis Report\n\n"

    # Model Predictions
    explanation += "### Model Predictions\n"
    explanation += f"- **URL Model:** {predictions.get('url_prob', 0)*100:.2f}% phishing probability\n"
    explanation += f"- **WHOIS Model:** {predictions.get('whois_prob', 0)*100:.2f}% phishing probability\n"
    if predictions.get('dns_prob') is not None:
        explanation += f"- **DNS Model:** {predictions['dns_prob']*100:.2f}% phishing probability\n"
    explanation += f"- **Ensemble:** {predictions['ensemble_prob']*100:.2f}% phishing probability\n"
    explanation += f"- **Final Verdict:** {predictions['verdict'].upper()}\n\n"

    # SHAP Feature Analysis
    if top_features:
        explanation += "### SHAP Feature Contributions\n\n"
        explanation += "Features are ranked by their contribution to the phishing score. "
        explanation += "Positive values increase phishing risk, negative values indicate legitimacy.\n\n"

        for model_type in ["url", "whois", "dns"]:
            if model_type not in top_features or not top_features[model_type]:
                continue

            explanation += f"#### {model_type.upper()} Model Features:\n\n"
            explanation += "| Rank | Feature | SHAP Value | Impact |\n"
            explanation += "|------|---------|------------|--------|\n"

            for i, feat in enumerate(top_features[model_type], 1):
                feature_name = feat.get('feature', 'unknown')
                shap_val = feat.get('shap_contribution', feat.get('importance', 0))
                impact = feat.get('impact', 'N/A')
                sign = "+" if shap_val > 0 else ""

                explanation += f"| {i} | `{feature_name}` | {sign}{shap_val:.4f} | {impact} |\n"

            explanation += "\n"

    # Interpretation Guide
    explanation += "### Interpretation Guide\n\n"
    explanation += "**SHAP Values:**\n"
    explanation += "- Values > +1.0: Strong indicators of phishing\n"
    explanation += "- Values 0.0 to +1.0: Moderate phishing signals\n"
    explanation += "- Values -1.0 to 0.0: Moderate legitimacy signals\n"
    explanation += "- Values < -1.0: Strong indicators of legitimacy\n\n"

    explanation += "**Model Ensemble:**\n"
    explanation += "- URL Model Weight: 60% (structural analysis)\n"
    explanation += "- WHOIS Model Weight: 40% (domain reputation)\n"
    explanation += "- DNS Model: Currently disabled (VM API integration pending)\n\n"

    # Risk Assessment
    risk_score = predictions['ensemble_prob']
    if risk_score >= 0.8:
        risk_level = "CRITICAL"
        recommendation = "Block immediately. High confidence phishing attempt."
    elif risk_score >= 0.6:
        risk_level = "HIGH"
        recommendation = "Flag and warn users. Likely malicious."
    elif risk_score >= 0.4:
        risk_level = "MEDIUM"
        recommendation = "Investigate further. Uncertain classification."
    elif risk_score >= 0.2:
        risk_level = "LOW"
        recommendation = "Likely legitimate, but monitor."
    else:
        risk_level = "MINIMAL"
        recommendation = "Legitimate with high confidence."

    explanation += f"### Risk Assessment\n\n"
    explanation += f"- **Risk Level:** {risk_level}\n"
    explanation += f"- **Confidence:** {abs(risk_score - 0.5) * 200:.1f}%\n"
    explanation += f"- **Recommendation:** {recommendation}\n"

    return explanation


def generate_dual_explanations(
    url: str,
    domain: str,
    predictions: Dict,
    top_features: Optional[Dict] = None
) -> Dict[str, str]:
    """
    Generate both layman and technical explanations.

    Returns:
        Dict with 'layman' and 'technical' keys
    """
    verdict = predictions.get('verdict', 'legit')
    risk_score = predictions.get('ensemble_prob', 0.5)

    return {
        "layman": generate_layman_explanation(url, verdict, risk_score, top_features),
        "technical": generate_technical_explanation(url, predictions, top_features)
    }
