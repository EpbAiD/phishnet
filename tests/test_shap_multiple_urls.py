#!/usr/bin/env python3
"""
Test SHAP-based explanations with multiple unseen URLs.
"""

import requests
import json

def test_shap_explanation(url: str):
    """Test the SHAP explainer with a given URL."""

    print(f"\n{'='*80}")
    print(f"Testing URL: {url}")
    print(f"{'='*80}\n")

    # Call the /explain endpoint
    response = requests.post(
        "http://localhost:8000/explain",
        json={
            "url": url,
            "include_shap": True
        }
    )

    if response.status_code != 200:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return

    data = response.json()

    # Print predictions
    print("📊 PREDICTIONS:")
    print(f"   URL Model:      {data['predictions']['url_prob']:.4f}" if data['predictions']['url_prob'] else "   URL Model:      N/A")
    print(f"   WHOIS Model:    {data['predictions']['whois_prob']:.4f}" if data['predictions']['whois_prob'] else "   WHOIS Model:    N/A")
    print(f"   DNS Model:      {data['predictions']['dns_prob']}" if data['predictions']['dns_prob'] else "   DNS Model:      N/A")
    print(f"   Ensemble:       {data['predictions']['ensemble_prob']:.4f}")
    print(f"   Verdict:        {data['verdict'].upper()}")
    print(f"   User Verdict:   {data['predictions']['user_verdict'].upper()}")
    print(f"   Confidence:     {data['confidence'].upper()}")

    # Print SHAP features
    if data.get('shap_features'):
        print("\n🔍 SHAP FEATURE CONTRIBUTIONS:\n")

        # URL model features
        if data['shap_features'].get('url'):
            print("   📌 URL Model Features:")
            for i, feat in enumerate(data['shap_features']['url'], 1):
                impact = feat.get('impact', 'N/A')
                shap_val = feat.get('shap_contribution', feat.get('importance', 0))
                sign = "+" if shap_val > 0 else ""
                print(f"      {i}. {feat['plain_language']}")
                print(f"         → {impact} (SHAP: {sign}{shap_val:.4f})")

        # WHOIS model features
        if data['shap_features'].get('whois'):
            print("\n   📌 WHOIS Model Features:")
            for i, feat in enumerate(data['shap_features']['whois'], 1):
                impact = feat.get('impact', 'N/A')
                shap_val = feat.get('shap_contribution', feat.get('importance', 0))
                sign = "+" if shap_val > 0 else ""
                print(f"      {i}. {feat['plain_language']}")
                print(f"         → {impact} (SHAP: {sign}{shap_val:.4f})")
    else:
        print("\n⚠️  No SHAP features extracted")

    # Print LLM explanation
    print("\n💬 LLM EXPLANATION:")
    print(f"\n{data['explanation']}\n")

    # Print latency
    print(f"⏱️  Total Latency: {data['latency_ms']:.2f} ms")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    # Test various URLs
    test_urls = [
        # Suspicious URLs
        "http://secure-login-apple.com-verify.tk/account",
        "https://amazon-security-check.xyz/update-payment",
        "http://192.168.1.1/admin/login.php",

        # Legitimate URLs
        "https://www.github.com",
        "https://stackoverflow.com/questions",
        "https://www.wikipedia.org",
    ]

    for url in test_urls:
        test_shap_explanation(url)
