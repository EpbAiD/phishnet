#!/usr/bin/env python3
"""
Test SHAP-based explanations via the /explain API endpoint.
"""

import requests
import json

def test_shap_explanation(url: str):
    """Test the SHAP explainer with a given URL."""

    print(f"\n{'='*80}")
    print(f"Testing SHAP Explanation for: {url}")
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
    print("📊 Predictions:")
    print(f"   URL Model:      {data['predictions']['url_prob']:.4f}" if data['predictions']['url_prob'] else "   URL Model:      N/A")
    print(f"   WHOIS Model:    {data['predictions']['whois_prob']:.4f}" if data['predictions']['whois_prob'] else "   WHOIS Model:    N/A")
    print(f"   DNS Model:      {data['predictions']['dns_prob']}" if data['predictions']['dns_prob'] else "   DNS Model:      N/A")
    print(f"   Ensemble:       {data['predictions']['ensemble_prob']:.4f}")
    print(f"   Verdict:        {data['verdict']}")
    print(f"   Confidence:     {data['confidence']}")

    # Print SHAP features
    if data.get('shap_features'):
        print("\n🔍 SHAP Feature Contributions:\n")

        # URL model features
        if data['shap_features'].get('url'):
            print("   URL Model Features:")
            for feat in data['shap_features']['url']:
                impact = feat.get('impact', 'N/A')
                shap_val = feat.get('shap_contribution', feat.get('importance', 0))
                print(f"      • {feat['plain_language']}")
                print(f"        Impact: {impact} (SHAP: {shap_val:+.4f})")

        # WHOIS model features
        if data['shap_features'].get('whois'):
            print("\n   WHOIS Model Features:")
            for feat in data['shap_features']['whois']:
                impact = feat.get('impact', 'N/A')
                shap_val = feat.get('shap_contribution', feat.get('importance', 0))
                print(f"      • {feat['plain_language']}")
                print(f"        Impact: {impact} (SHAP: {shap_val:+.4f})")
    else:
        print("\n⚠️  No SHAP features extracted")

    # Print LLM explanation
    print("\n💬 LLM Explanation:")
    print(f"   {data['explanation']}")

    # Print latency
    print(f"\n⏱️  Total Latency: {data['latency_ms']:.2f} ms")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    # Test with a suspicious URL
    test_shap_explanation("http://paypal-verify.suspicious-domain.xyz/login")

    # Test with a legit URL
    test_shap_explanation("https://www.google.com")
