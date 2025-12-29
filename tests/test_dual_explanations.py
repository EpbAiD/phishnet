#!/usr/bin/env python3
"""
Test dual explanation system (Layman + Technical).
"""

import requests
import json

def test_dual_explanations(url: str):
    """Test dual explanations with a given URL."""

    print(f"\n{'='*100}")
    print(f"Testing URL: {url}")
    print(f"{'='*100}\n")

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

    # Print basic info
    print(f"🎯 **VERDICT:** {data['verdict'].upper()} (Confidence: {data['confidence'].upper()})")
    print(f"📊 **Risk Score:** {data['predictions']['ensemble_prob']*100:.1f}%\n")

    # Print LAYMAN EXPLANATION
    print("=" * 100)
    print("👤 LAYMAN EXPLANATION (For End Users)")
    print("=" * 100)
    print(data['explanation'])
    print()

    # Print TECHNICAL EXPLANATION
    print("=" * 100)
    print("🔧 TECHNICAL EXPLANATION (For Engineers/Security Analysts)")
    print("=" * 100)
    if data.get('technical_explanation'):
        print(data['technical_explanation'])
    else:
        print("No technical explanation available")
    print()

    print(f"⏱️  **Latency:** {data['latency_ms']:.2f} ms")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    # Test with suspicious URL
    test_dual_explanations("http://secure-login-apple.com-verify.tk/account")

    # Test with legit URL
    test_dual_explanations("https://www.github.com")

    # Test with IP address (very suspicious)
    test_dual_explanations("http://192.168.1.1/admin/login.php")
