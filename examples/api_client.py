#!/usr/bin/env python3
"""
Python client for Phishing Detection API
Shows how to integrate the API into your Python application
"""

import requests
import json
from typing import Dict, Any


class PhishingDetectionClient:
    """Client for interacting with Phishing Detection API."""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = None):
        """
        Initialize the API client.

        Args:
            base_url: Base URL of the API (default: localhost:8000)
            api_key: Optional API key for authentication (for production)
        """
        self.base_url = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}

        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def health_check(self) -> Dict[str, Any]:
        """Check if API is running."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def predict_url(self, url: str) -> Dict[str, Any]:
        """
        Predict phishing risk using URL features only.
        Fast (~10-50ms), works 100% of the time.

        Args:
            url: The URL to check

        Returns:
            dict with risk_score, is_phishing, threshold, latency_ms, etc.
        """
        payload = {"url": url}
        response = requests.post(
            f"{self.base_url}/predict/url",
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def predict_whois(self, url: str) -> Dict[str, Any]:
        """
        Predict phishing risk using WHOIS features only.
        Slow (~1-10 seconds), may fail for some domains.

        Args:
            url: The URL to check

        Returns:
            dict with risk_score, is_phishing, threshold, latency_ms, etc.
        """
        payload = {"url": url}
        response = requests.post(
            f"{self.base_url}/predict/whois",
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def predict_ensemble(self, url: str) -> Dict[str, Any]:
        """
        Predict phishing risk using ensemble of URL + WHOIS models.
        Best accuracy (~1-10 seconds due to WHOIS lookup).

        Args:
            url: The URL to check

        Returns:
            dict with risk_score, is_phishing, threshold, latency_ms, debug info
        """
        payload = {"url": url}
        response = requests.post(
            f"{self.base_url}/predict/ensemble",
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def check_url_safe(self, url: str, use_ensemble: bool = True) -> bool:
        """
        Simple helper: Returns True if URL is safe, False if phishing.

        Args:
            url: The URL to check
            use_ensemble: Use ensemble model (slower, more accurate) vs URL only

        Returns:
            True if safe, False if phishing
        """
        if use_ensemble:
            result = self.predict_ensemble(url)
        else:
            result = self.predict_url(url)

        return not result["is_phishing"]


# ===============================================================
# Example Usage
# ===============================================================

if __name__ == "__main__":
    # Initialize client
    client = PhishingDetectionClient(base_url="http://localhost:8000")

    # Example URLs to test
    legitimate_urls = [
        "https://www.google.com",
        "https://github.com/microsoft/vscode",
        "https://stackoverflow.com/questions/11227809",
    ]

    phishing_urls = [
        "https://paypal-verify-account-now.com/login",
        "https://microsoft-security-alert.com/update",
        "https://appleid-unlock.com/verify",
    ]

    print("=" * 60)
    print("Health Check")
    print("=" * 60)
    health = client.health_check()
    print(json.dumps(health, indent=2))
    print()

    print("=" * 60)
    print("Testing Legitimate URLs")
    print("=" * 60)
    for url in legitimate_urls:
        try:
            result = client.predict_url(url)
            print(f"\nURL: {url}")
            print(f"Risk Score: {result['risk_score']:.1%}")
            print(f"Is Phishing: {result['is_phishing']}")
            print(f"Latency: {result['latency_ms']:.1f}ms")
        except Exception as e:
            print(f"Error checking {url}: {e}")

    print("\n" + "=" * 60)
    print("Testing Phishing URLs")
    print("=" * 60)
    for url in phishing_urls:
        try:
            result = client.predict_url(url)
            print(f"\nURL: {url}")
            print(f"Risk Score: {result['risk_score']:.1%}")
            print(f"Is Phishing: {result['is_phishing']}")
            print(f"Latency: {result['latency_ms']:.1f}ms")
        except Exception as e:
            print(f"Error checking {url}: {e}")

    print("\n" + "=" * 60)
    print("Testing Ensemble Model (Best Accuracy)")
    print("=" * 60)
    test_url = "https://github.com/tensorflow/tensorflow"
    try:
        result = client.predict_ensemble(test_url)
        print(f"\nURL: {test_url}")
        print(f"Risk Score: {result['risk_score']:.1%}")
        print(f"Is Phishing: {result['is_phishing']}")
        print(f"Latency: {result['latency_ms']:.0f}ms")
        print(f"\nDebug Info:")
        print(f"  URL Model: {result['debug']['url_prediction']['risk_score']:.1%}")
        print(f"  WHOIS Model: {result['debug']['whois_prediction']['risk_score']:.1%}")
        print(f"  Ensemble Formula: {result['debug']['ensemble']['formula']}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("Simple Safety Check")
    print("=" * 60)
    test_urls = [
        "https://www.google.com",
        "https://paypal-verify.com/login",
    ]
    for url in test_urls:
        is_safe = client.check_url_safe(url, use_ensemble=False)
        print(f"{url}: {'✓ SAFE' if is_safe else '✗ PHISHING'}")
