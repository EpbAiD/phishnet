#!/bin/bash
# ===============================================================
# API Examples - How to call the Phishing Detection API
# ===============================================================

BASE_URL="http://localhost:8000"

echo "========================================="
echo "1. Health Check"
echo "========================================="
curl -X GET "${BASE_URL}/health"
echo -e "\n"

echo "========================================="
echo "2. URL Prediction (Legitimate)"
echo "========================================="
curl -X POST "${BASE_URL}/predict/url" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.google.com"
  }'
echo -e "\n"

echo "========================================="
echo "3. URL Prediction (Phishing)"
echo "========================================="
curl -X POST "${BASE_URL}/predict/url" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://paypal-verify-account-now.com/login"
  }'
echo -e "\n"

echo "========================================="
echo "4. WHOIS Prediction"
echo "========================================="
curl -X POST "${BASE_URL}/predict/whois" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.oracle.com"
  }'
echo -e "\n"

echo "========================================="
echo "5. Ensemble Prediction (Best accuracy)"
echo "========================================="
curl -X POST "${BASE_URL}/predict/ensemble" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://github.com/microsoft/vscode"
  }'
echo -e "\n"

echo "========================================="
echo "6. Production API URL (if deployed)"
echo "========================================="
echo "Replace localhost:8000 with your deployed URL:"
echo "curl -X POST 'https://api.yourcompany.com/predict/ensemble' \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -H 'Authorization: Bearer YOUR_API_KEY' \\"
echo "  -d '{\"url\": \"https://example.com\"}'"
