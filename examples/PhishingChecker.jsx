/**
 * React component for checking URLs with Phishing Detection API
 * Example integration into a React app
 */

import React, { useState } from 'react';

const PhishingChecker = () => {
  const [url, setUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const API_BASE_URL = 'http://localhost:8000';

  const checkURL = async () => {
    if (!url.trim()) {
      setError('Please enter a URL');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/predict/ensemble`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url: url.trim() }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (riskScore) => {
    if (riskScore < 0.3) return '#4caf50'; // Green
    if (riskScore < 0.7) return '#ff9800'; // Orange
    return '#f44336'; // Red
  };

  return (
    <div style={{ maxWidth: '600px', margin: '50px auto', padding: '20px' }}>
      <h1>🔐 Phishing URL Checker</h1>

      <div style={{ marginBottom: '20px' }}>
        <input
          type="text"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && checkURL()}
          placeholder="Enter URL to check (e.g., https://example.com)"
          style={{
            width: '100%',
            padding: '12px',
            fontSize: '16px',
            border: '1px solid #ddd',
            borderRadius: '4px',
          }}
        />
      </div>

      <button
        onClick={checkURL}
        disabled={loading}
        style={{
          width: '100%',
          padding: '12px',
          fontSize: '16px',
          backgroundColor: loading ? '#ccc' : '#2196f3',
          color: 'white',
          border: 'none',
          borderRadius: '4px',
          cursor: loading ? 'not-allowed' : 'pointer',
        }}
      >
        {loading ? 'Checking...' : 'Check URL'}
      </button>

      {error && (
        <div
          style={{
            marginTop: '20px',
            padding: '15px',
            backgroundColor: '#ffebee',
            color: '#c62828',
            borderRadius: '4px',
          }}
        >
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && (
        <div
          style={{
            marginTop: '20px',
            padding: '20px',
            border: `3px solid ${getRiskColor(result.risk_score)}`,
            borderRadius: '8px',
            backgroundColor: '#f9f9f9',
          }}
        >
          <h2 style={{ marginTop: 0 }}>
            {result.is_phishing ? '⚠️ PHISHING DETECTED' : '✓ URL Appears Safe'}
          </h2>

          <div style={{ marginBottom: '15px' }}>
            <strong>URL:</strong> {result.url}
          </div>

          <div style={{ marginBottom: '15px' }}>
            <strong>Risk Score:</strong>{' '}
            <span style={{ color: getRiskColor(result.risk_score) }}>
              {(result.risk_score * 100).toFixed(1)}%
            </span>
          </div>

          <div style={{ marginBottom: '15px' }}>
            <strong>Threshold:</strong> {(result.threshold * 100).toFixed(1)}%
          </div>

          <div style={{ marginBottom: '15px' }}>
            <strong>Model:</strong> {result.model_name}
          </div>

          <div style={{ marginBottom: '15px' }}>
            <strong>Latency:</strong> {result.latency_ms.toFixed(0)}ms
          </div>

          {result.debug && (
            <details style={{ marginTop: '15px' }}>
              <summary style={{ cursor: 'pointer', fontWeight: 'bold' }}>
                📊 Model Details
              </summary>
              <div style={{ marginTop: '10px', fontSize: '14px' }}>
                <div>
                  <strong>URL Model:</strong>{' '}
                  {(result.debug.url_prediction.risk_score * 100).toFixed(1)}% (
                  {result.debug.url_prediction.latency_ms.toFixed(0)}ms, weight:{' '}
                  {result.debug.url_prediction.weight})
                </div>
                <div>
                  <strong>WHOIS Model:</strong>{' '}
                  {(result.debug.whois_prediction.risk_score * 100).toFixed(1)}%
                  ({result.debug.whois_prediction.latency_ms.toFixed(0)}ms,
                  weight: {result.debug.whois_prediction.weight})
                </div>
                <div>
                  <strong>Ensemble Formula:</strong>{' '}
                  {result.debug.ensemble.formula}
                </div>
              </div>
            </details>
          )}
        </div>
      )}

      <div
        style={{
          marginTop: '30px',
          padding: '15px',
          backgroundColor: '#e3f2fd',
          borderRadius: '4px',
          fontSize: '14px',
        }}
      >
        <strong>💡 Tip:</strong> This API uses machine learning to detect
        phishing URLs based on URL structure, WHOIS data, and domain reputation.
      </div>
    </div>
  );
};

export default PhishingChecker;
