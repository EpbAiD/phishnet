import React, { useState } from 'react';
import { predictUrl, explainUrl, submitFeedback } from '../api';
import { PredictionResponse, ExplainResponse, AnalysisMode, FeedbackState } from '../types';
import './Scanner.css';

function Scanner() {
  const [url, setUrl] = useState('');
  const [mode, setMode] = useState<AnalysisMode>('url');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | ExplainResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [feedbackState, setFeedbackState] = useState<FeedbackState>({
    correctLabel: null,
    explanationHelpful: null
  });
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);
  const [showDebug, setShowDebug] = useState(false);

  const handleAnalyze = async () => {
    if (!url.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);
    setFeedbackSubmitted(false);
    setFeedbackState({ correctLabel: null, explanationHelpful: null });

    try {
      if (mode === 'url') {
        const data = await predictUrl(url);
        setResult(data);
      } else {
        const data = await explainUrl(url);
        setResult(data);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setLoading(false);
    }
  };

  const handleFeedback = async () => {
    if (!result?.scan_id) return;

    try {
      await submitFeedback({
        scan_id: result.scan_id,
        correct_label: feedbackState.correctLabel,
        explanation_helpful: feedbackState.explanationHelpful,
        source: 'web'
      });
      setFeedbackSubmitted(true);
    } catch (err) {
      setError('Failed to submit feedback');
    }
  };

  const isPhishing = result ? ('is_phishing' in result ? result.is_phishing : result.verdict === 'phishing') : false;
  const riskScore = result ? ('risk_score' in result ? result.risk_score : result.predictions.ensemble_prob) : 0;

  return (
    <div className="scanner">
      <div className="input-section">
        <div className="mode-selector">
          <button
            className={`mode-btn ${mode === 'url' ? 'active' : ''}`}
            onClick={() => setMode('url')}
          >
            Quick Scan
          </button>
          <button
            className={`mode-btn ${mode === 'explain' ? 'active' : ''}`}
            onClick={() => setMode('explain')}
          >
            Full Analysis
          </button>
        </div>

        <div className="url-input">
          <input
            type="text"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="Enter URL to analyze..."
            onKeyPress={(e) => e.key === 'Enter' && handleAnalyze()}
          />
          <button onClick={handleAnalyze} disabled={loading || !url.trim()}>
            {loading ? 'Analyzing...' : 'Analyze'}
          </button>
        </div>

        <div className="example-urls">
          <span>Try:</span>
          <button onClick={() => setUrl('https://google.com')}>google.com</button>
          <button onClick={() => setUrl('http://secure-login-update.xyz/verify')}>Phishing Example</button>
        </div>
      </div>

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      {result && (
        <div className={`result ${isPhishing ? 'phishing' : 'safe'}`}>
          <div className="result-header">
            <span className="result-icon">{isPhishing ? '🚨' : '✅'}</span>
            <span className="result-verdict">{isPhishing ? 'PHISHING DETECTED' : 'SAFE'}</span>
          </div>

          <div className="result-score">
            <div className="score-bar">
              <div
                className="score-fill"
                style={{ width: `${riskScore * 100}%` }}
              />
            </div>
            <span className="score-text">Risk: {(riskScore * 100).toFixed(1)}%</span>
          </div>

          {'explanation' in result && result.explanation && (
            <div className="explanation">
              <h4>Analysis:</h4>
              <p>{result.explanation}</p>
            </div>
          )}

          <div className="result-details">
            <div className="detail">
              <span className="label">Model:</span>
              <span className="value">{'model_name' in result ? result.model_name : 'Ensemble'}</span>
            </div>
            <div className="detail">
              <span className="label">Latency:</span>
              <span className="value">{result.latency_ms.toFixed(0)}ms</span>
            </div>
            <div className="detail">
              <span className="label">Confidence:</span>
              <span className="value">{result.confidence}</span>
            </div>
          </div>

          {/* Feedback Section */}
          {result.scan_id && !feedbackSubmitted && (
            <div className="feedback-section">
              <h4>Help Improve Our Model</h4>

              <div className="feedback-row">
                <span className="feedback-label">Was this prediction correct?</span>
                <div className="feedback-buttons">
                  <button
                    className={`feedback-btn correct ${feedbackState.correctLabel === null ? 'selected' : ''}`}
                    onClick={() => setFeedbackState(prev => ({ ...prev, correctLabel: null }))}
                  >
                    Yes, it's {isPhishing ? 'Phishing' : 'Safe'}
                  </button>
                  <button
                    className={`feedback-btn incorrect ${feedbackState.correctLabel !== null ? 'selected' : ''}`}
                    onClick={() => setFeedbackState(prev => ({ ...prev, correctLabel: isPhishing ? 0 : 1 }))}
                  >
                    No, it's {isPhishing ? 'Safe' : 'Phishing'}
                  </button>
                </div>
              </div>

              <div className="feedback-row">
                <span className="feedback-label">Was the explanation helpful?</span>
                <div className="feedback-buttons">
                  <button
                    className={`feedback-btn ${feedbackState.explanationHelpful === true ? 'selected' : ''}`}
                    onClick={() => setFeedbackState(prev => ({ ...prev, explanationHelpful: true }))}
                  >
                    👍 Helpful
                  </button>
                  <button
                    className={`feedback-btn ${feedbackState.explanationHelpful === false ? 'selected' : ''}`}
                    onClick={() => setFeedbackState(prev => ({ ...prev, explanationHelpful: false }))}
                  >
                    👎 Not Helpful
                  </button>
                </div>
              </div>

              <button className="submit-feedback" onClick={handleFeedback}>
                Submit Feedback
              </button>
            </div>
          )}

          {feedbackSubmitted && (
            <div className="feedback-success">
              Thank you for your feedback!
            </div>
          )}

          {/* Debug Toggle */}
          <button className="debug-toggle" onClick={() => setShowDebug(!showDebug)}>
            {showDebug ? 'Hide' : 'Show'} Technical Details
          </button>

          {showDebug && (
            <pre className="debug-content">
              {JSON.stringify(result, null, 2)}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}

export default Scanner;
