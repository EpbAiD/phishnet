import { API_BASE } from './config';
import { PredictionResponse, ExplainResponse, HealthResponse, FeedbackPayload } from './types';

// Check API health
export async function checkHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE}/health`);
  if (!response.ok) throw new Error('API not available');
  return response.json();
}

// Predict URL (simple)
export async function predictUrl(url: string): Promise<PredictionResponse> {
  const response = await fetch(`${API_BASE}/predict/url`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url })
  });
  if (!response.ok) throw new Error('Prediction failed');
  return response.json();
}

// Explain URL (full analysis with AI explanation)
export async function explainUrl(url: string): Promise<ExplainResponse> {
  const response = await fetch(`${API_BASE}/explain`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url })
  });
  if (!response.ok) throw new Error('Explanation failed');
  return response.json();
}

// Submit feedback
export async function submitFeedback(payload: FeedbackPayload): Promise<void> {
  const response = await fetch(`${API_BASE}/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  if (!response.ok) throw new Error('Feedback submission failed');
}
