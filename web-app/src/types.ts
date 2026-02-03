// API Response Types
export interface PredictionResponse {
  url: string;
  risk_score: number;
  is_phishing: boolean;
  threshold: number;
  model_name: string;
  latency_ms: number;
  explanation: string;
  verdict: string;
  confidence: string;
  scan_id: number | null;
  debug?: Record<string, unknown>;
}

export interface ExplainResponse {
  url: string;
  explanation: string;
  technical_explanation?: string;
  predictions: {
    url_prob: number;
    whois_prob?: number;
    dns_prob?: number;
    ensemble_prob: number;
  };
  verdict: string;
  confidence: string;
  latency_ms: number;
  scan_id: number | null;
  shap_features?: Record<string, unknown>;
}

export interface HealthResponse {
  status: string;
  model_version: string;
  last_reload: string;
  reload_count: number;
  hot_reload_enabled: boolean;
}

export interface FeedbackPayload {
  scan_id: number;
  correct_label?: number | null;
  explanation_helpful?: boolean | null;
  explanation_comment?: string | null;
  source: 'web' | 'browser_extension';
}

// UI State Types
export type AnalysisMode = 'url' | 'explain';

export interface FeedbackState {
  correctLabel: number | null;
  explanationHelpful: boolean | null;
}
