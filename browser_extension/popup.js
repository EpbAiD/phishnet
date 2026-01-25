const API_BASE = 'http://localhost:8000';

// Track current scan for feedback
let currentScanId = null;
let currentPrediction = null;
let currentUrl = null;

// Get current tab URL and check it
async function checkCurrentPage() {
    const resultDiv = document.getElementById('result');

    // Show loading
    resultDiv.innerHTML = `
        <div class="status status-loading">
            <div class="spinner"></div>
            <div class="status-text">Analyzing...</div>
        </div>
    `;

    try {
        // Get current tab URL
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        const url = tab.url;
        currentUrl = url;

        // Skip chrome:// and extension URLs
        if (url.startsWith('chrome://') || url.startsWith('chrome-extension://')) {
            resultDiv.innerHTML = `
                <div class="status status-loading">
                    <div class="status-icon">ℹ️</div>
                    <div class="status-text">Cannot Check This Page</div>
                    <div class="status-score">Chrome internal pages cannot be analyzed</div>
                </div>
            `;
            return;
        }

        // Display URL
        resultDiv.innerHTML = `
            <div class="url-display">
                <strong>Checking:</strong><br>${url}
            </div>
            <div class="status status-loading">
                <div class="spinner"></div>
                <div class="status-text">Analyzing URL...</div>
            </div>
        `;

        // Check URL using API
        const response = await fetch(`${API_BASE}/predict/url`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: url })
        });

        if (!response.ok) {
            throw new Error('API request failed');
        }

        const data = await response.json();
        currentPrediction = data.is_phishing ? 1 : 0;

        // Record scan in database
        await recordScan(url, data);

        displayResult(data, url);

    } catch (error) {
        resultDiv.innerHTML = `
            <div class="error">
                <strong>Error:</strong> ${error.message}<br><br>
                Make sure the PhishNet API is running:<br>
                <code>docker-compose up</code>
            </div>
            <button onclick="checkCurrentPage()">Retry</button>
        `;
    }
}

// Record scan in database for feedback tracking
async function recordScan(url, predictionData) {
    try {
        const response = await fetch(`${API_BASE}/scan`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                url: url,
                prediction: predictionData.is_phishing ? 1 : 0,
                confidence: predictionData.risk_score,
                url_model_score: predictionData.risk_score,
                dns_model_score: null,
                whois_model_score: null,
                explanation: null,
                model_version: predictionData.model_name,
                source: 'browser_extension'
            })
        });

        if (response.ok) {
            const data = await response.json();
            currentScanId = data.id;
        }
    } catch (error) {
        console.error('Failed to record scan:', error);
        // Non-critical, continue without scan ID
    }
}

function displayResult(data, url) {
    const resultDiv = document.getElementById('result');
    const risk = data.risk_score * 100;
    let statusClass, icon, title, message, actions;

    if (data.is_phishing) {
        statusClass = 'status-danger';
        icon = '🚨';
        title = 'PHISHING DETECTED';
        message = 'This website may be trying to steal your information';
        actions = `
            <ul class="warning-list">
                <li>❌ DO NOT enter passwords or personal info</li>
                <li>❌ DO NOT download anything</li>
                <li>✅ Close this page immediately</li>
            </ul>
        `;
    } else if (risk < 10) {
        statusClass = 'status-safe';
        icon = '✅';
        title = 'SAFE';
        message = 'This website appears to be legitimate';
        actions = `
            <ul class="warning-list">
                <li>✅ This site passed security checks</li>
                <li>✅ Still verify HTTPS lock in address bar</li>
                <li>✅ Be cautious with sensitive information</li>
            </ul>
        `;
    } else {
        statusClass = 'status-loading';
        icon = '⚠️';
        title = 'BE CAUTIOUS';
        message = 'Some suspicious indicators detected';
        actions = `
            <ul class="warning-list">
                <li>⚠️ Proceed with caution</li>
                <li>⚠️ Verify the website is legitimate</li>
                <li>⚠️ Avoid entering sensitive information</li>
            </ul>
        `;
    }

    resultDiv.innerHTML = `
        <div class="url-display">
            <strong>URL:</strong><br>${url.substring(0, 60)}${url.length > 60 ? '...' : ''}
        </div>
        <div class="status ${statusClass}">
            <div class="status-icon">${icon}</div>
            <div class="status-text">${title}</div>
            <div class="status-score">Risk Score: ${risk.toFixed(1)}%</div>
        </div>
        <p style="font-size: 13px; margin: 15px 0;">${message}</p>
        ${actions}
        <div class="details">
            <div class="detail-row">
                <span class="detail-label">Model:</span>
                <span>${data.model_name}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Analysis Time:</span>
                <span>${data.latency_ms.toFixed(0)}ms</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Threshold:</span>
                <span>${(data.threshold * 100).toFixed(0)}%</span>
            </div>
        </div>

        <!-- Feedback Section -->
        <div class="feedback-section" id="feedback-section">
            <div class="feedback-title">Was this prediction correct?</div>
            <div class="feedback-buttons">
                <button class="feedback-btn" id="feedback-correct">✓ Correct</button>
                <button class="feedback-btn" id="feedback-safe">Actually Safe</button>
                <button class="feedback-btn" id="feedback-phishing">Actually Phishing</button>
            </div>
        </div>

        <button id="explain-btn">🤖 Why? (AI Explanation)</button>
        <button id="recheck-btn">🔄 Re-check</button>
    `;

    // Attach event listeners (CSP compliant)
    document.getElementById('explain-btn').addEventListener('click', () => showExplanation(url));
    document.getElementById('recheck-btn').addEventListener('click', checkCurrentPage);

    // Feedback button listeners
    document.getElementById('feedback-correct').addEventListener('click', () => submitFeedback('correct'));
    document.getElementById('feedback-safe').addEventListener('click', () => submitFeedback('safe'));
    document.getElementById('feedback-phishing').addEventListener('click', () => submitFeedback('phishing'));
}

// Submit feedback to API
async function submitFeedback(feedbackType) {
    if (!currentScanId) {
        console.error('No scan ID available for feedback');
        showFeedbackMessage('Unable to submit feedback - please try again');
        return;
    }

    let correctLabel = null;

    if (feedbackType === 'correct') {
        // User confirms prediction is correct - no correction needed
        correctLabel = null;
    } else if (feedbackType === 'safe') {
        // User says it's actually safe (legitimate)
        correctLabel = 0;
    } else if (feedbackType === 'phishing') {
        // User says it's actually phishing
        correctLabel = 1;
    }

    try {
        const response = await fetch(`${API_BASE}/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                scan_id: currentScanId,
                correct_label: correctLabel,
                explanation_helpful: null,
                explanation_comment: null,
                source: 'browser_extension'
            })
        });

        if (response.ok) {
            showFeedbackMessage('Thank you for your feedback!');
        } else {
            showFeedbackMessage('Failed to submit feedback');
        }
    } catch (error) {
        console.error('Failed to submit feedback:', error);
        showFeedbackMessage('Failed to submit feedback');
    }
}

// Show feedback submitted message
function showFeedbackMessage(message) {
    const feedbackSection = document.getElementById('feedback-section');
    if (feedbackSection) {
        feedbackSection.innerHTML = `
            <div class="feedback-submitted">${message}</div>
        `;
    }
}

async function showExplanation(url) {
    const resultDiv = document.getElementById('result');

    // Show loading state
    resultDiv.innerHTML = `
        <div class="status status-loading">
            <div class="spinner"></div>
            <div class="status-text">Generating AI Explanation...</div>
            <p style="font-size: 13px; margin-top: 10px;">Please wait while our AI analyzes this URL...</p>
        </div>
    `;

    try {
        const response = await fetch(`${API_BASE}/explain`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: url })
        });

        if (!response.ok) {
            throw new Error('Failed to get explanation');
        }

        const data = await response.json();

        // Update scan with explanation if we have a scan ID
        if (currentScanId) {
            updateScanExplanation(data);
        }

        displayExplanation(data, url);

    } catch (error) {
        resultDiv.innerHTML = `
            <div class="error">
                <strong>Error:</strong> ${error.message}<br><br>
                Make sure the PhishNet API is running with Groq API key configured.
            </div>
            <button id="back-btn">← Back to Results</button>
        `;
        document.getElementById('back-btn').addEventListener('click', checkCurrentPage);
    }
}

// Update scan record with explanation and model scores
async function updateScanExplanation(explanationData) {
    // This is a fire-and-forget update, no need to block
    try {
        await fetch(`${API_BASE}/scan/${currentScanId}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                explanation: explanationData.explanation,
                url_model_score: explanationData.predictions?.url_prob,
                whois_model_score: explanationData.predictions?.whois_prob,
                dns_model_score: explanationData.predictions?.dns_prob
            })
        });
    } catch (error) {
        console.error('Failed to update scan with explanation:', error);
    }
}

function displayExplanation(data, url) {
    const resultDiv = document.getElementById('result');
    const verdict = data.verdict === 'phishing' ? '🚨 PHISHING' : '✅ SAFE';
    const verdictClass = data.verdict === 'phishing' ? 'status-danger' : 'status-safe';

    resultDiv.innerHTML = `
        <div class="url-display">
            <strong>URL:</strong><br>${url.substring(0, 60)}${url.length > 60 ? '...' : ''}
        </div>
        <div class="status ${verdictClass}">
            <div class="status-text">${verdict}</div>
        </div>
        <div style="background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 15px 0; max-height: 300px; overflow-y: auto;">
            <h3 style="margin: 0 0 10px 0; font-size: 14px; color: #667eea;">🤖 AI Explanation:</h3>
            <div style="white-space: pre-wrap; font-size: 12px; line-height: 1.6;">${data.explanation || 'No explanation available'}</div>
        </div>
        ${data.predictions ? `
            <div class="details">
                <div class="detail-row">
                    <span class="detail-label">URL Model:</span>
                    <span>${(data.predictions.url_prob * 100).toFixed(1)}%</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">WHOIS Model:</span>
                    <span>${data.predictions.whois_prob ? (data.predictions.whois_prob * 100).toFixed(1) + '%' : 'N/A'}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Ensemble:</span>
                    <span>${(data.predictions.ensemble_prob * 100).toFixed(1)}%</span>
                </div>
            </div>
        ` : ''}

        <!-- Explanation Feedback -->
        <div class="feedback-section" id="explanation-feedback-section">
            <div class="feedback-title">Was this explanation helpful?</div>
            <div class="feedback-buttons">
                <button class="feedback-btn" id="explanation-helpful">👍 Helpful</button>
                <button class="feedback-btn" id="explanation-not-helpful">👎 Not Helpful</button>
            </div>
        </div>

        <button id="back-btn">← Back to Results</button>
    `;

    document.getElementById('back-btn').addEventListener('click', checkCurrentPage);

    // Explanation feedback listeners
    document.getElementById('explanation-helpful').addEventListener('click', () => submitExplanationFeedback(true));
    document.getElementById('explanation-not-helpful').addEventListener('click', () => submitExplanationFeedback(false));
}

// Submit explanation feedback
async function submitExplanationFeedback(helpful) {
    if (!currentScanId) {
        showExplanationFeedbackMessage('Unable to submit feedback');
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                scan_id: currentScanId,
                correct_label: null,
                explanation_helpful: helpful,
                explanation_comment: null,
                source: 'browser_extension'
            })
        });

        if (response.ok) {
            showExplanationFeedbackMessage('Thank you for your feedback!');
        } else {
            showExplanationFeedbackMessage('Failed to submit feedback');
        }
    } catch (error) {
        console.error('Failed to submit explanation feedback:', error);
        showExplanationFeedbackMessage('Failed to submit feedback');
    }
}

function showExplanationFeedbackMessage(message) {
    const feedbackSection = document.getElementById('explanation-feedback-section');
    if (feedbackSection) {
        feedbackSection.innerHTML = `
            <div class="feedback-submitted">${message}</div>
        `;
    }
}

// Check page when popup opens
document.addEventListener('DOMContentLoaded', checkCurrentPage);
