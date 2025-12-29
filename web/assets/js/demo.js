// PhishNet Demo - JavaScript Functionality

// Load system stats on page load
document.addEventListener('DOMContentLoaded', () => {
    loadSystemStats();
});

// Load system statistics
async function loadSystemStats() {
    try {
        // Try to fetch stats from API
        const response = await fetch(`${API_BASE_URL}/api/stats`);
        if (response.ok) {
            const stats = await response.json();
            document.getElementById('total-urls').textContent = formatNumber(stats.total_urls || 0);
            document.getElementById('model-accuracy').textContent = (stats.accuracy || 0).toFixed(1) + '%';
            document.getElementById('last-updated').textContent = formatDate(stats.last_updated);
        } else {
            // Use fallback values if API not available
            useFallbackStats();
        }
    } catch (error) {
        console.log('API not available, using fallback stats');
        useFallbackStats();
    }
}

function useFallbackStats() {
    document.getElementById('total-urls').textContent = '50,000+';
    document.getElementById('model-accuracy').textContent = '96.5%';
    document.getElementById('last-updated').textContent = 'Weekly';
}

// Format numbers with commas
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

// Format date
function formatDate(dateStr) {
    if (!dateStr) return 'Weekly';
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

// Set example URL
function setURL(url) {
    document.getElementById('url-input').value = url;
}

// Check URL
async function checkURL() {
    const urlInput = document.getElementById('url-input');
    const url = urlInput.value.trim();

    // Validate input
    if (!url) {
        showError('Please enter a URL to check');
        return;
    }

    // Basic URL validation
    if (!isValidURL(url)) {
        showError('Please enter a valid URL (e.g., https://example.com)');
        return;
    }

    // Show loading state
    setLoadingState(true);
    hideResults();
    hideError();

    try {
        // Call API
        const response = await fetch(`${API_BASE_URL}/api/check`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ url: url })
        });

        if (!response.ok) {
            throw new Error(`API returned ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();
        displayResults(result);

    } catch (error) {
        console.error('Error checking URL:', error);
        showError(
            `Failed to analyze URL. ${getErrorHelp(error)}`
        );
    } finally {
        setLoadingState(false);
    }
}

// Get helpful error message
function getErrorHelp(error) {
    const errorStr = error.toString().toLowerCase();

    if (errorStr.includes('fetch') || errorStr.includes('network')) {
        return `
            <br><br>
            <strong>Possible causes:</strong><br>
            • Backend API is not running<br>
            • CORS is not configured on the backend<br>
            • Wrong API_BASE_URL in the HTML file<br>
            <br>
            <strong>To fix:</strong><br>
            1. Make sure your FastAPI backend is running<br>
            2. Update API_BASE_URL in index.html<br>
            3. Add CORS middleware to your FastAPI app (see docs)
        `;
    }

    return 'Please check the console for details.';
}

// Validate URL format
function isValidURL(string) {
    try {
        // Add protocol if missing
        if (!string.match(/^https?:\/\//i)) {
            string = 'http://' + string;
            document.getElementById('url-input').value = string;
        }
        new URL(string);
        return true;
    } catch (_) {
        return false;
    }
}

// Display analysis results
function displayResults(result) {
    const resultsSection = document.getElementById('results-section');
    resultsSection.style.display = 'block';

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    // Overall prediction
    const isPhishing = result.prediction === 'phishing' || result.prediction === 1;
    const confidence = result.confidence || result.probability || 0;

    // Set prediction
    document.getElementById('prediction-icon').textContent = isPhishing ? '🚨' : '✅';
    document.getElementById('prediction-title').textContent = isPhishing ? 'Phishing Detected' : 'Safe URL';
    document.getElementById('prediction-title').className = isPhishing ? 'phishing' : 'safe';

    document.getElementById('prediction-desc').textContent = isPhishing
        ? 'This URL shows strong indicators of being a phishing attempt'
        : 'No significant phishing indicators detected';

    document.getElementById('confidence-value').textContent = (confidence * 100).toFixed(1) + '%';

    // Result badge
    document.getElementById('result-badge').textContent = isPhishing ? 'PHISHING' : 'SAFE';
    document.getElementById('result-badge').className = `result-badge ${isPhishing ? 'phishing' : 'safe'}`;

    // Individual model scores
    if (result.model_scores) {
        updateModelScore('url', result.model_scores.url_model || 0);
        updateModelScore('dns', result.model_scores.dns_model || 0);
        updateModelScore('whois', result.model_scores.whois_model || 0);
        updateModelScore('ensemble', result.model_scores.ensemble_model || confidence);
    } else {
        // Use overall confidence for all models if individual scores not available
        updateModelScore('url', confidence);
        updateModelScore('dns', confidence);
        updateModelScore('whois', confidence);
        updateModelScore('ensemble', confidence);
    }

    // Display features
    displayFeatures(result.features || {});
}

// Update individual model score
function updateModelScore(modelName, score) {
    const percentage = (score * 100).toFixed(1);
    document.getElementById(`${modelName}-score`).textContent = percentage + '%';
    document.getElementById(`${modelName}-progress`).style.width = percentage + '%';
}

// Display detected features
function displayFeatures(features) {
    const featuresList = document.getElementById('features-list');
    featuresList.innerHTML = '';

    // Key features to display
    const featureMap = {
        has_ip: 'IP Address in URL',
        has_at_symbol: '@ Symbol in URL',
        long_url: 'Unusually Long URL',
        short_url: 'Shortened URL Service',
        has_https: 'HTTPS Protocol',
        domain_age: 'Domain Age',
        dns_record: 'DNS Record Found',
        suspicious_tld: 'Suspicious TLD',
        subdomain_count: 'Subdomain Count',
        special_chars: 'Special Characters'
    };

    Object.entries(featureMap).forEach(([key, label]) => {
        const value = features[key];
        if (value !== undefined && value !== null) {
            const featureItem = createFeatureItem(label, value);
            featuresList.appendChild(featureItem);
        }
    });

    // If no features available, show message
    if (featuresList.children.length === 0) {
        featuresList.innerHTML = '<p style="color: var(--text-muted); text-align: center;">Feature analysis not available</p>';
    }
}

// Create feature item HTML
function createFeatureItem(label, value) {
    const div = document.createElement('div');
    div.className = 'feature-item';

    // Determine if feature is suspicious
    const isSuspicious = isFeatureSuspicious(label, value);

    div.innerHTML = `
        <div class="feature-icon ${isSuspicious ? 'detected' : 'safe'}">
            ${isSuspicious ? '⚠️' : '✓'}
        </div>
        <div class="feature-text">
            <span class="feature-label">${label}</span>
            <span class="feature-value">${formatFeatureValue(value)}</span>
        </div>
    `;

    return div;
}

// Check if feature value is suspicious
function isFeatureSuspicious(label, value) {
    const suspiciousFeatures = [
        'IP Address in URL',
        '@ Symbol in URL',
        'Unusually Long URL',
        'Shortened URL Service',
        'Suspicious TLD'
    ];

    if (suspiciousFeatures.includes(label)) {
        return value === true || value === 1 || value === 'yes';
    }

    if (label === 'HTTPS Protocol') {
        return value === false || value === 0 || value === 'no';
    }

    if (label === 'DNS Record Found') {
        return value === false || value === 0 || value === 'no';
    }

    return false;
}

// Format feature value for display
function formatFeatureValue(value) {
    if (typeof value === 'boolean') {
        return value ? 'Yes' : 'No';
    }
    if (typeof value === 'number') {
        return value.toFixed(0);
    }
    return value.toString();
}

// Show error message
function showError(message) {
    const errorSection = document.getElementById('error-section');
    const errorMessage = document.getElementById('error-message');

    errorMessage.innerHTML = message;
    errorSection.style.display = 'block';
    errorSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Hide error message
function hideError() {
    document.getElementById('error-section').style.display = 'none';
}

// Hide results
function hideResults() {
    document.getElementById('results-section').style.display = 'none';
}

// Reset form
function resetForm() {
    hideError();
    hideResults();
    document.getElementById('url-input').value = '';
    document.getElementById('url-input').focus();
}

// Set loading state
function setLoadingState(isLoading) {
    const btn = document.getElementById('check-btn');
    const btnText = document.getElementById('btn-text');
    const btnLoader = document.getElementById('btn-loader');

    btn.disabled = isLoading;

    if (isLoading) {
        btnText.style.display = 'none';
        btnLoader.style.display = 'inline-block';
    } else {
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
    }
}

// Allow Enter key to submit
document.getElementById('url-input').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        checkURL();
    }
});
