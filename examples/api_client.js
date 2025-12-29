/**
 * JavaScript/TypeScript client for Phishing Detection API
 * Can be used in Node.js, React, Vue, Angular, or any JavaScript app
 */

class PhishingDetectionClient {
  /**
   * Initialize the API client
   * @param {string} baseURL - Base URL of the API (default: localhost:8000)
   * @param {string|null} apiKey - Optional API key for authentication
   */
  constructor(baseURL = 'http://localhost:8000', apiKey = null) {
    this.baseURL = baseURL.replace(/\/$/, '');
    this.headers = {
      'Content-Type': 'application/json',
    };

    if (apiKey) {
      this.headers['Authorization'] = `Bearer ${apiKey}`;
    }
  }

  /**
   * Check if API is running
   * @returns {Promise<object>}
   */
  async healthCheck() {
    const response = await fetch(`${this.baseURL}/health`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  }

  /**
   * Predict phishing risk using URL features only
   * Fast (~10-50ms), works 100% of the time
   * @param {string} url - The URL to check
   * @returns {Promise<object>}
   */
  async predictURL(url) {
    const response = await fetch(`${this.baseURL}/predict/url`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({ url }),
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  }

  /**
   * Predict phishing risk using WHOIS features only
   * Slow (~1-10 seconds), may fail for some domains
   * @param {string} url - The URL to check
   * @returns {Promise<object>}
   */
  async predictWHOIS(url) {
    const response = await fetch(`${this.baseURL}/predict/whois`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({ url }),
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  }

  /**
   * Predict phishing risk using ensemble of URL + WHOIS models
   * Best accuracy (~1-10 seconds due to WHOIS lookup)
   * @param {string} url - The URL to check
   * @returns {Promise<object>}
   */
  async predictEnsemble(url) {
    const response = await fetch(`${this.baseURL}/predict/ensemble`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({ url }),
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  }

  /**
   * Simple helper: Returns true if URL is safe, false if phishing
   * @param {string} url - The URL to check
   * @param {boolean} useEnsemble - Use ensemble model (slower, more accurate) vs URL only
   * @returns {Promise<boolean>}
   */
  async isURLSafe(url, useEnsemble = true) {
    const result = useEnsemble
      ? await this.predictEnsemble(url)
      : await this.predictURL(url);
    return !result.is_phishing;
  }
}

// ===============================================================
// Example Usage (Node.js or Browser)
// ===============================================================

async function main() {
  const client = new PhishingDetectionClient('http://localhost:8000');

  // Health check
  console.log('='.repeat(60));
  console.log('Health Check');
  console.log('='.repeat(60));
  const health = await client.healthCheck();
  console.log(JSON.stringify(health, null, 2));

  // Test legitimate URL
  console.log('\n' + '='.repeat(60));
  console.log('Testing Legitimate URL');
  console.log('='.repeat(60));
  const legitResult = await client.predictURL('https://www.google.com');
  console.log(`URL: ${legitResult.url}`);
  console.log(`Risk Score: ${(legitResult.risk_score * 100).toFixed(1)}%`);
  console.log(`Is Phishing: ${legitResult.is_phishing}`);
  console.log(`Latency: ${legitResult.latency_ms.toFixed(1)}ms`);

  // Test phishing URL
  console.log('\n' + '='.repeat(60));
  console.log('Testing Phishing URL');
  console.log('='.repeat(60));
  const phishResult = await client.predictURL(
    'https://paypal-verify-account.com/login'
  );
  console.log(`URL: ${phishResult.url}`);
  console.log(`Risk Score: ${(phishResult.risk_score * 100).toFixed(1)}%`);
  console.log(`Is Phishing: ${phishResult.is_phishing}`);
  console.log(`Latency: ${phishResult.latency_ms.toFixed(1)}ms`);

  // Test ensemble
  console.log('\n' + '='.repeat(60));
  console.log('Testing Ensemble Model');
  console.log('='.repeat(60));
  const ensembleResult = await client.predictEnsemble(
    'https://github.com/microsoft/vscode'
  );
  console.log(`URL: ${ensembleResult.url}`);
  console.log(`Risk Score: ${(ensembleResult.risk_score * 100).toFixed(1)}%`);
  console.log(`Is Phishing: ${ensembleResult.is_phishing}`);
  console.log(`Latency: ${ensembleResult.latency_ms.toFixed(0)}ms`);
  console.log('\nDebug Info:');
  console.log(
    `  URL Model: ${(ensembleResult.debug.url_prediction.risk_score * 100).toFixed(1)}%`
  );
  console.log(
    `  WHOIS Model: ${(ensembleResult.debug.whois_prediction.risk_score * 100).toFixed(1)}%`
  );

  // Simple safety check
  console.log('\n' + '='.repeat(60));
  console.log('Simple Safety Check');
  console.log('='.repeat(60));
  const isSafe1 = await client.isURLSafe('https://www.google.com', false);
  const isSafe2 = await client.isURLSafe(
    'https://paypal-verify.com/login',
    false
  );
  console.log(`https://www.google.com: ${isSafe1 ? '✓ SAFE' : '✗ PHISHING'}`);
  console.log(
    `https://paypal-verify.com/login: ${isSafe2 ? '✓ SAFE' : '✗ PHISHING'}`
  );
}

// Run examples (Node.js only)
if (typeof require !== 'undefined' && require.main === module) {
  main().catch(console.error);
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = PhishingDetectionClient;
}
