// PhishNet Content Script - Runs on every page
// Features:
// 1. Auto-check current page and show banner
// 2. Scan all links on page (Gmail, Google Search, etc.)
// 3. Add safety badges to links
// 4. Show explanations on hover/click

const API_BASE = 'http://localhost:8000';
const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes

// Cache for URL check results
let urlCache = {};

console.log('🔒 PhishNet Protection Active');

// =========================================
// 1. AUTO-CHECK CURRENT PAGE
// =========================================

async function checkCurrentPage() {
    const currentUrl = window.location.href;

    // Skip chrome:// pages, localhost, and file://
    if (currentUrl.startsWith('chrome://') ||
        currentUrl.startsWith('chrome-extension://') ||
        currentUrl.startsWith('file://') ||
        currentUrl.includes('localhost')) {
        return;
    }

    try {
        const result = await checkURL(currentUrl, 'url'); // Fast check
        showPageBanner(result, currentUrl);
    } catch (error) {
        console.error('PhishNet: Error checking current page', error);
    }
}

function showPageBanner(result, url) {
    // Don't show banner if already checked recently
    if (document.getElementById('phishnet-page-banner')) {
        return;
    }

    const risk = result.risk_score * 100;
    let bannerClass, icon, title, message;

    if (result.is_phishing) {
        bannerClass = 'phishnet-banner-danger';
        icon = '🚨';
        title = 'DANGER - PHISHING WEBSITE DETECTED';
        message = 'This site may be trying to steal your information. Do not enter passwords or personal data.';
    } else if (risk < 10) {
        // Don't show banner for very safe sites (reduces annoyance)
        return;
    } else if (risk < 40) {
        bannerClass = 'phishnet-banner-safe';
        icon = '✅';
        title = 'This site appears safe';
        message = `PhishNet scanned this page (Risk: ${risk.toFixed(1)}%)`;
    } else {
        bannerClass = 'phishnet-banner-warning';
        icon = '⚠️';
        title = 'BE CAUTIOUS';
        message = `Some suspicious indicators detected (Risk: ${risk.toFixed(1)}%)`;
    }

    const banner = document.createElement('div');
    banner.id = 'phishnet-page-banner';
    banner.className = `phishnet-banner ${bannerClass}`;
    banner.innerHTML = `
        <div class="phishnet-banner-content">
            <div class="phishnet-banner-left">
                <div class="phishnet-banner-icon">${icon}</div>
                <div class="phishnet-banner-text">
                    <div class="phishnet-banner-title">${title}</div>
                    <div class="phishnet-banner-message">${message}</div>
                </div>
            </div>
            <div class="phishnet-banner-actions">
                ${result.is_phishing ? '<button class="phishnet-banner-btn phishnet-banner-btn-primary phishnet-close-tab-btn">Close Tab</button>' : ''}
                <button class="phishnet-banner-btn phishnet-banner-btn-primary" id="phishnet-explain-btn">Why?</button>
                <button class="phishnet-banner-btn phishnet-banner-btn-close" id="phishnet-banner-close">✕</button>
            </div>
        </div>
    `;

    document.body.insertBefore(banner, document.body.firstChild);

    // Add event listeners (CSP compliant)
    const closeBtn = banner.querySelector('#phishnet-banner-close');
    if (closeBtn) {
        closeBtn.addEventListener('click', () => {
            banner.remove();
        });
    }

    const explainBtn = banner.querySelector('#phishnet-explain-btn');
    if (explainBtn) {
        explainBtn.addEventListener('click', () => {
            showExplanationModal(url);
        });
    }

    const closeTabBtn = banner.querySelector('.phishnet-close-tab-btn');
    if (closeTabBtn) {
        closeTabBtn.addEventListener('click', () => {
            window.close();
        });
    }

    // Auto-hide safe banners after 5 seconds
    if (!result.is_phishing) {
        setTimeout(() => {
            banner.style.opacity = '0';
            setTimeout(() => banner.remove(), 300);
        }, 5000);
    }
}

// =========================================
// 2. SCAN ALL LINKS ON PAGE
// =========================================

async function scanAllLinks() {
    const links = document.querySelectorAll('a[href^="http"]');

    // Limit to first 50 links to avoid overloading API
    const linksToScan = Array.from(links).slice(0, 50);

    for (const link of linksToScan) {
        const url = link.href;

        // Skip if already has badge
        if (link.querySelector('.phishnet-link-badge')) {
            continue;
        }

        // Add "checking" badge
        addLinkBadge(link, 'checking', '⏳ Checking...', null);

        // Check URL
        try {
            const result = await checkURL(url, 'url');
            updateLinkBadge(link, result);
        } catch (error) {
            // Remove checking badge on error
            const badge = link.querySelector('.phishnet-link-badge');
            if (badge) badge.remove();
        }
    }
}

function addLinkBadge(link, type, text, result) {
    const badge = document.createElement('span');
    badge.className = `phishnet-link-badge phishnet-link-badge-${type}`;
    badge.textContent = text;
    badge.dataset.result = JSON.stringify(result);

    // Add tooltip on hover
    if (result) {
        badge.addEventListener('mouseenter', (e) => {
            showTooltip(e.target, result);
        });
        badge.addEventListener('mouseleave', hideTooltip);
        badge.addEventListener('click', async (e) => {
            e.preventDefault();
            e.stopPropagation();
            showExplanationModal(link.href);
        });
    }

    link.appendChild(badge);
}

function updateLinkBadge(link, result) {
    const badge = link.querySelector('.phishnet-link-badge');
    if (!badge) return;

    const risk = result.risk_score * 100;
    let type, text, linkClass;

    if (result.is_phishing) {
        type = 'danger';
        text = `🚨 ${risk.toFixed(0)}% Risk`;
        linkClass = 'phishnet-link-danger';
    } else if (risk < 20) {
        type = 'safe';
        text = `✅ Safe`;
        linkClass = 'phishnet-link-safe';
    } else {
        type = 'warning';
        text = `⚠️ ${risk.toFixed(0)}% Risk`;
        linkClass = 'phishnet-link-warning';
    }

    badge.className = `phishnet-link-badge phishnet-link-badge-${type}`;
    badge.textContent = text;
    badge.dataset.result = JSON.stringify(result);
    link.classList.add(linkClass);

    // Add tooltip
    badge.addEventListener('mouseenter', (e) => {
        showTooltip(e.target, result);
    });
    badge.addEventListener('mouseleave', hideTooltip);
    badge.addEventListener('click', async (e) => {
        e.preventDefault();
        e.stopPropagation();
        showExplanationModal(link.href);
    });
}

// =========================================
// 3. TOOLTIP FOR LINK BADGES
// =========================================

let currentTooltip = null;

function showTooltip(element, result) {
    hideTooltip();

    const tooltip = document.createElement('div');
    tooltip.className = 'phishnet-tooltip show';

    const risk = result.risk_score * 100;
    const verdict = result.is_phishing ? 'Phishing Detected' : 'Appears Safe';

    tooltip.innerHTML = `
        <div class="phishnet-tooltip-title">${verdict}</div>
        <div class="phishnet-tooltip-text">
            ${result.is_phishing ?
                'This URL shows signs of phishing. Click for details.' :
                'This URL passed security checks.'}
        </div>
        <div class="phishnet-tooltip-score">Risk Score: ${risk.toFixed(1)}%</div>
    `;

    document.body.appendChild(tooltip);

    const rect = element.getBoundingClientRect();
    tooltip.style.position = 'fixed';
    tooltip.style.top = (rect.bottom + 5) + 'px';
    tooltip.style.left = rect.left + 'px';

    currentTooltip = tooltip;
}

function hideTooltip() {
    if (currentTooltip) {
        currentTooltip.remove();
        currentTooltip = null;
    }
}

// =========================================
// 4. EXPLANATION MODAL
// =========================================

async function showExplanationModal(url) {
    // Show loading modal
    const modal = createModal('Loading explanation...', '<div style="text-align: center; padding: 40px;">⏳ Analyzing URL...</div>');
    document.body.appendChild(modal);
    modal.classList.add('show');

    try {
        const response = await fetch(`${API_BASE}/explain`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: url })
        });

        const data = await response.json();

        // Update modal with explanation
        updateModal(modal, data);
    } catch (error) {
        updateModal(modal, {
            explanation: `Error: ${error.message}\n\nMake sure PhishNet API is running at ${API_BASE}`
        });
    }
}

function createModal(title, content) {
    const modal = document.createElement('div');
    modal.className = 'phishnet-modal';
    modal.innerHTML = `
        <div class="phishnet-modal-content">
            <div class="phishnet-modal-header">
                <h2 class="phishnet-modal-title">${title}</h2>
            </div>
            <div class="phishnet-modal-body">
                ${content}
            </div>
            <div class="phishnet-modal-footer">
                <button class="phishnet-modal-close">Close</button>
            </div>
        </div>
    `;

    // Close on button click
    modal.querySelector('.phishnet-modal-close').addEventListener('click', () => {
        modal.remove();
    });

    // Close on backdrop click
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.remove();
        }
    });

    return modal;
}

function updateModal(modal, data) {
    const title = data.verdict === 'phishing' ? '🚨 Phishing Detected' : '✅ Site Appears Safe';
    const explanation = data.explanation || 'No explanation available';

    modal.querySelector('.phishnet-modal-title').textContent = title;
    modal.querySelector('.phishnet-modal-body').innerHTML = `
        <div style="white-space: pre-wrap; line-height: 1.6;">${explanation}</div>
        ${data.predictions ? `
            <div style="margin-top: 20px; padding-top: 20px; border-top: 1px solid #e0e0e0;">
                <strong>Model Predictions:</strong><br>
                URL Model: ${(data.predictions.url_prob * 100).toFixed(1)}%<br>
                WHOIS Model: ${data.predictions.whois_prob ? (data.predictions.whois_prob * 100).toFixed(1) + '%' : 'N/A'}<br>
                Ensemble: ${(data.predictions.ensemble_prob * 100).toFixed(1)}%
            </div>
        ` : ''}
    `;
}

// =========================================
// 5. API HELPER FUNCTIONS
// =========================================

async function checkURL(url, method = 'url') {
    // Check cache first
    const cacheKey = `${url}-${method}`;
    const cached = urlCache[cacheKey];

    if (cached && (Date.now() - cached.timestamp < CACHE_DURATION)) {
        return cached.result;
    }

    // Call API
    const response = await fetch(`${API_BASE}/predict/${method}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: url }),
        signal: AbortSignal.timeout(10000) // 10 second timeout
    });

    if (!response.ok) {
        throw new Error('API request failed');
    }

    const result = await response.json();

    // Cache result
    urlCache[cacheKey] = {
        result: result,
        timestamp: Date.now()
    };

    return result;
}

// =========================================
// 6. INITIALIZATION
// =========================================

// Auto-check current page when loaded
setTimeout(checkCurrentPage, 1000);

// Scan links after page loads (with delay to avoid interfering with page load)
setTimeout(scanAllLinks, 3000);

// Re-scan links when DOM changes (for dynamic content like Gmail)
const observer = new MutationObserver((mutations) => {
    // Debounce: only scan if there are new links
    const hasNewLinks = mutations.some(m =>
        Array.from(m.addedNodes).some(node =>
            node.tagName === 'A' || (node.querySelectorAll && node.querySelectorAll('a[href^="http"]').length > 0)
        )
    );

    if (hasNewLinks) {
        setTimeout(scanAllLinks, 500);
    }
});

observer.observe(document.body, {
    childList: true,
    subtree: true
});

console.log('✅ PhishNet: Auto-scan enabled for links');
