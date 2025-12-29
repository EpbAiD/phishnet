// Background service worker
// Handles extension lifecycle and background tasks

chrome.runtime.onInstalled.addListener(() => {
    console.log('PhishNet extension installed');
});

// Future: Can implement automatic URL checking and badge updates
