const BACKEND_URL = "http://localhost:5000/api/behaviour";

const RISKY_DOMAINS = [
    "mail.google.com",
    "outlook.live.com",
    "mail.yahoo.com",
    "mail.proton.me",
    "test_layer1.html"
];

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    if (changeInfo.status === 'complete' && tab.url) {
        try {
            const url = new URL(tab.url);
            let riskType = null;

            // Check for insecure protocol
            if (url.protocol === 'http:') {
                riskType = 'INSECURE_HTTP';
            } 
            // Check for webmail access (potential phishing/data leak surface)
            else if (RISKY_DOMAINS.some(domain => url.hostname.includes(domain) || url.href.includes(domain))) {
                riskType = 'WEBMAIL_ACCESS';
            }

            if (riskType) {
                console.log(`[Cyber Guard] Detected ${riskType} at ${url.hostname}`);
                sendTelemetry(riskType, url.hostname);
            }
        } catch (e) {
            // Ignore invalid URLs
        }
    }
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type) {
        console.log(`[Cyber Guard] Received content telemetry: ${message.type}`);
        // Add time pattern tracking (hour of day)
        const currentHour = new Date().getHours();
        const details = message.details || {};
        details.hour_of_day = currentHour;
        
        sendTelemetry(message.type, message.domain || (sender.tab ? sender.tab.url : 'unknown'), details);
    }
});

async function sendTelemetry(type, host, details = {}) {
    try {
        await fetch(BACKEND_URL, {
            method: 'POST',
            mode: 'cors',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                event: type,
                domain: host,
                timestamp: new Date().toISOString(),
                details: details
            })
        });
    } catch (e) {
        console.error("Failed to send telemetry to local backend:", e);
    }
}
