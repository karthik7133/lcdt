// content.js
let pageLoadTime = Date.now();
let firstInteractionTime = null;

// Response time (Time to interaction)
function recordInteraction() {
    if (!firstInteractionTime) {
        firstInteractionTime = Date.now();
        const responseTime = firstInteractionTime - pageLoadTime;
        chrome.runtime.sendMessage({
            type: 'RESPONSE_TIME',
            details: { response_time_ms: responseTime }
        });
    }
}

document.addEventListener('click', recordInteraction);
document.addEventListener('keydown', recordInteraction);

// Link clicks
document.addEventListener('click', (e) => {
    const link = e.target.closest('a');
    if (link) {
        chrome.runtime.sendMessage({
            type: 'LINK_CLICK',
            details: { href: link.href }
        });
    }
});

// Password security: Hashing for privacy
async function hashPassword(password) {
    const encoder = new TextEncoder();
    const data = encoder.encode(password);
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

// Password strength
function checkPasswordStrength(password) {
    let score = 0;
    if (password.length >= 8) score++;
    if (/[A-Z]/.test(password)) score++;
    if (/[0-9]/.test(password)) score++;
    if (/[^A-Za-z0-9]/.test(password)) score++;
    return score; // 0 to 4
}

document.addEventListener('blur', async (e) => {
    if (e.target && e.target.type === 'password' && e.target.value) {
        const password = e.target.value;
        const score = checkPasswordStrength(password);
        const pwHash = await hashPassword(password);

        // Always send a password event for habit tracking, but ONLY with the hash and score
        chrome.runtime.sendMessage({
            type: 'PASSWORD_ENTRY',
            details: { 
                score: score,
                password_hash: pwHash, // Hashed for privacy
                is_weak: score < 3
            }
        });

        if (score < 3) {
            console.log("[Cyber Guard] Weak password detected (hashed).");
            chrome.runtime.sendMessage({
                type: 'LOW_STRENGTH_PASSWORD',
                details: { score: score }
            });
        }
    }
}, true); // use capture phase for blur

// Sender detection (Gmail/Outlook)
let lastSendersDetected = "";

function detectSenders() {
    const hostname = window.location.hostname;
    let senders = [];
    
    if (hostname.includes('mail.google.com')) {
        const elements = document.querySelectorAll('[email]');
        elements.forEach(el => senders.push(el.getAttribute('email')));
    } else if (hostname.includes('outlook.live.com') || hostname.includes('outlook.office.com')) {
        const elements = document.querySelectorAll('.OWeJaf, .ms-Persona-primaryText'); // Example classes
        elements.forEach(el => senders.push(el.innerText));
    }
    
    // Filter, unique, and SORT to compare content reliably
    senders = [...new Set(senders)].filter(s => s && s.includes('@')).sort();
    const currentSendersStr = senders.join(',');
    
    if (senders.length > 0 && currentSendersStr !== lastSendersDetected) {
        lastSendersDetected = currentSendersStr;
        console.log(`[Cyber Guard] New senders detected: ${senders.length}`);
        chrome.runtime.sendMessage({
            type: 'EMAIL_SENDERS_DETECTED',
            details: { senders: senders, count: senders.length }
        });
    }
}

// Periodically check for senders on webmail
if (['mail.google.com', 'outlook.live.com', 'mail.yahoo.com', 'mail.proton.me', 'test_layer1.html'].some(d => window.location.href.includes(d))) {
    setInterval(detectSenders, 10000); // Check every 10 seconds
}
