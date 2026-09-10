// Generate particles
function createParticles() {
    const container = document.getElementById('particles');
    for (let i = 0; i < 30; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.animationDelay = Math.random() * 20 + 's';
        particle.style.animationDuration = (15 + Math.random() * 10) + 's';
        container.appendChild(particle);
    }
}
createParticles();

// Screen navigation
let currentScreen = 'home';
let cameraStream = null;
let selectedDecisionType = 'lbw';

// ---------------- AI Model Integration ----------------
// Point this at your running FastAPI backend (backend/app.py)
const API_BASE_URL = 'http://localhost:8000';

let currentVideoFile = null;   // the raw File the user uploaded
let predictionPromise = null;  // in-flight/completed prediction for the current video
let lastPrediction = null;     // resolved prediction result (or null if it failed)

async function getPrediction(file) {
    const formData = new FormData();
    formData.append('video', file);

    const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || `Server error (${response.status})`);
    }

    return response.json(); // { predicted_class, confidence, probabilities }
}

function showScreen(screenId) {
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    document.getElementById(screenId).classList.add('active');
    
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
        if (link.dataset.screen === screenId) {
            link.classList.add('active');
        }
    });
    
    currentScreen = screenId;
    
    if (screenId === 'review') {
        startAnalysisAnimation();
    }
    if (screenId === 'decision') {
        animateDecision();
    }
}

document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        if (link.dataset.screen) {
            showScreen(link.dataset.screen);
        }
    });
});

// Toast notifications
function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <span style="font-size: 1.5rem;">${type === 'success' ? '✓' : type === 'error' ? '✗' : 'ℹ'}</span>
        <span>${message}</span>
    `;
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'toastSlide 0.5s ease reverse';
        setTimeout(() => toast.remove(), 500);
    }, 3000);
}

// Video upload
const uploadZone = document.getElementById('uploadZone');

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('video/')) {
        processVideoFile(file);
    }
});

function handleVideoUpload(e) {
    const file = e.target.files[0];
    if (file) {
        processVideoFile(file);
    }
}

function processVideoFile(file) {
    currentVideoFile = file;
    predictionPromise = null;
    lastPrediction = null;

    // Show progress
    document.getElementById('uploadProgress').style.display = 'block';
    uploadZone.style.display = 'none';
    
    // Simulate upload progress
    let progress = 0;
    const interval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress >= 100) {
            progress = 100;
            clearInterval(interval);
            
            setTimeout(() => {
                document.getElementById('uploadProgress').style.display = 'none';
                document.getElementById('videoPreviewSection').style.display = 'block';
                
                const video = document.getElementById('uploadedVideo');
                video.src = URL.createObjectURL(file);
                
                showToast('Video uploaded successfully!', 'success');
            }, 500);
        }
        document.getElementById('uploadProgressBar').style.width = progress + '%';
        document.getElementById('uploadPercent').textContent = Math.round(progress) + '%';
    }, 100);
}

// Video controls
function playVideo() {
    document.getElementById('uploadedVideo').play();
}

function pauseVideo() {
    document.getElementById('uploadedVideo').pause();
}

function setSpeed(speed) {
    document.getElementById('uploadedVideo').playbackRate = speed;
    showToast(`Playback speed: ${speed}x`, 'info');
}

function setReviewSpeed(speed) {
    document.getElementById('reviewVideo').playbackRate = speed;
}

function prevFrame() {
    const video = document.getElementById('uploadedVideo');
    video.pause();
    video.currentTime = Math.max(0, video.currentTime - 1/30);
}

function nextFrame() {
    const video = document.getElementById('uploadedVideo');
    video.pause();
    video.currentTime += 1/30;
}

// Decision type selection
function selectDecisionType(element) {
    document.querySelectorAll('.decision-type-card').forEach(card => {
        card.classList.remove('selected');
    });
    element.classList.add('selected');
    selectedDecisionType = element.dataset.type;
}

// Camera functions
async function startCamera() {
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 1280, height: 720 } 
        });
        document.getElementById('cameraFeed').srcObject = cameraStream;
        showToast('Camera started successfully!', 'success');
        startLiveTracking();
    } catch (err) {
        showToast('Camera access denied!', 'error');
    }
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        document.getElementById('cameraFeed').srcObject = null;
        showToast('Camera stopped', 'info');
    }
}

function captureFrame() {
    showToast('Frame captured!', 'success');
}

// Live tracking animation
function startLiveTracking() {
    const canvas = document.getElementById('liveTrackingCanvas');
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight;
    
    let ballX = 50;
    let ballY = 50;
    
    function animate() {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Draw pitch
        ctx.fillStyle = '#4a7c3f';
        ctx.fillRect(canvas.width * 0.2, 0, canvas.width * 0.6, canvas.height);
        
        // Draw stumps
        ctx.fillStyle = '#ffffff';
        for (let i = 0; i < 3; i++) {
            ctx.fillRect(canvas.width * 0.48 + i * 15, canvas.height - 60, 5, 50);
        }
        
        // Draw ball trajectory
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(255, 51, 102, 0.5)';
        ctx.lineWidth = 2;
        ctx.moveTo(ballX, ballY);
        ctx.lineTo(canvas.width * 0.5, canvas.height - 30);
        ctx.stroke();
        
        // Draw ball
        ctx.beginPath();
        ctx.fillStyle = '#ff3366';
        ctx.shadowColor = '#ff3366';
        ctx.shadowBlur = 20;
        ctx.arc(ballX, ballY, 10, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
        
        ballX += 2;
        ballY += 3;
        
        if (ballY > canvas.height) {
            ballX = 50;
            ballY = 50;
        }
        
        if (cameraStream) {
            requestAnimationFrame(animate);
        }
    }
    animate();
}

// Start analysis
function startAnalysis() {
    showToast('Starting AI analysis...', 'info');

    // Copy video to review screen
    const uploadedVideo = document.getElementById('uploadedVideo');
    const reviewVideo = document.getElementById('reviewVideo');
    reviewVideo.src = uploadedVideo.src;

    // Fire the real model prediction now, in the background, so it's
    // (hopefully) ready by the time the user reaches the decision screen.
    if (currentVideoFile) {
        predictionPromise = getPrediction(currentVideoFile).catch(err => {
            console.error('Prediction failed:', err);
            return null; // caller checks for null and falls back to demo mode
        });
    }

    showScreen('review');
}

function analyzeCurrentFrame() {
    showToast('Analyzing current frame...', 'info');
    showScreen('review');
}

// Analysis animation
function startAnalysisAnimation() {
    const steps = ['step1', 'step2', 'step3', 'step4'];
    const progressBar = document.getElementById('analysisProgress');
    let currentStep = 0;
    
    // Reset
    steps.forEach(step => {
        document.getElementById(step).classList.remove('active', 'completed');
    });
    document.getElementById('step1').classList.add('active');
    progressBar.style.width = '0%';
    document.getElementById('analysisStatus').textContent = 'Analyzing...';
    document.getElementById('analysisStatus').className = 'badge badge-out';
    
    // Animate no-ball checks
    const checks = ['frontFootCheck', 'heightCheck', 'armCheck', 'wideCheck'];
    checks.forEach(check => {
        document.getElementById(check).className = 'check-status checking';
        document.getElementById(check).textContent = '...';
    });
    
    // Simulate analysis steps
    const interval = setInterval(() => {
        if (currentStep < steps.length) {
            document.getElementById(steps[currentStep]).classList.add('completed');
            document.getElementById(steps[currentStep]).classList.remove('active');
            document.getElementById(steps[currentStep]).textContent = '✓';
            
            currentStep++;
            progressBar.style.width = (currentStep * 25) + '%';
            
            if (currentStep < steps.length) {
                document.getElementById(steps[currentStep]).classList.add('active');
            }
            
            // Update no-ball checks progressively
            if (currentStep <= checks.length) {
                const checkIndex = currentStep - 1;
                if (checkIndex >= 0) {
                    const isValid = Math.random() > 0.2;
                    document.getElementById(checks[checkIndex]).className = `check-status ${isValid ? 'valid' : 'invalid'}`;
                    document.getElementById(checks[checkIndex]).textContent = isValid ? '✓' : '✗';
                }
            }
        } else {
            clearInterval(interval);
            document.getElementById('analysisStatus').textContent = 'Complete';
            document.getElementById('analysisStatus').className = 'badge badge-not-out';
            showToast('Analysis complete!', 'success');
        }
    }, 800);
    
    // Draw visualization
    drawVisualization();
}

// Draw ball trajectory visualization
function drawVisualization() {
    const canvas = document.getElementById('vizCanvas');
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight;
    
    // Clear
    ctx.fillStyle = '#1a1a3a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw pitch
    ctx.fillStyle = '#4a7c3f';
    ctx.fillRect(canvas.width * 0.3, 50, canvas.width * 0.4, canvas.height - 100);
    
    // Draw creases
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(canvas.width * 0.3, canvas.height - 80);
    ctx.lineTo(canvas.width * 0.7, canvas.height - 80);
    ctx.stroke();
    
    ctx.beginPath();
    ctx.moveTo(canvas.width * 0.3, 80);
    ctx.lineTo(canvas.width * 0.7, 80);
    ctx.stroke();
    
    // Draw stumps at bottom
    ctx.fillStyle = 'white';
    for (let i = 0; i < 3; i++) {
        ctx.fillRect(canvas.width * 0.47 + i * 15, canvas.height - 75, 6, 45);
    }
    
    // Draw stumps at top
    for (let i = 0; i < 3; i++) {
        ctx.fillRect(canvas.width * 0.47 + i * 15, 85, 6, 45);
    }
    
    // Animate ball trajectory
    let ballProgress = 0;
    const startX = canvas.width * 0.5;
    const startY = 60;
    const endX = canvas.width * 0.5;
    const endY = canvas.height - 60;
    const impactX = canvas.width * 0.52;
    const impactY = canvas.height * 0.6;
    
    function animateBall() {
        // Redraw background
        ctx.fillStyle = '#1a1a3a';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Redraw pitch
        ctx.fillStyle = '#4a7c3f';
        ctx.fillRect(canvas.width * 0.3, 50, canvas.width * 0.4, canvas.height - 100);
        
        // Draw creases
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(canvas.width * 0.3, canvas.height - 80);
        ctx.lineTo(canvas.width * 0.7, canvas.height - 80);
        ctx.stroke();
        
        // Draw stumps
        ctx.fillStyle = 'white';
        for (let i = 0; i < 3; i++) {
            ctx.fillRect(canvas.width * 0.47 + i * 15, canvas.height - 75, 6, 45);
        }
        
        // Draw trajectory line
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(0, 212, 255, 0.6)';
        ctx.lineWidth = 3;
        ctx.setLineDash([5, 5]);
        ctx.moveTo(startX, startY);
        ctx.lineTo(impactX, impactY);
        ctx.stroke();
        ctx.setLineDash([]);
        
        // Draw predicted path
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(255, 51, 102, 0.8)';
        ctx.lineWidth = 3;
        ctx.moveTo(impactX, impactY);
        ctx.lineTo(endX, endY);
        ctx.stroke();
        
        // Draw impact zone
        ctx.beginPath();
        ctx.fillStyle = 'rgba(255, 51, 102, 0.3)';
        ctx.arc(impactX, impactY, 20, 0, Math.PI * 2);
        ctx.fill();
        
        ctx.beginPath();
        ctx.strokeStyle = '#ff3366';
        ctx.lineWidth = 3;
        ctx.arc(impactX, impactY, 20, 0, Math.PI * 2);
        ctx.stroke();
        
        // Draw ball
        if (ballProgress < 1) {
            const currentX = startX + (impactX - startX) * ballProgress;
            const currentY = startY + (impactY - startY) * ballProgress;
            
            ctx.beginPath();
            ctx.fillStyle = '#ff3366';
            ctx.shadowColor = '#ff3366';
            ctx.shadowBlur = 15;
            ctx.arc(currentX, currentY, 12, 0, Math.PI * 2);
            ctx.fill();
            ctx.shadowBlur = 0;
            
            ballProgress += 0.02;
            requestAnimationFrame(animateBall);
        } else {
            // Final position - draw at impact
            ctx.beginPath();
            ctx.fillStyle = '#ff3366';
            ctx.shadowColor = '#ff3366';
            ctx.shadowBlur = 20;
            ctx.arc(impactX, impactY, 12, 0, Math.PI * 2);
            ctx.fill();
            ctx.shadowBlur = 0;
            
            // Draw labels
            ctx.fillStyle = 'white';
            ctx.font = '12px Rajdhani';
            ctx.fillText('IMPACT', impactX + 25, impactY);
            ctx.fillText('PITCHED', startX + 25, startY + 20);
        }
    }
    
    animateBall();
}

// Generate decision
async function generateDecision() {
    showScreen('decision');

    if (predictionPromise) {
        showToast('Waiting for model result...', 'info');
        lastPrediction = await predictionPromise;
        if (!lastPrediction) {
            showToast('Backend unreachable — showing demo result instead', 'error');
        }
    }

    animateDecision(lastPrediction);
}

// Human-readable labels for the LBW sub-model's raw class names
const LBW_LABELS = {
    OUTSIDE_LEG: 'Outside Leg', OUTSIDE_OFF: 'Outside Off', IN_LINE: 'In-line',
    OUTSIDE: 'Outside', HITTING: 'Hitting', MISSING: 'Missing'
};

// Builds the decision-screen content for an LBW appeal that has a full
// Pitching/Impact/Wickets breakdown from the LBW sub-model + rule engine.
function buildLbwBreakdownDisplay(prediction) {
    const b = prediction.lbw_breakdown;
    const isOut = b.final_decision === 'OUT';
    return {
        decision: isOut ? 'OUT' : 'NOT OUT',
        className: isOut ? 'out' : 'not-out',
        subText: b.reason,
        ballType: 'Legal',
        details: [
            `${b.pitching === 'OUTSIDE_LEG' ? '✗' : '✓'} Pitching: ${LBW_LABELS[b.pitching]}`,
            `${b.impact === 'OUTSIDE' ? '✗' : '✓'} Impact: ${LBW_LABELS[b.impact]}`,
            `${b.wickets === 'MISSING' ? '✗' : '✓'} Wickets: ${LBW_LABELS[b.wickets]}`,
            `ℹ Ball-type confidence: ${Math.round(prediction.confidence)}%`
        ],
        confidence: Math.round(
            (b.sub_confidences.pitching + b.sub_confidences.impact + b.sub_confidences.wickets) / 3
        )
    };
}

// Maps the model's ball-type class to what the decision screen shows.
// NOTE: the trained model classifies delivery TYPE (LBW-shaped / Legal /
// No ball / Wide) — it does not itself rule OUT vs NOT OUT for an LBW
// appeal (that needs pitching-line/impact/wicket-trajectory analysis).
// So an "LBW" prediction is shown as a flagged appeal for the umpire,
// not an automatic OUT.
const CLASS_DISPLAY = {
    'LBW': {
        decision: 'LBW APPEAL',
        className: 'out',
        subText: 'Possible LBW — flagged for umpire review',
        ballType: 'Legal',
        details: [
            '⚠ Delivery pattern matches LBW appeal',
            '⚠ Pitching line / impact / wicket-hitting not yet verified',
            '✓ Legal delivery confirmed',
            'ℹ Final OUT/NOT OUT call needs trajectory review'
        ]
    },
    'Legal Balls': {
        decision: 'LEGAL DELIVERY',
        className: 'not-out',
        subText: 'No violations detected',
        ballType: 'Legal',
        details: [
            '✓ Front foot behind crease',
            '✓ Within height and width limits',
            '✓ No LBW appeal pattern detected',
            '✓ Legal delivery confirmed'
        ]
    },
    'No balls': {
        decision: 'NO BALL',
        className: 'no-ball',
        subText: 'Illegal delivery detected',
        ballType: 'No Ball',
        details: [
            '✗ No-ball pattern detected',
            '⚠ Delivery invalidated',
            '✓ Free hit awarded',
            '✓ Original decision overturned'
        ]
    },
    'Wide Balls': {
        decision: 'WIDE BALL',
        className: 'no-ball',
        subText: 'Delivery outside batsman\'s reach',
        ballType: 'Wide',
        details: [
            '✗ Ball outside the wide guideline',
            '⚠ Delivery invalidated',
            '✓ Extra run awarded'
        ]
    }
};

// Animate decision display. `prediction` is the {predicted_class, confidence,
// probabilities} object from the backend, or null/undefined to fall back to
// a randomized demo (e.g. no backend connected / no video analyzed yet).
function animateDecision(prediction) {
    let decision, className, subText, ballType, detailLines, confidence;

    if (prediction && prediction.lbw_breakdown) {
        // Full Pitching/Impact/Wickets breakdown available -> real OUT/NOT OUT verdict
        const info = buildLbwBreakdownDisplay(prediction);
        decision = info.decision; className = info.className; subText = info.subText;
        ballType = info.ballType; detailLines = info.details; confidence = info.confidence;
    } else if (prediction && CLASS_DISPLAY[prediction.predicted_class]) {
        const info = CLASS_DISPLAY[prediction.predicted_class];
        decision = info.decision;
        className = info.className;
        subText = info.subText;
        ballType = info.ballType;
        detailLines = info.details;
        confidence = Math.round(prediction.confidence);
    } else {
        // Demo fallback (no video/backend available)
        const randomOutcome = Math.random();
        if (randomOutcome < 0.4) {
            decision = 'OUT'; className = 'out';
            subText = 'LBW - Leg Before Wicket (demo)'; ballType = 'Legal';
            detailLines = ['✓ Ball pitched in line with stumps', '✓ Impact point in-line',
                '✓ Ball trajectory hitting middle stump', '✗ No edge detected on Ultra Edge',
                '✓ Legal delivery confirmed'];
        } else if (randomOutcome < 0.8) {
            decision = 'NOT OUT'; className = 'not-out';
            subText = 'Ball Missing Stumps (demo)'; ballType = 'Legal';
            detailLines = ['✓ Ball pitched in line', '✗ Ball missing leg stump',
                '✗ Impact outside line', '✗ No edge detected', '✓ Legal delivery confirmed'];
        } else {
            decision = 'NO BALL'; className = 'no-ball';
            subText = 'Front Foot No Ball Detected (demo)'; ballType = 'No Ball';
            detailLines = ['✗ Front foot over crease line', '⚠ Delivery invalidated',
                '✓ Free hit awarded', '✓ Original decision overturned'];
        }
        confidence = 85 + Math.floor(Math.random() * 14);
    }

    const box = document.getElementById('finalDecisionBox');
    box.className = `final-decision ${className}`;
    document.getElementById('decisionText').textContent = decision;
    document.getElementById('decisionSubText').textContent = subText;
    document.getElementById('ballType').textContent = ballType;

    document.getElementById('decisionDetails').innerHTML =
        detailLines.map(line => `<p>${line}</p>`).join('');

    // Animate confidence bar
    setTimeout(() => {
        document.getElementById('confidenceFill').style.width = confidence + '%';
        document.getElementById('confidenceText').textContent = confidence + '%';
    }, 500);

    // Set date
    document.getElementById('decisionDate').textContent = new Date().toLocaleString();
}

// Modal functions
function showNoBallModal() {
    document.getElementById('modalTitle').textContent = 'No Ball Detection Setup';
    document.getElementById('modalContent').innerHTML = `
        <div class="form-group">
            <label class="form-label">Detection Sensitivity</label>
            <select class="form-select">
                <option>High (Recommended)</option>
                <option>Medium</option>
                <option>Low</option>
            </select>
        </div>
        <div class="form-group">
            <label class="form-label">Camera Angle</label>
            <select class="form-select">
                <option>Side-on View</option>
                <option>Behind Stumps</option>
                <option>Multi-Angle</option>
            </select>
        </div>
        <div class="form-group">
            <label class="form-label">Check Types</label>
            <div style="display: flex; flex-direction: column; gap: 0.5rem; margin-top: 0.5rem;">
                <label><input type="checkbox" checked> Front Foot No Ball</label>
                <label><input type="checkbox" checked> Height No Ball (Beamer)</label>
                <label><input type="checkbox" checked> Bowling Action</label>
                <label><input type="checkbox" checked> Wide Ball Detection</label>
            </div>
        </div>
    `;
    document.getElementById('modalOverlay').classList.add('active');
}

function showSettingsModal() {
    document.getElementById('modalTitle').textContent = 'System Settings';
    document.getElementById('modalContent').innerHTML = `
        <div class="form-group">
            <label class="form-label">Default Camera</label>
            <select class="form-select">
                <option>Main Camera</option>
                <option>Stump Cam</option>
                <option>Spider Cam</option>
            </select>
        </div>
        <div class="form-group">
            <label class="form-label">AI Model</label>
            <select class="form-select">
                <option>DRS Pro v3.0 (Recommended)</option>
                <option>DRS Standard v2.5</option>
                <option>DRS Lite v1.0</option>
            </select>
        </div>
        <div class="form-group">
            <label class="form-label">Frame Rate Analysis</label>
            <select class="form-select">
                <option>120 FPS (Ultra)</option>
                <option>60 FPS (High)</option>
                <option>30 FPS (Standard)</option>
            </select>
        </div>
    `;
    document.getElementById('modalOverlay').classList.add('active');
}

function closeModal() {
    document.getElementById('modalOverlay').classList.remove('active');
    showToast('Settings saved!', 'success');
}

document.getElementById('modalOverlay').addEventListener('click', (e) => {
    if (e.target === document.getElementById('modalOverlay')) {
        closeModal();
    }
});

// Save decision
function saveDecision() {
    showToast('Decision saved to database!', 'success');
}

// Download replay
function downloadReplay() {
    showToast('Preparing replay download...', 'info');
    setTimeout(() => {
        showToast('Replay downloaded successfully!', 'success');
    }, 2000);
}

// Share decision
function shareDecision() {
    showToast('Share link copied to clipboard!', 'success');
}

// Counter animation
function animateCounter(element, target) {
    let current = 0;
    const increment = target / 50;
    const timer = setInterval(() => {
        current += increment;
        if (current >= target) {
            element.textContent = target.toLocaleString();
            clearInterval(timer);
        } else {
            element.textContent = Math.floor(current).toLocaleString();
        }
    }, 30);
}

// Initialize counters on page load
window.addEventListener('load', () => {
    animateCounter(document.getElementById('totalDecisions'), 1247);
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeModal();
    }
    if (e.key === ' ' && currentScreen === 'upload') {
        const video = document.getElementById('uploadedVideo');
        if (video.paused) {
            video.play();
        } else {
            video.pause();
        }
        e.preventDefault();
    }
});