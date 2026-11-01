// YIDIE Full Biometric eKYC Platform
let mediaStream = null;
let ws = null;
let frameInterval = null;
let timerInterval = null;
let remainingTime = 5.0;
let currentDocFile = 'ID.jpeg'; // default sample

const video = document.getElementById('webcam-video');
const canvas = document.getElementById('capture-canvas');
const ctx = canvas.getContext('2d');

// Web Audio API Synth Chimes
const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
function playChime(freq = 587.33, type = 'sine', duration = 0.15) {
  try {
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = type;
    osc.frequency.value = freq;
    gain.gain.setValueAtTime(0.1, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + duration);
  } catch (e) {}
}

// Sample Document Selector
function selectSampleDoc(filename, chipId) {
  currentDocFile = filename;
  document.querySelectorAll('.sample-chip').forEach(c => c.classList.remove('active'));
  document.getElementById(chipId).classList.add('active');

  document.getElementById('doc-filename-display').innerText = `${filename} (Sample Selected)`;
  document.getElementById('doc-thumb-preview').src = `/api/sample-doc/${filename}`;
  playChime(440, 'triangle', 0.1);
}

// Custom Document Upload
async function handleCustomDocUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append('document', file);

  try {
    const res = await fetch('/api/upload-document', { method: 'POST', body: formData });
    const data = await res.json();
    currentDocFile = data.filename;
    document.querySelectorAll('.sample-chip').forEach(c => c.classList.remove('active'));
    document.getElementById('doc-filename-display').innerText = `${file.name} (Uploaded)`;
    document.getElementById('doc-thumb-preview').src = data.doc_url;
    playChime(523.25, 'sine', 0.15);
  } catch (err) {
    alert('Failed to upload document: ' + err);
  }
}

// Camera Management
async function initCamera() {
  if (mediaStream) return;
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' }
    });
    video.srcObject = mediaStream;
  } catch (err) {
    console.error('Camera error:', err);
    alert('Camera permission required.');
  }
}

function stopCamera() {
  if (mediaStream) {
    mediaStream.getTracks().forEach(t => t.stop());
    mediaStream = null;
    video.srcObject = null;
  }
}

// Full eKYC Initiation
async function startFullEkycVerification() {
  await initCamera();
  resetUI();
  playChime(440, 'triangle', 0.1);

  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/ekyc`;
  
  ws = new WebSocket(wsUrl);
  document.getElementById('btn-start').disabled = true;
  document.getElementById('btn-start').innerText = 'Biometric Scan in Progress...';
  document.getElementById('scan-line').style.display = 'block';

  ws.onopen = () => {
    // Send initial configuration with chosen document
    ws.send(JSON.stringify({
      action: 'init',
      document: currentDocFile
    }));
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.type === 'challenge_init') {
      setupChallenge(data.challenge);
      startFrameStreaming();
    } else if (data.type === 'telemetry') {
      updateHUD(data);
    } else if (data.type === 'action_passed') {
      playChime(880, 'sine', 0.25);
      showReturnToCenter(data.message);
    } else if (data.type === 'final_verdict') {
      displayFinalVerdict(data);
      stopVerification();
    }
  };

  ws.onerror = (err) => {
    console.error('WebSocket error:', err);
    stopVerification();
  };
}

function stopVerification() {
  if (ws) {
    ws.close();
    ws = null;
  }
  clearInterval(frameInterval);
  clearInterval(timerInterval);
  resetUI();
}

function resetUI() {
  document.getElementById('challenge-banner').style.display = 'none';
  document.getElementById('scan-line').style.display = 'none';
  document.getElementById('oval-guide').classList.remove('active');
  document.getElementById('btn-start').disabled = false;
  document.getElementById('btn-start').innerHTML = '<span>🛡️ Run Full eKYC Verification</span>';
}

function setupChallenge(challenge) {
  const banner = document.getElementById('challenge-banner');
  const icon = document.getElementById('challenge-icon');
  const msg = document.getElementById('challenge-msg');
  const timer = document.getElementById('challenge-timer');

  banner.style.display = 'flex';
  banner.style.background = 'rgba(15, 23, 42, 0.9)';
  
  if (challenge === 'turn_right') {
    icon.innerText = '👉';
    msg.innerText = 'Please turn your head to YOUR RIGHT';
  } else if (challenge === 'turn_left') {
    icon.innerText = '👈';
    msg.innerText = 'Please turn your head to YOUR LEFT';
  } else if (challenge === 'blink') {
    icon.innerText = '👁️';
    msg.innerText = 'Please BLINK your eyes naturally';
  }

  remainingTime = 7.0;
  timer.innerText = `${remainingTime.toFixed(1)}s`;
  clearInterval(timerInterval);
  timerInterval = setInterval(() => {
    remainingTime -= 0.1;
    if (remainingTime > 0) {
      timer.innerText = `${remainingTime.toFixed(1)}s`;
    } else {
      timer.innerText = '0.0s';
      clearInterval(timerInterval);
    }
  }, 100);
}

function showReturnToCenter(message) {
  const banner = document.getElementById('challenge-banner');
  const icon = document.getElementById('challenge-icon');
  const msg = document.getElementById('challenge-msg');
  const timer = document.getElementById('challenge-timer');

  document.getElementById('oval-guide').classList.add('active');
  banner.style.background = 'rgba(16, 185, 129, 0.9)';
  icon.innerText = '🎯';
  msg.innerText = message || 'Challenge Passed! Look straight at the camera...';
  timer.innerText = 'Locking Portrait';
}

function startFrameStreaming() {
  canvas.width = 480;
  canvas.height = 360;

  frameInterval = setInterval(() => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    
    ctx.save();
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    ctx.restore();

    const base64Data = canvas.toDataURL('image/jpeg', 0.7);
    ws.send(JSON.stringify({ frame: base64Data }));
  }, 100);
}

function updateHUD(t) {
  const ratio = t.turn_ratio !== undefined ? t.turn_ratio : 0.5;
  document.getElementById('hud-yaw-val').innerText = ratio.toFixed(2);
  const yawPercent = Math.min(Math.max(ratio * 100, 5), 95);
  document.getElementById('yaw-bar').style.width = `${yawPercent}%`;
  
  if (ratio < 0.43) {
    document.getElementById('hud-yaw-label').innerText = 'Right';
  } else if (ratio > 0.57) {
    document.getElementById('hud-yaw-label').innerText = 'Left';
  } else {
    document.getElementById('hud-yaw-label').innerText = 'Center';
  }

  const ear = t.avg_ear !== undefined ? t.avg_ear : 0.3;
  document.getElementById('hud-ear-val').innerText = ear.toFixed(2);
  const earPercent = Math.min(Math.max((ear / 0.4) * 100, 5), 100);
  document.getElementById('ear-bar').style.width = `${earPercent}%`;
  document.getElementById('hud-ear-label').innerText = ear < 0.20 ? 'Closed' : 'Open';

  const sharpness = t.sharpness !== undefined ? Math.round(t.sharpness) : 100;
  document.getElementById('hud-blur-val').innerText = sharpness;
  const blurPercent = Math.min(Math.max((sharpness / 200) * 100, 10), 100);
  document.getElementById('blur-bar').style.width = `${blurPercent}%`;
  document.getElementById('hud-blur-label').innerText = sharpness < 35 ? 'Blurry' : 'Sharp';

  // Handle Real-Time Warning Alert for Wrong Direction
  const banner = document.getElementById('challenge-banner');
  const msg = document.getElementById('challenge-msg');
  if (t.warning) {
    banner.style.background = 'rgba(239, 68, 68, 0.95)';
    msg.innerText = t.warning;
    playChime(260, 'sawtooth', 0.1);
  }
}

// Render Final Decision & Side-by-Side Comparison
function displayFinalVerdict(data) {
  const verdictBox = document.getElementById('verdict-box');
  const verdictIcon = document.getElementById('verdict-icon');
  const verdictTitle = document.getElementById('verdict-title');
  const verdictSub = document.getElementById('verdict-sub');

  const liveness = data.liveness || {};
  const verification = data.face_verification || {};

  const isLivenessLive = liveness.result === 'live';
  const isMatch = verification.verdict === 'VERIFIED';
  const isUncertain = verification.verdict === 'UNCERTAIN';
  const isApproved = isLivenessLive && isMatch;

  if (isApproved) {
    playChime(659.25, 'sine', 0.4);
    verdictBox.className = 'verdict-box approved';
    verdictIcon.innerText = '🛡️';
    verdictTitle.innerText = 'eKYC APPROVED';
    verdictSub.innerText = 'Live human confirmed & Identity matched with ID Document.';
  } else if (!isLivenessLive) {
    playChime(220, 'sawtooth', 0.4);
    verdictBox.className = 'verdict-box rejected';
    verdictIcon.innerText = '❌';
    const reason = (liveness.details?.attack_type || '') + ' ' + (liveness.details?.reason || '');
    if (reason.includes('Timeout')) {
      verdictTitle.innerText = 'REJECTED: CHALLENGE TIMEOUT';
      verdictSub.innerText = 'User did not complete required movement within 7 seconds.';
    } else if (reason.includes('Oppos') || reason.includes('Instruction')) {
      verdictTitle.innerText = 'REJECTED: OPPOSING ACTION (ATTACK)';
      verdictSub.innerText = liveness.layers?.active || 'User performed the opposite of the instructed movement.';
    } else {
      verdictTitle.innerText = 'REJECTED: SPOOF ATTACK';
      verdictSub.innerText = 'Failed presentation attack check. Face verification cancelled.';
    }
  } else if (isUncertain) {
    playChime(440, 'triangle', 0.4);
    verdictBox.className = 'verdict-box uncertain';
    verdictIcon.innerText = '⚠️';
    verdictTitle.innerText = 'UNCERTAIN: BORDERLINE MATCH';
    verdictSub.innerText = `Live person confirmed, but facial similarity (${(verification.similarity * 100).toFixed(1)}%) is in the borderline zone (30% - 40%). Possible beard, age gap, or low-res passport photo.`;
  } else {
    playChime(311.13, 'sawtooth', 0.4);
    verdictBox.className = 'verdict-box rejected';
    verdictIcon.innerText = '❌';
    verdictTitle.innerText = 'REJECTED: FACE MISMATCH';
    verdictSub.innerText = 'Live person detected, but face does not match ID document portrait.';
  }

  // Populate Side-by-Side Comparison Box
  const comparisonBox = document.getElementById('comparison-box');
  const avatarDoc = document.getElementById('avatar-doc');
  const avatarSelfie = document.getElementById('avatar-selfie');
  const simText = document.getElementById('similarity-score-text');

  avatarDoc.src = data.doc_image_url || '/api/sample-doc/' + currentDocFile;
  avatarSelfie.src = data.selfie_url || '';
  
  if (verification.similarity !== undefined) {
    simText.innerText = `${(verification.similarity * 100).toFixed(1)}%`;
    if (isMatch) {
      simText.className = 'similarity-badge match';
      avatarDoc.className = 'face-avatar match';
      avatarSelfie.className = 'face-avatar match';
    } else if (isUncertain) {
      simText.className = 'similarity-badge uncertain';
      avatarDoc.className = 'face-avatar';
      avatarSelfie.className = 'face-avatar';
    } else {
      simText.className = 'similarity-badge mismatch';
      avatarDoc.className = 'face-avatar';
      avatarSelfie.className = 'face-avatar';
    }
  } else {
    simText.innerText = 'N/A';
    simText.className = 'similarity-badge mismatch';
  }
  comparisonBox.style.display = 'flex';

  // Fill Metrics
  const d = liveness.details || {};
  document.getElementById('val-active-gate').innerText = liveness.layers?.active || (isLivenessLive ? 'Passed' : 'Failed');
  document.getElementById('val-model-score').innerText = d.model_score !== undefined ? d.model_score.toFixed(4) : '--';
  document.getElementById('val-moire-score').innerText = d.moire_score !== undefined ? d.moire_score.toFixed(4) : '--';
  document.getElementById('val-refl-score').innerText = d.reflection_score !== undefined ? d.reflection_score.toFixed(4) : '--';
  
  const simStr = verification.similarity !== undefined ? `Score: ${verification.similarity.toFixed(4)} (${verification.verdict})` : 'Skipped (Liveness Failed)';
  document.getElementById('val-adaface-match').innerText = simStr;
  document.getElementById('val-final-decision').innerText = isApproved ? 'APPROVED ✅' : 'REJECTED ❌';
}

window.addEventListener('DOMContentLoaded', () => {
  initCamera();
  document.getElementById('doc-thumb-preview').src = `/api/sample-doc/${currentDocFile}`;
});
