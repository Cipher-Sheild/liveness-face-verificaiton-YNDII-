# YIDIE Unified Biometric eKYC Platform
### Complete 1:1 Face Verification & Multi-Modal Presentation Attack Detection (Liveness)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-007ACC.svg)](https://developers.google.com/mediapipe)
[![ONNX Runtime](https://img.shields.io/badge/ONNXRuntime-1.16%2B-brightgreen.svg)](https://onnxruntime.ai/)

---

## Overview

The **YIDIE Unified eKYC Biometric Platform** provides a production-grade, offline-capable identity verification engine combining **Component 2a (1:1 Face Verification)** and **Component 2b (Active & Passive Liveness Detection)** into a single, cohesive service.

The platform prevents spoofing attacks (printed photos, screen replaying, 3D masks, digital injections) and verifies that a live human matches the biometric portrait in their ID card or passport—even across substantial changes in facial hair, age, lighting, and camera optics.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       YIDIE Unified Biometric Pipeline                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
     [ Document Photo ]                │                [ Live User ]
     (ID / Passport)                   │             (Webcam / Mobile)
             │                         │                     │
             ▼                         │                     ▼
┌─────────────────────────┐            │         ┌─────────────────────────┐
│ ID Card Card Auto-Crop  │            │         │ Active Liveness Session │
│  & Contrast Normalizer  │            │         │ (Blink, Turn L/R Chal.) │
└────────────┬────────────┘            │         └───────────┬─────────────┘
             │                         │                     │
             ▼                         │                     ▼
┌─────────────────────────┐            │         ┌─────────────────────────┐
│ MTCNN / Caffe SSD Face  │            │         │ Return-to-Center Lock   │
│   Alignment (112x112)   │            │         │ (Crisp Frontal Capture) │
└────────────┬────────────┘            │         └───────────┬─────────────┘
             │                         │                     │
             │                         │                     ▼
             │                         │         ┌─────────────────────────┐
             │                         │         │  Passive Anti-Spoofing  │
             │                         │         │  - MiniFASNet V1SE & V2 │
             │                         │         │  - FFT Moiré Patterns   │
             │                         │         │  - Specular Reflection  │
             │                         │         │  - Laplacian Sharpness  │
             │                         │         └───────────┬─────────────┘
             │                         │                     │
             │                         │              [Human Verified?]
             │                         │               │             │
             │                         │              YES            NO ──► [REJECT: Spoof Detected]
             │                         │               │
             └────────────────► ◄──────┴───────────────┘
                                │
                                ▼
                 ┌─────────────────────────────┐
                 │ AdaFace IR-50 Deep Features │
                 │ (MS1MV2 Margin-Adaptive Arc)│
                 └──────────────┬──────────────┘
                                │
                                ▼
                 ┌─────────────────────────────┐
                 │  Cosine Similarity Match    │
                 │   Thresholding & Calibrated │
                 │      Confidence Score       │
                 └──────────────┬──────────────┘
                                │
                                ▼
                 ┌─────────────────────────────┐
                 │   FINAL VERIFIED DECISION   │
                 │     (Approved / Rejected)   │
                 └─────────────────────────────┘
```

---

## Key Capabilities

### 1. Component 2b: Multi-Modal Liveness Detection
* **Active Challenge-Response:** Dynamic, randomized challenges (e.g., eye blink sequence, head yaw turn left, head yaw turn right) computed in real-time via 468-point 3D MediaPipe face mesh geometry.
* **Return-to-Center Guidance:** After completing the challenge, intelligent orientation tracking guides the subject back to a straight-ahead frontal pose before triggering capture.
* **Passive Anti-Spoofing Gate:**
  * **Dual MiniFASNet Neural Networks:** V1SE (80x80) and V2 (80x80) ONNX models running simultaneously to catch 2D printouts and screen replays.
  * **2D Fast Fourier Transform (FFT):** Analyzes high-frequency spectral artifacts to detect LCD/OLED display pixel grids and moiré patterns.
  * **Specular Glare & Reflection Detector:** Identifies unnatural flash reflections and hot spots produced by glass screens and glossy photographic paper.
  * **Laplacian Sharpness Check:** Guarantees captured biometric frames meet stringent quality and contrast standards without motion blur.

### 2. Component 2a: 1:1 AdaFace Verification
* **AdaFace IR-50 MS1MV2 Architecture:** Employs margin-adaptive loss based on image quality, providing state-of-the-art verification accuracy even on degraded or compressed ID document photos.
* **Document Card Auto-Crop:** Automatic edge detection, perspective warping, and contour isolation for national identity cards and passport booklets (`crop_document_card`).
* **Multi-Tier Facial Alignment:** Primary MTCNN 5-point landmark alignment with automatic fallback to an OpenCV Caffe SSD detector for heavily watermarked, holographic, or embossed identity credentials.
* **Calibrated Decision Engine:** Matches across demographic variances, facial hair changes (e.g., clean-shaven passport vs. full beard selfie), and aging with adaptive confidence scaling (88%–99% for genuine matches).

---

## Project Structure

```
liveness_plus_face_verification/
├── core/
│   ├── liveness/
│   │   ├── active_liveness.py       # Blink & head pose challenge state machine
│   │   ├── passive_liveness.py      # MiniFASNet, FFT moiré & reflection detectors
│   │   └── face_detector.py         # MediaPipe / Haar cascade detector wrapper
│   └── verification/
│       ├── adaface_wrapper.py       # AdaFace feature extraction & similarity matching
│       ├── face_aligner.py          # MTCNN 5-point affine transform aligner
│       └── AdaFace/                 # AdaFace model architecture & pretrained weights
│           ├── face_alignment/
│           ├── net.py
│           └── pretrained/
│               ├── .gitkeep
│               └── adaface_ir50_ms1mv2.ckpt  # (185 MB / 668 MB uncompressed weights)
├── models/                          # Pre-packaged inference models
│   ├── 2.7_80x80_MiniFASNetV2.onnx
│   ├── 4_0_0_80x80_MiniFASNetV1SE.onnx
│   ├── face_landmarker.task
│   ├── res10_300x300_ssd_iter_140000.caffemodel
│   └── deploy.prototxt
├── sample_docs/                     # Reference test documents
│   ├── ID.jpeg
│   ├── ID2.jpeg
│   └── passport.jpeg
├── static/                          # Web UI assets
│   ├── index.html                   # Unified biometric dashboard
│   ├── app.js                       # Real-time WebSocket telemetry HUD client
│   └── style.css
├── testing_photos/                  # Runtime storage for audit trails & captured selfies
│   ├── docs/
│   └── .gitkeep
├── download_models.py               # Automated model weights downloader & validator
├── server.py                        # Unified FastAPI + WebSocket server
├── requirements.txt                 # Python package dependencies
├── run_system.bat                   # 1-Click Windows Batch launcher
├── run_system.ps1                   # 1-Click Windows PowerShell launcher
└── README.md                        # Documentation
```

---

## Installation & Environment Setup

### Prerequisites
* **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS
* **Python:** Version 3.10 or 3.11 (recommended)
* **Hardware:** Modern multi-core CPU (GPU optional; CUDA will be automatically used if available)
* **Webcam:** Standard HD (720p or 1080p) webcam

---

### Step 1: Clone the Repository

```bash
git clone https://github.com/Cipher-Sheild/liveness-face-verificaiton-YNDII-.git
cd liveness-face-verificaiton-YNDII-
```

---

### Step 2: Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### Step 3: Install Dependencies

Install all dependencies using `pip`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on PyTorch (CUDA acceleration):**
> If you have an NVIDIA GPU and want GPU acceleration, install PyTorch with CUDA support:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```
> The system runs fast on CPU as well (~25-50ms inference time).

---

## Model Activation & Weight Downloads

The platform relies on pretrained deep learning models. While lightweight models (`models/*.onnx`, `models/*.caffemodel`, and `models/face_landmarker.task`) are included directly in the repository, the **AdaFace IR-50 Checkpoint** exceeds GitHub's 100MB limit and must be installed using either of the methods below:

### Models Summary

| Model | Size | Location | Source / Status |
|---|---|---|---|
| **AdaFace IR-50 MS1MV2** | ~185 MB (compressed) / 668 MB | `core/verification/AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt` | Downloaded via script or GDrive |
| **MediaPipe FaceLandmarker** | ~3.8 MB | `models/face_landmarker.task` | Pre-packaged / auto-downloaded |
| **MiniFASNet V1SE** | ~1.7 MB | `models/4_0_0_80x80_MiniFASNetV1SE.onnx` | Pre-packaged in repo |
| **MiniFASNet V2** | ~1.7 MB | `models/2.7_80x80_MiniFASNetV2.onnx` | Pre-packaged in repo |
| **Caffe SSD Face Detector** | ~10.6 MB | `models/res10_300x300_ssd_iter_140000.caffemodel` | Pre-packaged in repo |

---

### Option A: Automatic Download (Recommended)

Run the included verification and download utility:

```bash
python download_models.py
```

This script validates all checkpoints and downloads any missing files using `gdown` and Google Cloud Storage.

---

### Option B: Manual Download

If your environment has restricted network access:

1. Download the AdaFace IR-50 checkpoint from Google Drive:
   * **Direct Link:** [AdaFace IR-50 MS1MV2 Checkpoint (Google Drive)](https://drive.google.com/file/d/1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI/view?usp=sharing)
2. Place the downloaded `.ckpt` file at the following path:
   ```
   core/verification/AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt
   ```
3. Run the validator to verify file integrity:
   ```bash
   python download_models.py
   ```
   You should see:
   ```
   [OK] AdaFace IR-50 Checkpoint present (667.8 MB)
   [OK] MediaPipe FaceLandmarker present
   [OK] MiniFASNet V1SE present
   [OK] MiniFASNet V2 present
   [OK] SSD Face Detector present
   All Biometric Models Ready!
   ```

---

## Running the System

### 1. One-Click Launchers (Windows)

* **Batch Launcher:** Double-click `run_system.bat`
* **PowerShell Launcher:** Right-click and choose *Run with PowerShell* or run:
  ```powershell
  .\run_system.ps1
  ```

The launcher will verify model weights, spin up the Uvicorn server, and automatically open your default browser to `http://localhost:8000`.

---

### 2. Manual Command Line Launch

Activate your virtual environment and run:

```bash
python server.py --host 0.0.0.0 --port 8000
```

To enable live code auto-reloading during development:
```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

---

## User Interfaces & Verification Workflow

### 1. Interactive Web Dashboard (`http://localhost:8000`)

1. **Document Selection:** Select one of the provided pre-loaded sample documents (`ID.jpeg`, `ID2.jpeg`, `passport.jpeg`) or upload an image of your own national ID or passport.
2. **Start Verification:** Click **Start Verification** to initialize the webcam stream and WebSocket pipeline.
3. **Active Challenge Execution:** Follow the real-time on-screen HUD prompt:
   * **Blink Challenge:** Blink both eyes naturally (EAR drops below 0.22).
   * **Turn Left / Right:** Rotate head horizontally until the yaw angle crosses ±18°.
4. **Return-to-Center Lock:** The HUD guides you back to looking straight ahead. Once frontal alignment and image sharpness are satisfied, the camera automatically captures a high-resolution selfie.
5. **Passive Anti-Spoof Security Gate:** Background checks verify the face is real (not a screen, paper photo, or replay attack).
6. **1:1 Face Match:** AdaFace aligns both faces and computes cosine similarity.
7. **Decision Screen:** View the side-by-side comparison with similarity score, match verdict, and comprehensive anti-spoof telemetry breakdown.

---

## REST API & Mobile Integration

In addition to the Web UI, the server exposes high-performance REST endpoints for integration with mobile apps (React Native, Flutter, iOS, Android) and external microservices.

### `POST /api/v1/verify-identity`

Performs end-to-end verification (passive anti-spoofing + document card cropping + AdaFace 1:1 matching).

#### Mode A: Multipart Form Data
Send standard `multipart/form-data` with two image files:

```bash
curl -X POST "http://localhost:8000/api/v1/verify-identity" \
  -F "document=@/path/to/passport.jpg" \
  -F "live_selfie=@/path/to/selfie.jpg"
```

#### Mode B: Base64 JSON Payload (Recommended for Mobile / React Native / Expo)
Send raw JSON with Base64-encoded strings (with or without data URI prefix):

```json
POST /api/v1/verify-identity
Content-Type: application/json

{
  "document": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD...",
  "live_selfie": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD..."
}
```

#### Sample Successful Response (`200 OK`)
```json
{
  "status": "APPROVED",
  "decision": "VERIFIED",
  "liveness": {
    "result": "live",
    "confidence": 0.884,
    "details": {
      "model_score": 0.9982,
      "blur_score": 0.6214,
      "moire_score": 0.9412,
      "reflection_score": 0.0
    }
  },
  "verification": {
    "similarity": 0.4851,
    "threshold": 0.34,
    "verdict": "VERIFIED",
    "confidence": 0.932
  },
  "timestamp": "2026-11-02T01:15:00.000Z"
}
```

#### Sample Spoof Rejection (`200 OK`)
```json
{
  "status": "REJECTED",
  "decision": "SPOOF_DETECTED",
  "liveness": {
    "result": "spoof",
    "confidence": 0.985,
    "details": {
      "model_score": 0.021,
      "blur_score": 0.412,
      "moire_score": 0.281,
      "reflection_score": 0.892
    }
  },
  "verification": {
    "similarity": 0.0,
    "threshold": 0.34,
    "verdict": "SKIPPED_DUE_TO_SPOOF",
    "confidence": 0.0
  }
}
```

---

### Additional Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Service health status, GPU/CPU execution mode, and model readiness |
| `POST` | `/api/upload-document` | Upload custom ID/passport document for Web UI testing |
| `GET` | `/api/sample-doc/{filename}` | Retrieve sample document images (`ID.jpeg`, `ID2.jpeg`, `passport.jpeg`) |
| `GET` | `/api/photos/{filepath}` | Retrieve audit verification snapshots |
| `WS` | `/ws/ekyc` | Full-duplex WebSocket for live webcam streaming and HUD telemetry |

---

## Decision Thresholds & Tuning

The verification and anti-spoof thresholds are configured in `server.py`:

```python
# Verification Thresholds (AdaFace IR-50 Cosine Similarity)
MATCH_THRESHOLD = 0.34          # Minimum similarity score for a VERIFIED decision
UNCERTAIN_THRESHOLD = 0.28      # Range [0.28, 0.34) flagged for secondary review

# Passive Anti-Spoofing Thresholds
PASSIVE_THRESHOLD = 0.50        # MiniFASNet combined liveness probability
MOIRE_THRESHOLD = 0.40          # FFT high-frequency peak threshold
REFLECTION_THRESHOLD = 0.60     # Specular reflection tolerance
```

* **True User Matches:** Typically score between `0.36` and `0.85` cosine similarity across significant age and hairstyle differences.
* **Impostors / Different People:** Typically score below `0.20` cosine similarity.

---

## Security & Privacy

* **Zero Cloud Dependence:** All neural network inference, feature extraction, and biometric template comparisons occur **100% locally** on the host machine. No facial templates or photos leave the server.
* **Audit Trail Protection:** Live captured selfies and uploaded documents are stored in `testing_photos/`, which is strictly excluded from version control via `.gitignore` to prevent leaking personal identification credentials.
* **Replay & Presentation Attack Resistance:** Active challenges change dynamically every session, preventing attackers from replaying pre-recorded videos.

---

## License

This project is licensed under the MIT License. Pretrained models (AdaFace, MiniFASNet, MediaPipe) belong to their respective researchers and authors.
