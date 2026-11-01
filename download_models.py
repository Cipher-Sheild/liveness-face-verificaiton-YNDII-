"""
Model Downloader & Verifier for Unified eKYC Biometric System.
Automatically downloads missing model weights (AdaFace IR-50, MediaPipe, MiniFASNet).
"""

import os
import sys
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

ADAFACE_DIR = os.path.join(PROJECT_ROOT, "core", "verification", "AdaFace", "pretrained")
ADAFACE_CKPT = os.path.join(ADAFACE_DIR, "adaface_ir50_ms1mv2.ckpt")
ADAFACE_GDRIVE_ID = "1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI"

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LANDMARKER_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")
LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

def download_file(url: str, dest_path: str):
    print(f"Downloading: {url} -> {dest_path}")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    urllib.request.urlretrieve(url, dest_path)
    print("Downloaded successfully.")

def ensure_adaface_checkpoint():
    os.makedirs(ADAFACE_DIR, exist_ok=True)
    if os.path.exists(ADAFACE_CKPT) and os.path.getsize(ADAFACE_CKPT) > 100 * 1024 * 1024:
        print(f"[OK] AdaFace IR-50 Checkpoint present ({round(os.path.getsize(ADAFACE_CKPT)/1024/1024, 1)} MB): {ADAFACE_CKPT}")
        return True

    print("\n" + "=" * 60)
    print("  Downloading AdaFace IR-50 MS1MV2 Pretrained Checkpoint (185 MB)...")
    print("=" * 60)

    try:
        import gdown
        output = gdown.download(id=ADAFACE_GDRIVE_ID, output=ADAFACE_CKPT, quiet=False)
        if output and os.path.exists(ADAFACE_CKPT):
            print(f"[OK] AdaFace weights downloaded to: {ADAFACE_CKPT}")
            return True
    except Exception as e:
        print(f"[WARNING] Automatic gdown failed: {e}")

    print("\n[MANUAL DOWNLOAD REQUIRED]")
    print("Please download the checkpoint manually from Google Drive:")
    print("  Link: https://drive.google.com/file/d/1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI/view?usp=sharing")
    print(f"  Destination: {ADAFACE_CKPT}\n")
    return False

def ensure_mediapipe_landmarker():
    os.makedirs(MODELS_DIR, exist_ok=True)
    if os.path.exists(LANDMARKER_PATH) and os.path.getsize(LANDMARKER_PATH) > 1024 * 1024:
        print(f"[OK] MediaPipe FaceLandmarker present: {LANDMARKER_PATH}")
        return True

    print(f"Downloading MediaPipe FaceLandmarker task...")
    try:
        download_file(LANDMARKER_URL, LANDMARKER_PATH)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download FaceLandmarker: {e}")
        return False

def check_all_models():
    print("=" * 60)
    print("  YIDIE Unified eKYC — Model Verification & Setup")
    print("=" * 60)
    
    ok1 = ensure_adaface_checkpoint()
    ok2 = ensure_mediapipe_landmarker()

    minifasnet1 = os.path.join(MODELS_DIR, "4_0_0_80x80_MiniFASNetV1SE.onnx")
    minifasnet2 = os.path.join(MODELS_DIR, "2.7_80x80_MiniFASNetV2.onnx")
    caffemodel = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

    for p, name in [(minifasnet1, "MiniFASNet V1SE"), (minifasnet2, "MiniFASNet V2"), (caffemodel, "SSD Face Detector")]:
        if os.path.exists(p):
            print(f"[OK] {name} present: {p}")
        else:
            print(f"[MISSING] {name} missing at {p}")

    print("=" * 60)
    if ok1 and ok2:
        print("All Biometric Models Ready!")
    else:
        print("Please check the warnings above before starting the server.")
    print("=" * 60)

if __name__ == "__main__":
    check_all_models()
