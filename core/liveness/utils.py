import os
import urllib.request
from datetime import datetime
import cv2

def download_file(url, filepath):
    """Downloads a file from a URL if it doesn't already exist."""
    if not os.path.exists(filepath):
        print(f"Downloading {os.path.basename(filepath)}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading {filepath}: {e}")
            raise

def get_project_root():
    """Returns the absolute path to the project root (liveness_plus_face_verification/)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(current_dir)
    if os.path.exists(os.path.join(parent, "models")):
        return parent
    grandparent = os.path.dirname(parent)
    if os.path.exists(os.path.join(grandparent, "models")):
        return grandparent
    return grandparent if grandparent else parent

def save_testing_photo(image, prefix="photo", status=""):
    """Saves an image into testing_photos directory with timestamp and status."""
    if image is None:
        return None
    photos_dir = os.path.join(get_project_root(), "testing_photos")
    os.makedirs(photos_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    status_str = f"_{status}" if status else ""
    filename = f"{prefix}_{timestamp}{status_str}.jpg"
    filepath = os.path.join(photos_dir, filename)
    cv2.imwrite(filepath, image)
    print(f"[SAVED] Photo recorded from camera saved to: {filepath}")
    return filepath

