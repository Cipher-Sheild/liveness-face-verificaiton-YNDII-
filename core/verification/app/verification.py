"""Document portrait to liveness face verification API."""

import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADAFACE_DIR = PROJECT_ROOT / "AdaFace"
if str(ADAFACE_DIR) not in sys.path:
    sys.path.insert(0, str(ADAFACE_DIR))

from inference import cosine_similarity, get_embedding, load_pretrained_model


MATCH_THRESHOLD = 0.34
UNCERTAIN_THRESHOLD = 0.25


class FaceVerifier:
    """Load AdaFace once, then compare document and liveness images.

    The document argument is the full uploaded ID/passport image. AdaFace's
    MTCNN alignment locates and crops the face before embedding it.
    """

    def __init__(self, model=None):
        self.model = model if model is not None else load_pretrained_model()

    def verify(self, document_image, live_face_image):
        document_image = Path(document_image)
        live_face_image = Path(live_face_image)
        for label, path in (("Document image", document_image),
                            ("Liveness face image", live_face_image)):
            if not path.is_file():
                raise FileNotFoundError(f"{label} not found: {path}")

        started = time.perf_counter()
        document_embedding = get_embedding(self.model, str(document_image))
        document_seconds = time.perf_counter() - started
        if document_embedding is None:
            raise ValueError(
                "No face could be detected in the document. "
                "Check that the portrait is visible and the document is upright."
            )

        started = time.perf_counter()
        live_embedding = get_embedding(self.model, str(live_face_image))
        live_seconds = time.perf_counter() - started
        if live_embedding is None:
            raise ValueError("No face could be detected in the liveness image.")

        similarity = float(cosine_similarity(document_embedding, live_embedding))
        verdict, confidence = _decision(similarity)
        return {
            "similarity": round(similarity, 4),
            "threshold": MATCH_THRESHOLD,
            "verdict": verdict,
            "confidence": round(confidence, 4),
            "timing_seconds": {
                "document": round(document_seconds, 3),
                "liveness_face": round(live_seconds, 3),
                "total": round(document_seconds + live_seconds, 3),
            },
        }


def _decision(similarity):
    if similarity >= MATCH_THRESHOLD:
        verdict = "VERIFIED"
        margin = max(0.001, 1.0 - MATCH_THRESHOLD)
        ratio = (similarity - MATCH_THRESHOLD) / margin
        confidence = 0.88 + 0.11 * min(1.0, ratio * 2.5)
    elif similarity >= UNCERTAIN_THRESHOLD:
        verdict = "UNCERTAIN"
        diff = MATCH_THRESHOLD - UNCERTAIN_THRESHOLD
        ratio = (similarity - UNCERTAIN_THRESHOLD) / max(0.001, diff)
        confidence = 0.55 + 0.25 * ratio
    else:
        verdict = "NOT_VERIFIED"
        confidence = max(0.02, 0.40 * (similarity / max(0.001, UNCERTAIN_THRESHOLD)))
    return verdict, round(max(0.0, min(1.0, confidence)), 4)
