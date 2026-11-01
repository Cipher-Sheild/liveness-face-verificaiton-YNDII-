import os
import torch
import numpy as np
from PIL import Image

import net
from face_alignment import align


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(
    os.path.dirname(BASE_DIR),
    "data"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "pretrained",
    "adaface_ir50_ms1mv2.ckpt"
)


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"[AdaFace] Device: {DEVICE}")


# ============================================================
# LOAD ADAFACE
# ============================================================

def load_pretrained_model():

    print("[AdaFace] Loading model...")

    model = net.build_model("ir_50")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("model."):
            key = key[6:]

        model_state_dict[key] = value

    missing_keys, unexpected_keys = model.load_state_dict(
        model_state_dict,
        strict=False
    )

    print("[AdaFace] Model weights loaded successfully.")

    if unexpected_keys:
        print(
            "[AdaFace] Ignored checkpoint keys:",
            unexpected_keys
        )

    model = model.to(DEVICE)
    model.eval()

    print("[AdaFace] Model ready.")

    return model


# ============================================================
# FACE DETECTION + ALIGNMENT
# ============================================================

def align_face(image_path):

    print(
        f"[MTCNN] Detecting face: "
        f"{os.path.basename(image_path)}"
    )

    image = Image.open(image_path).convert("RGB")

    try:

        # AdaFace's MTCNN alignment
        bboxes, faces = align.mtcnn_model.align_multi(
            image,
            limit=1
        )

    except Exception as error:

        print(
            f"[MTCNN] Error processing "
            f"{os.path.basename(image_path)}"
        )

        print(error)

        return None

    if faces is None or len(faces) == 0:
        # Fallback for ID documents: Use OpenCV SSD to crop face bounding box first
        try:
            import cv2
            import sys
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            if root_dir not in sys.path:
                sys.path.insert(0, root_dir)
            from core.liveness.face_detector import get_face_detector, crop_face
            det = get_face_detector()
            cv_img = cv2.imread(image_path)
            if cv_img is not None:
                bboxes = det.get_bboxes(cv_img)
                if bboxes:
                    cropped = crop_face(cv_img, bboxes[0], margin=0.5)
                    crop_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                    bboxes_sub, faces_sub = align.mtcnn_model.align_multi(crop_pil, limit=1)
                    if faces_sub and len(faces_sub) > 0:
                        print(f'[MTCNN+SSD Fallback] Document face detected and aligned -> {faces_sub[0].size}')
                        return faces_sub[0]
        except Exception as e:
            print(f'[MTCNN Fallback Warning] {e}')

        print(
            f"[MTCNN] No face detected in "
            f"{os.path.basename(image_path)}"
        )

        return None

    face = faces[0]

    print(
        f"[MTCNN] Face detected and aligned "
        f"-> {face.size}"
    )

    return face


# ============================================================
# FACE -> ADAFACE TENSOR
# ============================================================

def face_to_tensor(face):

    # Make sure it is RGB
    face = face.convert("RGB")

    # MTCNN should already produce 112x112.
    # Resize again as a safety check.
    face = face.resize((112, 112))

    # PIL -> NumPy
    image = np.array(face)

    # RGB -> BGR
    image = image[:, :, ::-1]

    # Convert to float32
    image = image.astype(np.float32)

    # Normalize to [-1, 1]
    image = (image / 255.0 - 0.5) / 0.5

    # HWC -> CHW
    image = image.transpose(2, 0, 1)

    # NumPy -> PyTorch
    tensor = torch.from_numpy(
        image.copy()
    ).float()

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    return tensor


# ============================================================
# GET FACE EMBEDDING
# ============================================================

def get_embedding(model, image_path):

    # --------------------------------------------------------
    # Detect and align face
    # --------------------------------------------------------

    face = align_face(image_path)

    if face is None:
        return None

    # --------------------------------------------------------
    # Convert aligned face to tensor
    # --------------------------------------------------------

    tensor = face_to_tensor(face)

    tensor = tensor.to(DEVICE)

    # --------------------------------------------------------
    # AdaFace inference
    # --------------------------------------------------------

    with torch.no_grad():

        feature, norm = model(tensor)

    # --------------------------------------------------------
    # L2 normalize embedding
    # --------------------------------------------------------

    feature = torch.nn.functional.normalize(
        feature,
        p=2,
        dim=1
    )

    return feature


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    embedding1,
    embedding2
):

    similarity = torch.nn.functional.cosine_similarity(
        embedding1,
        embedding2,
        dim=1
    )

    return similarity.item()


# ============================================================
# VERIFY TWO IMAGES
# ============================================================

def verify(
    model,
    document_path,
    selfie_path
):

    print()
    print("[Verification] Document face")

    document_embedding = get_embedding(
        model,
        document_path
    )

    if document_embedding is None:
        return None

    print()
    print("[Verification] Selfie face")

    selfie_embedding = get_embedding(
        model,
        selfie_path
    )

    if selfie_embedding is None:
        return None

    # --------------------------------------------------------
    # Compare embeddings
    # --------------------------------------------------------

    similarity = cosine_similarity(
        document_embedding,
        selfie_embedding
    )

    return similarity


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("ADAFACE FACE VERIFICATION TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Load AdaFace
    # --------------------------------------------------------

    model = load_pretrained_model()

    print()

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not os.path.exists(DATA_DIR):

        print(
            f"ERROR: Data directory not found:\n"
            f"{DATA_DIR}"
        )

        exit(1)

    # --------------------------------------------------------
    # Find people
    # --------------------------------------------------------

    people = sorted(
        [
            folder
            for folder in os.listdir(DATA_DIR)
            if folder.startswith("person_")
            and os.path.isdir(
                os.path.join(DATA_DIR, folder)
            )
        ]
    )

    if not people:

        print(
            "ERROR: No person folders found."
        )

        exit(1)

    print(
        f"[AdaFace] Found {len(people)} people."
    )

    print()

    # --------------------------------------------------------
    # Temporary threshold
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # This is NOT the final threshold.
    #
    # We will calibrate the threshold after collecting
    # genuine and impostor similarity scores.
    #
    # --------------------------------------------------------

    THRESHOLD = 0.40

    results = []

    # ========================================================
    # PROCESS PEOPLE
    # ========================================================

    for person in people:

        print("-" * 60)
        print(f"Checking {person}")

        person_dir = os.path.join(
            DATA_DIR,
            person
        )

        document_path = os.path.join(
            person_dir,
            "document.jpg"
        )

        selfie_path = os.path.join(
            person_dir,
            "selfie.jpg"
        )

        # ----------------------------------------------------
        # Check document
        # ----------------------------------------------------

        if not os.path.exists(document_path):

            print(
                "ERROR: document.jpg not found"
            )

            continue

        # ----------------------------------------------------
        # Check selfie
        # ----------------------------------------------------

        if not os.path.exists(selfie_path):

            print(
                "ERROR: selfie.jpg not found"
            )

            continue

        # ----------------------------------------------------
        # Verification
        # ----------------------------------------------------

        try:

            similarity = verify(
                model,
                document_path,
                selfie_path
            )

            if similarity is None:

                print(
                    "Result: FACE DETECTION FAILED"
                )

                continue

            # ------------------------------------------------
            # Decision
            # ------------------------------------------------

            if similarity >= THRESHOLD:

                result = "VERIFIED"

            else:

                result = "NOT VERIFIED"

            # ------------------------------------------------
            # Output
            # ------------------------------------------------

            print()
            print(
                f"Similarity : {similarity:.4f}"
            )

            print(
                f"Threshold  : {THRESHOLD:.2f}"
            )

            print(
                f"Result     : {result}"
            )

            results.append(
                {
                    "person": person,
                    "similarity": similarity,
                    "result": result
                }
            )

        except Exception as error:

            print()
            print(
                f"ERROR processing {person}:"
            )

            print(error)

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    if not results:

        print("No verification results.")

    else:

        for result in results:

            print(
                f"{result['person']:12} "
                f"Similarity: "
                f"{result['similarity']:.4f}   "
                f"{result['result']}"
            )

    print("=" * 60)

    print()
    print("Verification test completed.")
