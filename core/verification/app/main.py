import json
import sys
import time
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

# face-verification/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# AdaFace directory
ADAFACE_DIR = PROJECT_ROOT / "AdaFace"

# Add AdaFace to Python path
if str(ADAFACE_DIR) not in sys.path:
    sys.path.insert(0, str(ADAFACE_DIR))


# ============================================================
# IMPORT ADAFACE
# ============================================================

from inference import (
    load_pretrained_model,
    get_embedding,
    cosine_similarity,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RESULTS_FILE = OUTPUT_DIR / "results.json"


# ============================================================
# THRESHOLDS
# ============================================================

# Temporary threshold.
# We will calibrate this properly later using genuine
# and impostor pairs.

MATCH_THRESHOLD = 0.40
UNCERTAIN_THRESHOLD = 0.30


# ============================================================
# DECISION
# ============================================================

def make_decision(similarity):

    if similarity >= MATCH_THRESHOLD:

        verdict = "VERIFIED"

        confidence = 0.50 + (
            0.50
            * (similarity - MATCH_THRESHOLD)
            / (1.0 - MATCH_THRESHOLD)
        )

    elif similarity >= UNCERTAIN_THRESHOLD:

        verdict = "UNCERTAIN"

        confidence = 0.50

    else:

        verdict = "NOT_VERIFIED"

        confidence = 0.50 + (
            0.50
            * (UNCERTAIN_THRESHOLD - similarity)
            / UNCERTAIN_THRESHOLD
        )

    confidence = max(
        0.0,
        min(1.0, confidence)
    )

    return verdict, float(confidence)


# ============================================================
# PROCESS ONE PERSON
# ============================================================

def process_person(model, person_dir):

    person_id = person_dir.name

    document_path = person_dir / "document.jpg"
    selfie_path = person_dir / "selfie.jpg"

    print("-" * 60)
    print(f"Checking {person_id}")

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not document_path.exists():

        raise FileNotFoundError(
            f"Document not found: {document_path}"
        )

    if not selfie_path.exists():

        raise FileNotFoundError(
            f"Selfie not found: {selfie_path}"
        )

    # --------------------------------------------------------
    # DOCUMENT
    # --------------------------------------------------------

    print()
    print("[Verification] Document face")

    document_start = time.perf_counter()
    document_embedding = get_embedding(
        model,
        str(document_path)
    )
    document_elapsed = time.perf_counter() - document_start
    print(f"[Timing] Document image processed in {document_elapsed:.3f} seconds")

    if document_embedding is None:

        raise RuntimeError(
            "No face detected in document image."
        )

    # --------------------------------------------------------
    # SELFIE
    # --------------------------------------------------------

    print()
    print("[Verification] Selfie face")

    selfie_start = time.perf_counter()
    selfie_embedding = get_embedding(
        model,
        str(selfie_path)
    )
    selfie_elapsed = time.perf_counter() - selfie_start
    print(f"[Timing] Selfie image processed in {selfie_elapsed:.3f} seconds")

    if selfie_embedding is None:

        raise RuntimeError(
            "No face detected in selfie image."
        )

    # --------------------------------------------------------
    # COSINE SIMILARITY
    # --------------------------------------------------------

    similarity = cosine_similarity(
        document_embedding,
        selfie_embedding
    )

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    verdict, confidence = make_decision(
        similarity
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print()
    print(
        f"Similarity : {similarity:.4f}"
    )

    print(
        f"Threshold  : {MATCH_THRESHOLD:.2f}"
    )

    print(
        f"Result     : {verdict}"
    )

    print(
        f"Confidence : {confidence:.2%}"
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    return {
        "person_id": person_id,

        "document": str(
            document_path.relative_to(
                PROJECT_ROOT
            )
        ),

        "selfie": str(
            selfie_path.relative_to(
                PROJECT_ROOT
            )
        ),

        "similarity": round(
            similarity,
            4
        ),

        "threshold": MATCH_THRESHOLD,

        "verdict": verdict,

        "confidence": round(
            confidence,
            4
        )
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("ADAFACE FACE VERIFICATION SYSTEM")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD ADAFACE
    # --------------------------------------------------------

    print("[AdaFace] Loading model...")

    model = load_pretrained_model()

    print("[AdaFace] Model ready.")
    print()

    # --------------------------------------------------------
    # CHECK DATA DIRECTORY
    # --------------------------------------------------------

    if not DATA_DIR.exists():

        raise FileNotFoundError(
            f"Data directory not found:\n{DATA_DIR}"
        )

    # --------------------------------------------------------
    # FIND PERSON FOLDERS
    # --------------------------------------------------------

    person_directories = sorted(
        [
            path
            for path in DATA_DIR.iterdir()
            if path.is_dir()
            and path.name.startswith("person_")
        ]
    )

    if not person_directories:

        raise FileNotFoundError(
            f"No person folders found in:\n{DATA_DIR}"
        )

    print(
        f"[AdaFace] Found "
        f"{len(person_directories)} people."
    )

    print()

    # --------------------------------------------------------
    # PROCESS ALL PEOPLE
    # --------------------------------------------------------

    results = []

    for person_dir in person_directories:

        try:

            result = process_person(
                model,
                person_dir
            )

            results.append(result)

        except Exception as error:

            print()
            print(
                f"[ERROR] {person_dir.name}: "
                f"{error}"
            )

            results.append(
                {
                    "person_id": person_dir.name,
                    "verdict": "ERROR",
                    "confidence": 0.0,
                    "error": str(error)
                }
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    verified_count = sum(
        1
        for result in results
        if result.get("verdict") == "VERIFIED"
    )

    not_verified_count = sum(
        1
        for result in results
        if result.get("verdict") == "NOT_VERIFIED"
    )

    uncertain_count = sum(
        1
        for result in results
        if result.get("verdict") == "UNCERTAIN"
    )

    error_count = sum(
        1
        for result in results
        if result.get("verdict") == "ERROR"
    )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    for result in results:

        person_id = result["person_id"]

        similarity = result.get(
            "similarity"
        )

        verdict = result.get(
            "verdict",
            "ERROR"
        )

        if similarity is not None:

            print(
                f"{person_id:<12} "
                f"Similarity: "
                f"{similarity:.4f}   "
                f"{verdict}"
            )

        else:

            print(
                f"{person_id:<12} "
                f"{verdict}"
            )

    print("=" * 60)

    print(
        f"VERIFIED      : {verified_count}"
    )

    print(
        f"NOT VERIFIED  : {not_verified_count}"
    )

    print(
        f"UNCERTAIN     : {uncertain_count}"
    )

    print(
        f"ERROR         : {error_count}"
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output = {

        "system":
            "AdaFace Face Verification",

        "model":
            "AdaFace IR-50 MS1MV2",

        "alignment":
            "MTCNN",

        "embedding_dimension":
            512,

        "similarity_metric":
            "cosine",

        "thresholds": {

            "verified":
                MATCH_THRESHOLD,

            "uncertain":
                UNCERTAIN_THRESHOLD
        },

        "summary": {

            "total":
                len(results),

            "verified":
                verified_count,

            "not_verified":
                not_verified_count,

            "uncertain":
                uncertain_count,

            "error":
                error_count
        },

        "results":
            results
    }

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 60)
    print("VERIFICATION TEST COMPLETED")
    print("=" * 60)

    print(
        f"Results saved to:"
    )

    print(
        RESULTS_FILE
    )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
