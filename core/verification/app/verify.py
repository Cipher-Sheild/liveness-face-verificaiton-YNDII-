"""Command line entry point for one document/liveness comparison."""

import argparse
import json

from app.verification import FaceVerifier


def main():
    parser = argparse.ArgumentParser(
        description="Compare an ID/passport portrait with a liveness face image."
    )
    parser.add_argument("--document", required=True, help="Path to full ID/passport image")
    parser.add_argument("--live-face", required=True, help="Path to liveness-approved face image")
    args = parser.parse_args()

    result = FaceVerifier().verify(args.document, args.live_face)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
