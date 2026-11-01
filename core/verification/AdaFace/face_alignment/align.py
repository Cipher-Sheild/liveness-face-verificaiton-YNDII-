import sys
import os

from face_alignment import mtcnn
import argparse
from PIL import Image
from tqdm import tqdm
import random
from datetime import datetime
import torch


# ============================================================
# DEVICE SELECTION
# ============================================================
# Apple Silicon Mac:
#   MPS  -> Apple GPU
# Otherwise:
#   CPU
#
# This replaces the original AdaFace CUDA-only setting:
# mtcnn.MTCNN(device='cuda:0', ...)
# ============================================================

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

print(f"[AdaFace] Face alignment device: {device}")


# ============================================================
# MTCNN FACE DETECTOR
# ============================================================

mtcnn_model = mtcnn.MTCNN(
    device=device,
    crop_size=(112, 112)
)


# ============================================================
# PADDING
# ============================================================

def add_padding(
    pil_img,
    top,
    right,
    bottom,
    left,
    color=(0, 0, 0)
):
    width, height = pil_img.size

    new_width = width + right + left
    new_height = height + top + bottom

    result = Image.new(
        pil_img.mode,
        (new_width, new_height),
        color
    )

    result.paste(
        pil_img,
        (left, top)
    )

    return result


# ============================================================
# FACE ALIGNMENT
# ============================================================

def get_aligned_face(
    image_path,
    rgb_pil_image=None
):
    """
    Detect and align a face from an image.

    Parameters
    ----------
    image_path : str
        Path to the image.

    rgb_pil_image : PIL.Image.Image, optional
        PIL RGB image. If provided, image_path is ignored.

    Returns
    -------
    PIL.Image.Image or None
        Aligned 112x112 face, or None if no face is detected.
    """

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    if rgb_pil_image is None:
        img = Image.open(image_path).convert("RGB")
    else:
        assert isinstance(
            rgb_pil_image,
            Image.Image
        ), (
            "Face alignment module requires "
            "PIL image or path to the image"
        )

        img = rgb_pil_image

    # --------------------------------------------------------
    # Find and align face
    # --------------------------------------------------------

    try:
        bboxes, faces = mtcnn_model.align_multi(
            img,
            limit=1
        )

        # Make sure a face was actually detected
        if faces is None or len(faces) == 0:
            print("No face detected.")
            return None

        face = faces[0]

    except Exception as e:
        print("Face detection failed due to error.")
        print(e)
        face = None

    return face