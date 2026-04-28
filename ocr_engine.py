"""
ocr_engine.py — Core prediction engine for PharmaLens.

Flow:
    Image → quality check → preprocess → CRNN → fuzzy match → result
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from rapidfuzz import process, fuzz
import torchvision.transforms as transforms

from model   import CRNN
from charset import decode


# ── Thresholds ─────────────────────────────────────────────────────────────────
BLUR_THRESHOLD     = 50
BRIGHTNESS_MIN     = 20
BRIGHTNESS_MAX     = 240
FUZZY_MATCH_CUTOFF = 75

# ── Image settings (must match training) ──────────────────────────────────────
IMG_HEIGHT = 32
IMG_WIDTH  = 128

# ── Transform (same as validation transform in dataset.py) ────────────────────
TRANSFORM = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# ── Globals loaded once at startup ────────────────────────────────────────────
_model     = None
_device    = None
_lookup    = {}
_med_names = []


def load_engine(model_path: str, csv_paths):
    """
    Load the trained CRNN model and build medicine lookup table.
    Call this once before using predict().
    """
    global _model, _device, _lookup, _med_names

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Engine] Loading model from: {model_path}")

    checkpoint = torch.load(model_path, map_location=_device)
    _model     = CRNN(hidden_size=256).to(_device)
    _model.load_state_dict(checkpoint["model_state"])
    _model.eval()

    # Build lookup table from all CSVs
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]

    dfs = [pd.read_csv(p) for p in csv_paths]
    df  = pd.concat(dfs).drop_duplicates(subset=["MEDICINE_NAME"])

    for _, row in df.iterrows():
        name = str(row["MEDICINE_NAME"]).strip()
        key  = name.lower()
        _lookup[key] = {
            "MEDICINE_NAME" : name,
            "GENERIC_NAME"  : str(row["GENERIC_NAME"]).strip()
                              if "GENERIC_NAME" in row and pd.notna(row["GENERIC_NAME"])
                              else "N/A"
        }

    _med_names = list(_lookup.keys())
    print(f"[Engine] {len(_med_names)} medicines loaded. Ready ✅")


def _check_quality(image_path: str):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False, "Cannot read image. Upload a valid JPG or PNG."

    blur = cv2.Laplacian(img, cv2.CV_64F).var()
    if blur < BLUR_THRESHOLD:
        return False, f"Image too blurry. Please upload a clearer photo."

    brightness = np.mean(img)
    if brightness < BRIGHTNESS_MIN:
        return False, "Image too dark. Please use better lighting."
    if brightness > BRIGHTNESS_MAX:
        return False, "Image appears blank or overexposed."

    return True, "OK"


def _preprocess_image(image_path: str) -> torch.Tensor:
    """Load, preprocess and convert image to tensor."""
    img  = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img  = clahe.apply(img)
    img  = cv2.fastNlMeansDenoising(img, h=10)
    img  = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    pil_image = Image.fromarray(img)
    tensor    = TRANSFORM(pil_image).unsqueeze(0)   # [1, 1, 32, 128]
    return tensor


def _predict_text(image_tensor: torch.Tensor) -> str:
    """Run CRNN on image tensor and return decoded text."""
    image_tensor = image_tensor.to(_device)

    with torch.no_grad():
        logits  = _model(image_tensor)              # [seq, 1, classes]
        indices = logits.argmax(dim=2)              # [seq, 1]
        indices = indices.squeeze(1).tolist()       # [seq]

    return decode(indices)


def predict(image_path: str) -> dict:
    """
    Full prediction pipeline.

    Returns dict with:
        status         → "success" | "low_quality" | "not_found"
        medicine_name  → matched name or None
        generic_name   → matched generic or None
        raw_prediction → raw OCR text
        confidence     → fuzzy match score 0-100
        message        → human readable message
    """

    # Layer 1 — Quality check
    ok, reason = _check_quality(image_path)
    if not ok:
        return {
            "status"        : "low_quality",
            "medicine_name" : None,
            "generic_name"  : None,
            "raw_prediction": None,
            "confidence"    : 0,
            "message"       : reason
        }

    # Layer 2 — Preprocess + predict
    tensor   = _preprocess_image(image_path)
    raw_text = _predict_text(tensor)

    # Layer 3 — Fuzzy match against medicine database
    match = process.extractOne(
        raw_text.lower(),
        _med_names,
        scorer    = fuzz.WRatio,
        score_cutoff = FUZZY_MATCH_CUTOFF
    )

    if not match:
        return {
            "status"        : "not_found",
            "medicine_name" : None,
            "generic_name"  : None,
            "raw_prediction": raw_text,
            "confidence"    : 0,
            "message"       : f"Detected '{raw_text}' but could not match to a known medicine."
        }

    matched_key = match[0]
    confidence  = round(match[1], 1)
    info        = _lookup[matched_key]

    return {
        "status"        : "success",
        "medicine_name" : info["MEDICINE_NAME"],
        "generic_name"  : info["GENERIC_NAME"],
        "raw_prediction": raw_text,
        "confidence"    : confidence,
        "message"       : "Medicine identified successfully."
    }
