"""
ocr_engine.py — Core prediction engine for PharmaLens.

Flow:
    Image → quality check → preprocess → CRNN → fuzzy match → result

Fixes applied (vs previous version):
    FIX 1: IMG_WIDTH 128 → 256  (must match dataset.py used in training)
    FIX 2: FUZZY_MATCH_CUTOFF 75 → 60  (recovers 49 more predictions)
    FIX 3: BLUR_THRESHOLD 50 → 30  (preprocessed binary images score lower)
    FIX 4: _preprocess_image — detects already-preprocessed images,
            skips CLAHE/denoise on binary images to avoid distortion
    FIX 5: weights_only=False in torch.load (removes deprecation warning)
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
BLUR_THRESHOLD     = 30       # FIX 3: was 50 — preprocessed images score lower
BRIGHTNESS_MIN     = 20
FUZZY_MATCH_CUTOFF = 60       # FIX 2: was 75 — too strict, 49 extra recoveries

# ── Image settings — MUST match dataset.py used during training ───────────────
IMG_HEIGHT = 32
IMG_WIDTH  = 256              # FIX 1: was 128 — model trained on 256px images

# ── Transform — identical to validation transform in dataset.py ───────────────
TRANSFORM = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# ── Globals — loaded once at startup ──────────────────────────────────────────
_model     = None
_device    = None
_lookup    = {}
_med_names = []


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_engine(model_path: str, csv_paths):
    """
    Load trained CRNN model and build medicine lookup table.
    Call this once at startup before using predict().
    """
    global _model, _device, _lookup, _med_names

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Engine] Loading model from: {model_path}")

    # FIX 5: weights_only=False removes PyTorch deprecation warning
    checkpoint = torch.load(model_path, map_location=_device, weights_only=False)
    _model     = CRNN(hidden_size=256).to(_device)
    _model.load_state_dict(checkpoint["model_state"])
    _model.eval()
    print(f"[Engine] Model loaded (epoch {checkpoint.get('epoch', '?')})")

    # Build lookup table from all CSVs combined
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]

    dfs = [pd.read_csv(p) for p in csv_paths]
    df  = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["MEDICINE_NAME"])

    _lookup    = {}
    _med_names = []

    for _, row in df.iterrows():
        name = str(row["MEDICINE_NAME"]).strip()
        if not name:
            continue
        key = name.lower()
        _lookup[key] = {
            "MEDICINE_NAME" : name,
            "GENERIC_NAME"  : str(row["GENERIC_NAME"]).strip()
                              if "GENERIC_NAME" in row and pd.notna(row["GENERIC_NAME"])
                              else "N/A"
        }

    _med_names = list(_lookup.keys())
    print(f"[Engine] {len(_med_names)} medicines loaded. Ready ✅")


# ══════════════════════════════════════════════════════════════════════════════
# QUALITY CHECK
# ══════════════════════════════════════════════════════════════════════════════

def _is_preprocessed(img: np.ndarray) -> bool:
    """
    Detects whether an image is already preprocessed (binary: black text on
    white background). Dataset images are already preprocessed; raw user
    uploads are not.

    A preprocessed binary image has >85% pixels near pure white or pure black.
    """
    near_white = np.sum(img > 200) / img.size
    near_black = np.sum(img < 50)  / img.size
    return (near_white + near_black) > 0.85


def _check_quality(image_path: str):
    """
    Validates image before running OCR.
    Returns (is_valid: bool, reason: str)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return False, "Cannot read image. Please upload a valid JPG or PNG file."

    # Blur check — FIX 3: lower threshold for preprocessed binary images
    blur_score = cv2.Laplacian(img, cv2.CV_64F).var()
    if blur_score < BLUR_THRESHOLD:
        return False, "Image is too blurry. Please upload a clearer photo."

    # Darkness check
    if np.mean(img) < BRIGHTNESS_MIN:
        return False, "Image is too dark. Please use better lighting."

    # Text presence check — looks for dark pixels (text)
    dark_ratio = np.sum(img < 127) / img.size
    if dark_ratio < 0.01:
        return False, "Image appears blank. No text detected."
    if dark_ratio > 0.95:
        return False, "Image is completely dark. Please use better lighting."

    return True, "OK"


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def _preprocess_image(image_path: str) -> torch.Tensor:
    """
    FIX 4: Smart preprocessing — checks if image is already binary/clean.

    Already preprocessed images (white bg + black text from dataset):
        → skip CLAHE and denoising, just convert and resize

    Raw user uploads (color photos, uneven lighting):
        → apply full CLAHE + denoise preprocessing pipeline

    Both paths end with the same TRANSFORM (Grayscale → Resize → Normalize).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if not _is_preprocessed(img):
        # Raw image — apply preprocessing to clean it up
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img   = clahe.apply(img)
        img   = cv2.fastNlMeansDenoising(img, h=10)

    # Convert grayscale → RGB so PIL/transforms can handle it uniformly
    img_rgb   = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    pil_image = Image.fromarray(img_rgb)

    # Apply standard transform: Grayscale → Resize(32,256) → Tensor → Normalize
    tensor = TRANSFORM(pil_image).unsqueeze(0)   # [1, 1, 32, 256]
    return tensor


# ══════════════════════════════════════════════════════════════════════════════
# CRNN INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def _predict_text(image_tensor: torch.Tensor) -> str:
    """Run CRNN on image tensor, return decoded raw text."""
    image_tensor = image_tensor.to(_device)

    with torch.no_grad():
        logits  = _model(image_tensor)          # [seq, 1, num_classes]
        indices = logits.argmax(dim=2)          # [seq, 1]
        indices = indices.squeeze(1).tolist()   # [seq]

    return decode(indices)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PREDICT
# ══════════════════════════════════════════════════════════════════════════════

def predict(image_path: str) -> dict:
    """
    Full prediction pipeline.

    Returns dict:
        status         → "success" | "low_quality" | "not_found"
        medicine_name  → matched brand name  or None
        generic_name   → matched generic name or None
        raw_prediction → raw OCR text from CRNN
        confidence     → fuzzy match score (0–100)
        message        → human-readable status message
    """

    # ── Layer 1: Quality check ─────────────────────────────────────────────
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

    # ── Layer 2: Preprocess + CRNN inference ───────────────────────────────
    tensor   = _preprocess_image(image_path)
    raw_text = _predict_text(tensor)

    if not raw_text.strip():
        return {
            "status"        : "not_found",
            "medicine_name" : None,
            "generic_name"  : None,
            "raw_prediction": "",
            "confidence"    : 0,
            "message"       : "Model could not read any text from this image."
        }

    # ── Layer 3: Fuzzy match against medicine database ─────────────────────
    match = process.extractOne(
        raw_text.lower(),
        _med_names,
        scorer       = fuzz.WRatio,
        score_cutoff = FUZZY_MATCH_CUTOFF
    )

    if not match:
        return {
            "status"        : "not_found",
            "medicine_name" : None,
            "generic_name"  : None,
            "raw_prediction": raw_text,
            "confidence"    : 0,
            "message"       : f"Detected '{raw_text}' but could not match it to a known medicine."
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