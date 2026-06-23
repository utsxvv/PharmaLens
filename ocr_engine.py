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


BLUR_THRESHOLD     = 30       
BRIGHTNESS_MIN     = 20
FUZZY_MATCH_CUTOFF = 60       

IMG_HEIGHT = 32
IMG_WIDTH  = 256              

TRANSFORM = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

_model     = None
_device    = None
_lookup    = {}
_med_names = []

def load_engine(model_path: str, csv_paths):
    global _model, _device, _lookup, _med_names

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Engine] Loading model from: {model_path}")

    checkpoint = torch.load(model_path, map_location=_device, weights_only=False)
    _model     = CRNN(hidden_size=256).to(_device)
    _model.load_state_dict(checkpoint["model_state"])
    _model.eval()
    print(f"[Engine] Model loaded (epoch {checkpoint.get('epoch', '?')})")

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
    print(f"[Engine] {len(_med_names)} medicines loaded. Ready [OK]")


def _is_preprocessed(img: np.ndarray) -> bool:
    near_white = np.sum(img > 200) / img.size
    near_black = np.sum(img < 50)  / img.size
    return (near_white + near_black) > 0.85


def _check_quality(image_path: str):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return False, "Cannot read image. Please upload a valid JPG or PNG file."

    blur_score = cv2.Laplacian(img, cv2.CV_64F).var()
    if blur_score < BLUR_THRESHOLD:
        return False, "Image is too blurry. Please upload a clearer photo."

    if np.mean(img) < BRIGHTNESS_MIN:
        return False, "Image is too dark. Please use better lighting."

    dark_ratio = np.sum(img < 127) / img.size
    if dark_ratio < 0.01:
        return False, "Image appears blank. No text detected."
    if dark_ratio > 0.95:
        return False, "Image is completely dark. Please use better lighting."

    return True, "OK"

def _preprocess_image(image_path: str) -> torch.Tensor:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if not _is_preprocessed(img):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img   = clahe.apply(img)
        img   = cv2.fastNlMeansDenoising(img, h=10)

    img_rgb   = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    pil_image = Image.fromarray(img_rgb)

    tensor = TRANSFORM(pil_image).unsqueeze(0)   
    return tensor

def _predict_text(image_tensor: torch.Tensor) -> str:
    image_tensor = image_tensor.to(_device)

    with torch.no_grad():
        logits  = _model(image_tensor)          
        indices = logits.argmax(dim=2)          
        indices = indices.squeeze(1).tolist()   

    return decode(indices)

def predict(image_path: str) -> dict:

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