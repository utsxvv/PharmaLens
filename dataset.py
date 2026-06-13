"""
dataset.py — Loads medicine name images and encodes labels for CRNN training.

Your CSV format:
    IMAGE | MEDICINE_NAME | GENERIC_NAME

What this file does:
    1. Reads the CSV
    2. Opens each image
    3. Preprocesses image  (grayscale → resize → normalize)
    4. Encodes label text  ("Amoxicillin" → [10, 40, 48, ...])
    5. Returns tensors ready for the model
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from charset import encode


# ── Image settings ─────────────────────────────────────────────────────────────
#
# CRNN expects a fixed-height image. Width can vary but we standardize it too
# so batching is easier. These values work well for single medicine words.
#
IMG_HEIGHT = 32     # pixels — standard for CRNN
IMG_WIDTH  = 256    # pixels — wide enough for long names like "Clarithromycin"


class MedicineDataset(Dataset):
    """
    PyTorch Dataset for handwritten medicine name images.

    Args:
        csv_path   : path to CSV file (IMAGE | MEDICINE_NAME | GENERIC_NAME)
        images_dir : folder containing the image files
        augment    : if True, apply random augmentations (use for training only)
    """

    def __init__(self, csv_path: str, images_dir: str, augment: bool = False):
        self.images_dir = images_dir
        self.augment    = augment

        # Load CSV
        self.df = pd.read_csv(csv_path)

        # Validate columns
        required = {"IMAGE", "MEDICINE_NAME"}
        missing  = required - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        # Drop rows with missing values in required columns
        before   = len(self.df)
        self.df  = self.df.dropna(subset=["IMAGE", "MEDICINE_NAME"]).reset_index(drop=True)
        dropped  = before - len(self.df)
        if dropped:
            print(f"[Dataset] Dropped {dropped} rows with missing values.")

        # Remove rows where encoded label would be empty
        # (i.e. medicine name has no recognizable characters)
        valid_mask = self.df["MEDICINE_NAME"].apply(
            lambda name: len(encode(str(name).strip())) > 0
        )
        self.df = self.df[valid_mask].reset_index(drop=True)

        print(f"[Dataset] {csv_path}  →  {len(self.df)} valid samples")

        # ── Image transforms ───────────────────────────────────────────────
        # Training: add slight augmentations to help model generalize
        # Validation/Testing: clean transforms only
        if augment:
            self.transform = transforms.Compose([
                transforms.Grayscale(),                        # convert to 1 channel
                transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),    # fixed size
                transforms.RandomRotation(degrees=5),          # slight tilt ±3°
                transforms.ColorJitter(brightness=0.4,         # vary brightness
                                       contrast=0.4),
                transforms.RandomAffine(degrees=0,
                                        shear=5), 
                transforms.ToTensor(),                         # → [0,1] float tensor
                transforms.Normalize(mean=[0.5],               # → [-1,1] range
                                     std=[0.5]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        label  = str(row["MEDICINE_NAME"]).strip()

        # ── Load image ─────────────────────────────────────────────────────
        img_path = os.path.join(self.images_dir, str(row["IMAGE"]))
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # If image is broken, return a blank black image
            image = Image.fromarray(np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8))

        image_tensor = self.transform(image)

        # ── Encode label ───────────────────────────────────────────────────
        # "Amoxicillin" → [10, 40, 48, 57, 12, 12, 21, 22, 21, 21]
        encoded_label  = encode(label)
        label_tensor   = torch.tensor(encoded_label, dtype=torch.long)
        label_length   = torch.tensor(len(encoded_label), dtype=torch.long)

        return {
            "image"        : image_tensor,   # shape: [1, 32, 128]
            "label"        : label_tensor,   # shape: [label_length]
            "label_length" : label_length,   # scalar
            "text"         : label,          # original string (for evaluation)
        }

    def get_lookup_table(self) -> dict:
        """
        Returns dict: medicine_name_lowercase → {MEDICINE_NAME, GENERIC_NAME}
        Used for post-prediction lookup of GENERIC_NAME.
        """
        lookup = {}
        for _, row in self.df.iterrows():
            name = str(row["MEDICINE_NAME"]).strip()
            key  = name.lower()
            lookup[key] = {
                "MEDICINE_NAME" : name,
                "GENERIC_NAME"  : str(row["GENERIC_NAME"]).strip()
                                  if "GENERIC_NAME" in row and pd.notna(row["GENERIC_NAME"])
                                  else "N/A"
            }
        return lookup


def collate_fn(batch):
    """
    Custom collate function for DataLoader.

    Why needed:
        Labels have different lengths ("Tab" vs "Clarithromycin").
        PyTorch cannot stack variable-length tensors by default.
        This function pads labels so they can be stacked into a batch.
    """
    images        = torch.stack([item["image"] for item in batch])
    label_lengths = torch.stack([item["label_length"] for item in batch])
    texts         = [item["text"] for item in batch]

    # Pad all labels to the length of the longest label in this batch
    max_len = max(item["label"].size(0) for item in batch)
    padded_labels = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, item in enumerate(batch):
        length = item["label"].size(0)
        padded_labels[i, :length] = item["label"]

    return {
        "image"        : images,           # [batch, 1, 32, 128]
        "label"        : padded_labels,    # [batch, max_label_len]
        "label_length" : label_lengths,    # [batch]
        "text"         : texts,            # list of strings
    }
