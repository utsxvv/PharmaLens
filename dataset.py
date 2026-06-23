import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from charset import encode


IMG_HEIGHT = 32     
IMG_WIDTH  = 256


class MedicineDataset(Dataset):
    def __init__(self, csv_path: str, images_dir: str, augment: bool = False):
        self.images_dir = images_dir
        self.augment    = augment

        self.df = pd.read_csv(csv_path)

        required = {"IMAGE", "MEDICINE_NAME"}
        missing  = required - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        before   = len(self.df)
        self.df  = self.df.dropna(subset=["IMAGE", "MEDICINE_NAME"]).reset_index(drop=True)
        dropped  = before - len(self.df)
        if dropped:
            print(f"[Dataset] Dropped {dropped} rows with missing values.")

        valid_mask = self.df["MEDICINE_NAME"].apply(
            lambda name: len(encode(str(name).strip())) > 0
        )
        self.df = self.df[valid_mask].reset_index(drop=True)

        print(f"[Dataset] {csv_path}  →  {len(self.df)} valid samples")

        if augment:
            self.transform = transforms.Compose([
                transforms.Grayscale(),                        
                transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),    
                transforms.RandomRotation(degrees=5),          
                transforms.ColorJitter(brightness=0.4,         
                                       contrast=0.4),
                transforms.RandomAffine(degrees=0,
                                        shear=5), 
                transforms.ToTensor(),                         
                transforms.Normalize(mean=[0.5],               
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

        img_path = os.path.join(self.images_dir, str(row["IMAGE"]))
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.fromarray(np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8))

        image_tensor = self.transform(image)

        encoded_label  = encode(label)
        label_tensor   = torch.tensor(encoded_label, dtype=torch.long)
        label_length   = torch.tensor(len(encoded_label), dtype=torch.long)

        return {
            "image"        : image_tensor,   
            "label"        : label_tensor,   
            "label_length" : label_length,   
            "text"         : label,          
        }

    def get_lookup_table(self) -> dict:
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
    images        = torch.stack([item["image"] for item in batch])
    label_lengths = torch.stack([item["label_length"] for item in batch])
    texts         = [item["text"] for item in batch]

    max_len = max(item["label"].size(0) for item in batch)
    padded_labels = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, item in enumerate(batch):
        length = item["label"].size(0)
        padded_labels[i, :length] = item["label"]

    return {
        "image"        : images,           
        "label"        : padded_labels,    
        "label_length" : label_lengths,    
        "text"         : texts,            
    }
