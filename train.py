"""
train.py — Train the CRNN model from scratch on medicine name images.

Usage:
    python train.py

What happens:
    1. Loads your dataset from DataSet/
    2. Builds CRNN model from scratch
    3. Trains using CTC loss
    4. Saves best model to  saved_model/crnn_best.pth
    5. Saves training log  to  training_log.csv
"""

import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset  import MedicineDataset, collate_fn
from model    import CRNN, count_parameters
from charset  import decode


# ── Configuration ──────────────────────────────────────────────────────────────

CONFIG = {
    # Dataset paths
    "train_csv"    : "DataSet/Training/training_labels.csv",
    "train_images" : "DataSet/Training/training_words",
    "val_csv"      : "DataSet/Validation/validation_labels.csv",
    "val_images"   : "DataSet/Validation/validation_words",

    # Model
    "hidden_size"  : 256,      # LSTM hidden units

    # Training
    "epochs"       : 30,
    "batch_size"   : 32,       # samples per batch
    "learning_rate": 0.001,    # how fast model learns
    "num_workers"  : 0,        # MUST be 0 on Windows

    # Output
    "save_dir"     : "saved_model",
    "log_path"     : "training_log.csv",
}


def compute_accuracy(model, loader, device):
    """
    Compute exact-match accuracy on a dataset.
    Exact match = predicted text == true text (case-insensitive).
    """
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            texts  = batch["text"]

            logits      = model(images)                    # [seq, batch, classes]
            log_probs   = logits.argmax(dim=2)             # [seq, batch]
            log_probs   = log_probs.permute(1, 0)          # [batch, seq]

            for i, indices in enumerate(log_probs):
                predicted = decode(indices.tolist())
                if predicted.lower() == texts[i].lower():
                    correct += 1
                total += 1

    return (correct / total * 100) if total > 0 else 0.0


def train():
    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  PharmaLens — CRNN Training from Scratch")
    print(f"{'='*55}")
    print(f"  Device      : {device}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset = MedicineDataset(
        CONFIG["train_csv"],
        CONFIG["train_images"],
        augment=True             # augmentations on for training
    )
    val_dataset = MedicineDataset(
        CONFIG["val_csv"],
        CONFIG["val_images"],
        augment=False            # no augmentations for validation
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = CONFIG["batch_size"],
        shuffle     = True,
        num_workers = CONFIG["num_workers"],
        collate_fn  = collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = CONFIG["batch_size"],
        shuffle     = False,
        num_workers = CONFIG["num_workers"],
        collate_fn  = collate_fn,
    )

    print(f"  Train       : {len(train_dataset)} samples")
    print(f"  Validation  : {len(val_dataset)} samples")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CRNN(hidden_size=CONFIG["hidden_size"]).to(device)
    print(f"  Parameters  : {count_parameters(model):,}")
    print(f"  Epochs      : {CONFIG['epochs']}")
    print(f"  Batch size  : {CONFIG['batch_size']}")
    print(f"{'='*55}\n")

    # ── Loss & Optimizer ──────────────────────────────────────────────────────
    #
    # CTC Loss — perfect for sequence recognition without alignment.
    # It figures out on its own which output position maps to which character.
    #
    criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = CONFIG["learning_rate"],
        weight_decay = 1e-4,   # slight regularization to prevent overfitting
    )

    # Reduce learning rate when validation loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5, verbose=True
    )

    # ── Training Loop ─────────────────────────────────────────────────────────
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    best_val_loss = float("inf")
    log_rows      = []

    for epoch in range(1, CONFIG["epochs"] + 1):

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{CONFIG['epochs']} [Train]"):
            images        = batch["image"].to(device)          # [B, 1, 32, 128]
            labels        = batch["label"].to(device)          # [B, max_label_len]
            label_lengths = batch["label_length"].to(device)   # [B]

            # Forward pass
            logits    = model(images)                          # [seq, B, classes]
            log_probs = nn.functional.log_softmax(logits, dim=2)

            # CTC needs sequence lengths (same for all images after CNN)
            seq_len       = logits.size(0)
            input_lengths = torch.full(
                size=(images.size(0),), fill_value=seq_len, dtype=torch.long
            ).to(device)

            # Flatten labels for CTC (remove padding)
            flat_labels = []
            for i in range(labels.size(0)):
                length = label_lengths[i].item()
                flat_labels.append(labels[i, :length])
            flat_labels = torch.cat(flat_labels)

            # Compute loss
            loss = criterion(log_probs, flat_labels, input_lengths, label_lengths)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)  # prevent exploding gradients
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch:02d}/{CONFIG['epochs']} [Val]  "):
                images        = batch["image"].to(device)
                labels        = batch["label"].to(device)
                label_lengths = batch["label_length"].to(device)

                logits    = model(images)
                log_probs = nn.functional.log_softmax(logits, dim=2)

                seq_len       = logits.size(0)
                input_lengths = torch.full(
                    (images.size(0),), fill_value=seq_len, dtype=torch.long
                ).to(device)

                flat_labels = []
                for i in range(labels.size(0)):
                    length = label_lengths[i].item()
                    flat_labels.append(labels[i, :length])
                flat_labels = torch.cat(flat_labels)

                loss      = criterion(log_probs, flat_labels, input_lengths, label_lengths)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # Compute accuracy every 5 epochs (slow on CPU so don't do every epoch)
        if epoch % 5 == 0 or epoch == 1:
            val_acc = compute_accuracy(model, val_loader, device)
            acc_str = f"  Val Acc: {val_acc:.1f}%"
        else:
            val_acc = None
            acc_str = ""

        print(f"\nEpoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}{acc_str}")

        # Adjust learning rate
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path     = os.path.join(CONFIG["save_dir"], "crnn_best.pth")
            torch.save({
                "epoch"       : epoch,
                "model_state" : model.state_dict(),
                "val_loss"    : best_val_loss,
                "config"      : CONFIG,
            }, save_path)
            print(f"           ✅ Best model saved  (val_loss: {best_val_loss:.4f})")

        log_rows.append({
            "epoch"      : epoch,
            "train_loss" : round(avg_train_loss, 4),
            "val_loss"   : round(avg_val_loss, 4),
            "val_acc"    : round(val_acc, 2) if val_acc is not None else "",
        })

    # ── Save Log ───────────────────────────────────────────────────────────────
    with open(CONFIG["log_path"], "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_acc"])
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\n{'='*55}")
    print(f"  Training complete!")
    print(f"  Best val loss : {best_val_loss:.4f}")
    print(f"  Model saved   : {CONFIG['save_dir']}/crnn_best.pth")
    print(f"  Log saved     : {CONFIG['log_path']}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    train()
