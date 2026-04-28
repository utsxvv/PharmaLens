"""
evaluate.py — Evaluate trained CRNN on the test set.

Usage:
    python evaluate.py

Outputs:
    evaluation_report.csv   — per-image results
    evaluation_summary.txt  — overall accuracy
"""

import os
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from rapidfuzz import fuzz
from torch.utils.data import DataLoader

from dataset import MedicineDataset, collate_fn
from model   import CRNN
from charset import decode


CONFIG = {
    "test_csv"     : "DataSet/Testing/testing_labels.csv",
    "test_images"  : "DataSet/Testing/testing_words",
    "model_path"   : "saved_model/crnn_best.pth",
    "report_path"  : "evaluation_report.csv",
    "summary_path" : "evaluation_summary.txt",
    "batch_size"   : 32,
    "num_workers"  : 0,
    "fuzzy_cutoff" : 80,
}


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Evaluate] Device: {device}")

    # ── Load Model ─────────────────────────────────────────────────────────────
    print(f"[Evaluate] Loading model from: {CONFIG['model_path']}")
    checkpoint = torch.load(CONFIG["model_path"], map_location=device)

    model = CRNN(hidden_size=256).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"[Evaluate] Model loaded  (trained for {checkpoint['epoch']} epochs)")

    # ── Test Dataset ───────────────────────────────────────────────────────────
    test_dataset = MedicineDataset(
        CONFIG["test_csv"],
        CONFIG["test_images"],
        augment=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = CONFIG["batch_size"],
        shuffle     = False,
        num_workers = CONFIG["num_workers"],
        collate_fn  = collate_fn,
    )
    print(f"[Evaluate] Test samples: {len(test_dataset)}")

    # ── Load test CSV for generic name lookup ──────────────────────────────────
    df_test = pd.read_csv(CONFIG["test_csv"])

    # ── Run Predictions ────────────────────────────────────────────────────────
    results = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch["image"].to(device)
            texts  = batch["text"]

            logits    = model(images)                   # [seq, B, classes]
            log_probs = logits.argmax(dim=2)            # [seq, B]
            log_probs = log_probs.permute(1, 0)         # [B, seq]

            for i, indices in enumerate(log_probs):
                predicted  = decode(indices.tolist())
                true_label = texts[i]
                similarity = fuzz.ratio(predicted.lower(), true_label.lower())

                results.append({
                    "true_medicine" : true_label,
                    "predicted"     : predicted,
                    "similarity"    : similarity,
                    "exact_match"   : predicted.lower() == true_label.lower(),
                    "fuzzy_match"   : similarity >= CONFIG["fuzzy_cutoff"],
                })

    # ── Report ─────────────────────────────────────────────────────────────────
    report_df = pd.DataFrame(results)
    report_df.to_csv(CONFIG["report_path"], index=False)

    total       = len(results)
    exact       = report_df["exact_match"].sum()
    fuzzy       = report_df["fuzzy_match"].sum()
    avg_sim     = report_df["similarity"].mean()

    summary = f"""
PharmaLens — CRNN Evaluation Summary
======================================
Test images          : {total}
Exact match accuracy : {exact}/{total}  ({100*exact/total:.1f}%)
Fuzzy match accuracy : {fuzzy}/{total}  ({100*fuzzy/total:.1f}%)
  (fuzzy threshold   : {CONFIG['fuzzy_cutoff']}%)
Average similarity   : {avg_sim:.1f}%

Worst Predictions:
"""
    mistakes = report_df[~report_df["fuzzy_match"]].sort_values("similarity")
    for _, row in mistakes.head(10).iterrows():
        summary += f"  Expected: {row['true_medicine']:<20}  Got: {row['predicted']:<20}  ({row['similarity']}%)\n"

    print(summary)

    with open(CONFIG["summary_path"], "w") as f:
        f.write(summary)

    print(f"[Evaluate] Report  → {CONFIG['report_path']}")
    print(f"[Evaluate] Summary → {CONFIG['summary_path']}")


if __name__ == "__main__":
    evaluate()
