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
    "fuzzy_cutoff" : 60,    
}


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Evaluate] Device     : {device}")

    print(f"[Evaluate] Loading model: {CONFIG['model_path']}")

    checkpoint = torch.load(CONFIG["model_path"], map_location=device, weights_only=False)

    model = CRNN(hidden_size=256).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"[Evaluate] Loaded (epoch {checkpoint.get('epoch', '?')})")

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

    results = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch["image"].to(device)
            texts  = batch["text"]

            logits    = model(images)                
            log_probs = logits.argmax(dim=2)         
            log_probs = log_probs.permute(1, 0)      

            for i, indices in enumerate(log_probs):
                predicted  = decode(indices.tolist())
                true_label = texts[i]

                char_sim   = fuzz.ratio(predicted.lower(), true_label.lower())
                fuzzy_sim  = fuzz.WRatio(predicted.lower(), true_label.lower())
                exact      = predicted.lower() == true_label.lower()
                fuzzy_ok   = fuzzy_sim >= CONFIG["fuzzy_cutoff"]

                results.append({
                    "true_medicine"  : true_label,
                    "predicted"      : predicted,
                    "char_similarity": round(char_sim, 2),
                    "fuzzy_score"    : round(fuzzy_sim, 2),
                    "exact_match"    : exact,
                    "fuzzy_match"    : fuzzy_ok,
                })

    report_df = pd.DataFrame(results)
    report_df.to_csv(CONFIG["report_path"], index=False)

    total    = len(results)
    exact    = report_df["exact_match"].sum()
    fuzzy    = report_df["fuzzy_match"].sum()
    avg_sim  = report_df["char_similarity"].mean()

    exact_only   = exact
    fuzzy_only   = fuzzy - exact   
    still_wrong  = total - fuzzy

    summary = f"""
PharmaLens — CRNN Evaluation Summary
======================================
Test images          : {total}
Exact match accuracy : {exact}/{total}  ({100*exact/total:.1f}%)
Fuzzy match accuracy : {fuzzy}/{total}  ({100*fuzzy/total:.1f}%)
  (fuzzy threshold   : {CONFIG['fuzzy_cutoff']}%)
Average char sim     : {avg_sim:.1f}%

Breakdown:
  Exact correct      : {exact_only}   ({100*exact_only/total:.1f}%)
  Fuzzy recovered    : {fuzzy_only}   ({100*fuzzy_only/total:.1f}%)
  Still incorrect    : {still_wrong}   ({100*still_wrong/total:.1f}%)

Worst Predictions (completely wrong):
"""
    mistakes = report_df[~report_df["fuzzy_match"]].sort_values("fuzzy_score")
    for _, row in mistakes.head(10).iterrows():
        summary += (
            f"  Expected: {row['true_medicine']:<20}"
            f"  Got: {row['predicted']:<20}"
            f"  ({row['fuzzy_score']:.1f}%)\n"
        )

    print(summary)

    with open(CONFIG["summary_path"], "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"[Evaluate] Report  → {CONFIG['report_path']}")
    print(f"[Evaluate] Summary → {CONFIG['summary_path']}")


if __name__ == "__main__":
    evaluate()