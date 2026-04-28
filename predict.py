"""
predict.py — Test your trained CRNN on a single image from terminal.

Usage:
    python predict.py <image_path>

Example:
    python predict.py DataSet/Testing/testing_words/01.png
"""

import sys
from ocr_engine import load_engine, predict


def main():
    if len(sys.argv) < 2:
        print("Usage  : python predict.py <image_path>")
        print("Example: python predict.py DataSet/Testing/testing_words/01.png")
        sys.exit(1)

    image_path = sys.argv[1]

    # Load model and medicine database
    load_engine(
        model_path = "saved_model/crnn_best.pth",
        csv_paths  = [
            "DataSet/Training/training_labels.csv",
            "DataSet/Validation/validation_labels.csv",
            "DataSet/Testing/testing_labels.csv",
        ]
    )

    # Run prediction
    result = predict(image_path)

    # Display result
    print("\n" + "="*50)
    print("  PharmaLens — Prediction Result")
    print("="*50)
    print(f"  Status         : {result['status']}")
    print(f"  Raw OCR output : {result['raw_prediction']}")
    print(f"  Medicine Name  : {result['medicine_name']}")
    print(f"  Generic Name   : {result['generic_name']}")
    print(f"  Confidence     : {result['confidence']}%")
    print(f"  Message        : {result['message']}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
