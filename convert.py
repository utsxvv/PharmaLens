import json
import cv2
import os
import numpy as np
import pandas as pd
import re

JSON_FILE     = "label_studio_export.json"
IMAGE_FOLDER  = "images"
OUTPUT_FOLDER = "DataSet/Training/training_words"
CSV_FILE      = "DataSet/Training/training_labels.csv"
AUGMENT_COUNT = 7

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def get_next_image_id(folder):

    max_id = -1

    for file in os.listdir(folder):

        if file.lower().endswith((".png", ".jpg", ".jpeg")):

            name = os.path.splitext(file)[0]

            if name.isdigit():
                max_id = max(max_id, int(name))

    return max_id + 1


def get_skew_angle(binary_image):
    inverted = cv2.bitwise_not(binary_image)

    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilated = cv2.dilate(inverted, kernel, iterations=2)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    angles = []
    for contour in contours:
        if cv2.contourArea(contour) > 200:
            angle = cv2.minAreaRect(contour)[-1]

            if angle > 45:
                angle = angle - 90
            elif angle < -45:
                angle = angle + 90

            if -20 < angle < 20:
                angles.append(angle)

    return float(np.median(angles)) if angles else 0.0


def deskew(image, angle):
    if abs(angle) < 0.5:
        return image

    h, w   = image.shape[:2]
    center = (w // 2, h // 2)
    M      = cv2.getRotationMatrix2D(center, angle, 1.0)

    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255
    )


def remove_small_noise(binary_image, min_area=15):
    inverted  = cv2.bitwise_not(binary_image)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        inverted, connectivity=8
    )

    cleaned = np.zeros_like(inverted)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255

    return cv2.bitwise_not(cleaned)


def preprocess_crop(crop):
    import cv2
    import numpy as np

    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)

    binary = cv2.copyMakeBorder(
        binary,
        top=10,
        bottom=10,
        left=10,
        right=10,
        borderType=cv2.BORDER_CONSTANT,
        value=255
    )

    target_h = 64

    h, w = binary.shape

    scale = target_h / h
    new_w = max(1, int(w * scale))

    interp = (
        cv2.INTER_AREA
        if scale < 1
        else cv2.INTER_CUBIC
    )

    binary = cv2.resize(
        binary,
        (new_w, target_h),
        interpolation=interp
    )

    target_w = 256

    canvas = np.ones(
        (target_h, target_w),
        dtype=np.uint8
    ) * 255

    if new_w > target_w:
        binary = cv2.resize(
            binary,
            (target_w, target_h),
            interpolation=cv2.INTER_AREA
        )
        canvas = binary
    else:
        canvas[:, :new_w] = binary

    return canvas


def augment_single_image(image, n=10):

    augmented = []

    h, w = image.shape

    for _ in range(n):

        aug = image.copy()

        # Small rotation
        angle = np.random.uniform(-3, 3)

        M = cv2.getRotationMatrix2D(
            (w // 2, h // 2),
            angle,
            1.0
        )

        aug = cv2.warpAffine(
            aug,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255
        )

        scale_x = np.random.uniform(0.95, 1.05)

        new_w = max(1, int(w * scale_x))

        stretched = cv2.resize(
            aug,
            (new_w, h),
            interpolation=cv2.INTER_CUBIC
        )

        aug = cv2.resize(
            stretched,
            (w, h),
            interpolation=cv2.INTER_CUBIC
        )

        if np.random.random() > 0.5:

            shear = np.random.uniform(-0.05, 0.05)

            M_shear = np.float32([
                [1, shear, 0],
                [0, 1, 0]
            ])

            aug = cv2.warpAffine(
                aug,
                M_shear,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255
            )

        tx = np.random.randint(-5, 6)
        ty = np.random.randint(-2, 3)

        M_shift = np.float32([
            [1, 0, tx],
            [0, 1, ty]
        ])

        aug = cv2.warpAffine(
            aug,
            M_shift,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255
        )

        if np.random.random() > 0.5:

            kernel = np.ones((2,2), np.uint8)

            if np.random.random() > 0.5:
                aug = cv2.dilate(aug, kernel, iterations=1)
            else:
                aug = cv2.erode(aug, kernel, iterations=1)

        augmented.append(aug)

    return augmented


with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

next_image_id = get_next_image_id(OUTPUT_FOLDER)

print(f"Starting from image ID: {next_image_id}")

for item in data:

    full_name  = os.path.basename(item["data"]["ocr"])
    image_name = full_name.split("-", 1)[1]
    img_path   = os.path.join(IMAGE_FOLDER, image_name)

    print(f"\nLoading: {img_path}")

    img = cv2.imread(img_path)

    if img is None:
        print(f"Not found: {img_path}")
        continue

    H, W = img.shape[:2]

    annotations = item["annotations"][0]["result"]
    boxes       = {}
    texts       = {}

    for ann in annotations:
        ann_id = ann["id"]
        if ann["type"] == "rectangle":
            boxes[ann_id] = ann["value"]
        elif ann["type"] == "textarea":
            text_list = ann["value"].get("text", [])
            if text_list:
                texts[ann_id] = text_list[0]

    for ann_id in boxes:

        if ann_id not in texts:
            continue

        val   = boxes[ann_id]
        x     = int(val["x"]      / 100 * W)
        y     = int(val["y"]      / 100 * H)
        w     = int(val["width"]  / 100 * W)
        h     = int(val["height"] / 100 * H)
        label = texts[ann_id].strip()

        crop = img[y:y+h, x:x+w]

        if crop.size == 0:
            print(f"  ⚠ Empty crop for: {label} — skipping")
            continue

        try:
            processed = preprocess_crop(crop)
        except Exception as e:
            print(f"  Preprocessing failed for {label}: {e} — saving raw")
            gray      = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
            processed = cv2.resize(gray, (256, 64))

        filename = f"{next_image_id}.png"

        cv2.imwrite(
            os.path.join(OUTPUT_FOLDER, filename),
            processed
        )

        rows.append({
            "IMAGE": filename,
            "MEDICINE_NAME": label,
            "GENERIC_NAME": ""
        })

        next_image_id += 1
        print(f"  ✅ {filename}  →  {label}  [original]")

        for aug_img in augment_single_image(processed, n=AUGMENT_COUNT):
            aug_filename = f"{next_image_id}.png"

            cv2.imwrite(
                os.path.join(OUTPUT_FOLDER, aug_filename),
                aug_img
            )

            rows.append({
                "IMAGE": aug_filename,
                "MEDICINE_NAME": label,
                "GENERIC_NAME": ""
            })

            next_image_id += 1

        print(f"  ✅ {AUGMENT_COUNT} augmented versions saved for: {label}")


new_df = pd.DataFrame(rows)

existing_df = pd.read_csv(CSV_FILE)

combined_df = pd.concat(
    [existing_df, new_df],
    ignore_index=True
)

combined_df.to_csv(
    CSV_FILE,
    index=False
)

print("\n===================================")
print("DONE")
print(f"Total crops saved : {len(rows)}")
print(f"  Original images : {len(rows) // (AUGMENT_COUNT + 1)}")
print(f"  Augmented images: {len(rows) - len(rows) // (AUGMENT_COUNT + 1)}")
print(f"CSV saved as      : {CSV_FILE}")
print("===================================")
print("\nNext steps:")
print("  1. Fill GENERIC_NAME column in new_labels.csv")
print("  2. Copy word_crops/ into DataSet/Training/training_words/")
print("  3. Merge new_labels.csv into training_labels.csv")
print("  4. Retrain model on Kaggle")