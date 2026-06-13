import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# =====================================================
# CONFIG
# =====================================================

SOURCE_IMAGES = "crop_words"
SOURCE_CSV    = "new_labels.csv"

VAL_FOLDER = "DataSet/Validation/validation_words"
TEST_FOLDER = "DataSet/Testing/testing_words"

VAL_CSV  = "DataSet/Validation/validation_labels.csv"
TEST_CSV = "DataSet/Testing/testing_labels.csv"

# =====================================================
# HELPERS
# =====================================================

def get_next_id(folder):

    ids = []

    for f in os.listdir(folder):

        if f.lower().endswith(".png"):

            name = os.path.splitext(f)[0]

            if name.isdigit():
                ids.append(int(name))

    return max(ids) + 1 if ids else 0


# =====================================================
# LOAD NEW DATA
# =====================================================

df = pd.read_csv(SOURCE_CSV)

print(f"Total new samples: {len(df)}")

# =====================================================
# SPLIT
# =====================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    shuffle=True
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    shuffle=True
)

print(f"Train: {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Testing: {len(test_df)}")

# =====================================================
# VALIDATION
# =====================================================

next_val_id = get_next_id(VAL_FOLDER)

print(f"Validation starts from: {next_val_id}")

new_val_rows = []

for _, row in val_df.iterrows():

    old_name = row["IMAGE"]

    src = os.path.join(SOURCE_IMAGES, old_name)

    new_name = f"{next_val_id}.png"

    dst = os.path.join(VAL_FOLDER, new_name)

    shutil.copy2(src, dst)

    row = row.copy()
    row["IMAGE"] = new_name

    new_val_rows.append(row)

    next_val_id += 1

new_val_df = pd.DataFrame(new_val_rows)

existing_val = pd.read_csv(VAL_CSV)

combined_val = pd.concat(
    [existing_val, new_val_df],
    ignore_index=True
)

combined_val.to_csv(
    VAL_CSV,
    index=False
)

# =====================================================
# TESTING
# =====================================================

next_test_id = get_next_id(TEST_FOLDER)

print(f"Testing starts from: {next_test_id}")

new_test_rows = []

for _, row in test_df.iterrows():

    old_name = row["IMAGE"]

    src = os.path.join(SOURCE_IMAGES, old_name)

    new_name = f"{next_test_id}.png"

    dst = os.path.join(TEST_FOLDER, new_name)

    shutil.copy2(src, dst)

    row = row.copy()
    row["IMAGE"] = new_name

    new_test_rows.append(row)

    next_test_id += 1

new_test_df = pd.DataFrame(new_test_rows)

existing_test = pd.read_csv(TEST_CSV)

combined_test = pd.concat(
    [existing_test, new_test_df],
    ignore_index=True
)

combined_test.to_csv(
    TEST_CSV,
    index=False
)

# =====================================================
# SAVE TRAIN CSV
# =====================================================

train_df.to_csv(
    "new_training_split.csv",
    index=False
)

print("\n====================================")
print("DONE")
print(f"Validation added : {len(new_val_df)}")
print(f"Testing added    : {len(new_test_df)}")
print(f"Training left    : {len(train_df)}")
print("====================================")