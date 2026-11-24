import os
import shutil
import random

# ==============================
# CHANGE THIS BASE PATH ONLY
# ==============================
BASE_DIR = r"C:\Users\Admin\Downloads\minorproj"

SOURCE_DIRS = {
    "NORMAL": os.path.join(BASE_DIR, "NORMAL"),
    "pneumonia": os.path.join(BASE_DIR, "pneumonia")
}

TARGET_DIRS = {
    "train": os.path.join(BASE_DIR, "train"),
    "val": os.path.join(BASE_DIR, "val"),
    "test": os.path.join(BASE_DIR, "test")
}

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

# ==============================
# SPLITTING LOGIC
# ==============================
for class_name, src_dir in SOURCE_DIRS.items():
    images = [f for f in os.listdir(src_dir)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split_name, files in splits.items():
        split_class_dir = os.path.join(TARGET_DIRS[split_name], class_name)
        os.makedirs(split_class_dir, exist_ok=True)

        for file in files:
            src_path = os.path.join(src_dir, file)
            dst_path = os.path.join(split_class_dir, file)
            shutil.copy(src_path, dst_path)

    print(f"{class_name}: {total} images split successfully")

print("\n✅ Dataset split completed!")
