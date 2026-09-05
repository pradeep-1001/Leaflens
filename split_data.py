import os
import sys
import shutil
import random

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "dataset", "raw")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

def prepare_splits():
    if not os.path.exists(RAW_DIR):
        print(f"[!] Error: Please create and put your downloaded dataset folders inside:\n   {RAW_DIR}")
        return

    # Discover classes
    classes = [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))]
    if not classes:
        print(f"[!] No class folders found in {RAW_DIR}.")
        print("    Please put your disease folders (e.g. Downy_Mildew, Powdery_Mildew, Healthy, etc.) inside dataset/raw/")
        return

    print(f"[*] Found {len(classes)} classes: {classes}")

    # Create train, val, test directories
    for split in ["train", "val", "test"]:
        for cls in classes:
            os.makedirs(os.path.join(DATASET_DIR, split, cls), exist_ok=True)

    total_images_processed = 0

    # Copy files
    for cls in classes:
        cls_dir = os.path.join(RAW_DIR, cls)
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Shuffle for unbiased sampling
        random.seed(42)
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * SPLIT_RATIOS["train"])
        n_val = int(n_total * SPLIT_RATIOS["val"])

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }

        print(f"\n[+] Processing '{cls}' ({n_total} images):")
        for split_name, split_imgs in splits.items():
            for img in split_imgs:
                src = os.path.join(cls_dir, img)
                dst = os.path.join(DATASET_DIR, split_name, cls, img)
                shutil.copyfile(src, dst)
            print(f"    -> {split_name.capitalize()}: {len(split_imgs)} images")
        
        total_images_processed += n_total

    print(f"\n[OK] Successfully organized {total_images_processed} images into train/val/test splits!")
    print(">>> Next step: Run 'python train.py' to begin training!")

if __name__ == "__main__":
    prepare_splits()
