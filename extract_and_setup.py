import os
import sys
import zipfile
import shutil
import random

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ZIP_PATH = r"c:\Users\HP\OneDrive\Documents\Niphad Grape Leaf Disease Dataset (NGLD).zip"
BASE_DIR = r"c:\Users\HP\OneDrive\Documents\Leaflens"
RAW_DIR = os.path.join(BASE_DIR, "dataset", "raw")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DEMO_TEST_DIR = os.path.join(BASE_DIR, "test_samples")

# Standardized class names mapping
CLASS_MAPPING = {
    "Bacterial Leaf Spot": "Bacterial_Leaf_Spot",
    "Downy Mildew": "Downy_Mildew",
    "Healthy Leaves": "Healthy",
    "Powdery Mildew": "Powdery_Mildew"
}

def extract_and_organize():
    print("=" * 60)
    print("📦 EXTRACTING & ORGANIZING NIPHAD GRAPE DATASET")
    print("=" * 60)

    # 1. Clean previous extraction if any
    for d in [RAW_DIR, DEMO_TEST_DIR]:
        os.makedirs(d, exist_ok=True)

    for target_class in CLASS_MAPPING.values():
        os.makedirs(os.path.join(RAW_DIR, target_class), exist_ok=True)
        os.makedirs(os.path.join(DEMO_TEST_DIR, target_class), exist_ok=True)

    print(f"Reading zip archive: {ZIP_PATH}...")
    extracted_counts = {v: 0 for v in CLASS_MAPPING.values()}

    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        for member in z.infolist():
            if member.is_dir():
                continue
            
            # Identify class
            matched_class = None
            for raw_name, std_name in CLASS_MAPPING.items():
                if f"/{raw_name}/" in member.filename.replace("\\", "/"):
                    matched_class = std_name
                    break
            
            if matched_class:
                filename = os.path.basename(member.filename)
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    dest_path = os.path.join(RAW_DIR, matched_class, filename)
                    with z.open(member) as src, open(dest_path, 'wb') as dst:
                        shutil.copyfileobj(src, dst)
                    extracted_counts[matched_class] += 1

    print("\n[+] Raw Images Extracted:")
    total_raw = 0
    for cls, count in extracted_counts.items():
        print(f"  • {cls:20s}: {count} images")
        total_raw += count
    print(f"  Total Extracted      : {total_raw} images")

    # 2. Split dataset into train (70%), val (15%), test (15%)
    print("\n" + "=" * 60)
    print("🔪 CREATING TRAIN (70%) / VAL (15%) / TEST (15%) SPLITS")
    print("=" * 60)

    for split in ["train", "val", "test"]:
        for cls in CLASS_MAPPING.values():
            os.makedirs(os.path.join(DATASET_DIR, split, cls), exist_ok=True)

    demo_sample_counts = {v: 0 for v in CLASS_MAPPING.values()}

    for cls in CLASS_MAPPING.values():
        cls_raw_dir = os.path.join(RAW_DIR, cls)
        images = [f for f in os.listdir(cls_raw_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.seed(42)
        random.shuffle(images)

        n = len(images)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        # Copy to train/val/test
        for img in train_imgs:
            shutil.copyfile(os.path.join(cls_raw_dir, img), os.path.join(DATASET_DIR, "train", cls, img))
        for img in val_imgs:
            shutil.copyfile(os.path.join(cls_raw_dir, img), os.path.join(DATASET_DIR, "val", cls, img))
        for img in test_imgs:
            shutil.copyfile(os.path.join(cls_raw_dir, img), os.path.join(DATASET_DIR, "test", cls, img))

        # Save 15 high-quality dedicated sample images directly into test_samples/ for quick demo & validation!
        for img in test_imgs[:15]:
            shutil.copyfile(os.path.join(cls_raw_dir, img), os.path.join(DEMO_TEST_DIR, cls, img))
            demo_sample_counts[cls] += 1

        print(f"  • {cls:20s}: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)} | Demo Samples={demo_sample_counts[cls]}")

    print("\n" + "=" * 60)
    print("✅ DATASET SETUP COMPLETE!")
    print(f"📁 Training Data       : {os.path.join(DATASET_DIR, 'train')}")
    print(f"📁 Validation Data     : {os.path.join(DATASET_DIR, 'val')}")
    print(f"📁 Test Data           : {os.path.join(DATASET_DIR, 'test')}")
    print(f"🎯 Dedicated Demo Tests: {DEMO_TEST_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    extract_and_organize()
