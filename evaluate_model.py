import os
import sys
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Reconfigure stdout for Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "best_model.pth")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate():
    print("=" * 60)
    print("📊 LEAFLENS AI PRO - COMPREHENSIVE TEST BENCHMARK")
    print(f"⚡ Device: {DEVICE}")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"[!] Error: Model checkpoint not found at {MODEL_PATH}")
        return

    # Data transformation
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(TEST_DIR, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    class_names = test_dataset.classes
    print(f"[*] Total Test Samples: {len(test_dataset)} across {len(class_names)} classes")

    # Load Model
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model = models.mobilenet_v3_large()
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()

    y_true = []
    y_pred = []

    print("[+] Evaluating model on unseen test set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    # Classification Report
    print("\n" + "=" * 60)
    print("📈 CLASSIFICATION REPORT (PRECISION, RECALL, F1-SCORE)")
    print("=" * 60)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=[c.replace('_', '\n') for c in class_names],
                yticklabels=[c.replace('_', ' ') for c in class_names])
    plt.title("LeafLens AI Pro - Confusion Matrix\n(Niphad Grape Leaf Dataset)", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Predicted Disease", fontsize=11, fontweight='bold')
    plt.ylabel("Actual Disease", fontsize=11, fontweight='bold')
    plt.tight_layout()
    
    cm_path = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[OK] Confusion matrix plot saved to: {cm_path}")
    print("=" * 60)

if __name__ == "__main__":
    evaluate()
