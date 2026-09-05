import os
import sys
import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(MODELS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 60)
print("LEAFLENS AI PRO - MODEL TRAINING PIPELINE")
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"Active GPU: {torch.cuda.get_device_name(0)}")
print("=" * 60)

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

def start_training(epochs=15, batch_size=32, lr=0.0003):
    train_path = os.path.join(DATASET_DIR, "train")
    val_path = os.path.join(DATASET_DIR, "val")
    test_path = os.path.join(DATASET_DIR, "test")

    if not os.path.exists(train_path):
        print(f"[!] Error: {train_path} not found. Did you run 'python split_data.py'?")
        return

    # Load datasets
    image_datasets = {
        'train': datasets.ImageFolder(train_path, data_transforms['train']),
        'val': datasets.ImageFolder(val_path, data_transforms['val'])
    }
    if os.path.exists(test_path):
        image_datasets['test'] = datasets.ImageFolder(test_path, data_transforms['test'])

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == 'train'), num_workers=0)
        for x in image_datasets
    }

    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    print(f"[*] Number of classes: {num_classes}")
    print(f"[*] Disease Classes: {class_names}")
    print(f"[*] Total Training Images: {len(image_datasets['train'])}")
    print(f"[*] Total Validation Images: {len(image_datasets['val'])}")

    print("\n[+] Loading MobileNetV3-Large pre-trained backbone...")
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_model_path = os.path.join(MODELS_DIR, "best_model.pth")
    start_time = time.time()

    for epoch in range(epochs):
        print(f"\n--- Epoch [{epoch + 1}/{epochs}] ---")
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = (running_corrects.double() / len(image_datasets[phase])).item()
            print(f"  {phase.upper():5s} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc * 100:.2f}%")

            if phase == 'val' and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'class_names': class_names,
                    'val_accuracy': best_val_acc
                }, best_model_path)
                print(f"  [+] Saved new best checkpoint: {best_val_acc * 100:.2f}%")

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"[OK] Training Complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s!")
    print(f"[OK] Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print(f"[OK] Model saved to: {best_model_path}")
    print("=" * 60)

if __name__ == "__main__":
    start_training(epochs=15, batch_size=32)
