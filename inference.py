import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

class LeafLensDiagnostics:
    def __init__(self, model_filename="best_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = os.path.join(MODELS_DIR, model_filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at '{model_path}'. Run 'python train.py' first.")

        checkpoint = torch.load(model_path, map_location=self.device)
        self.class_names = checkpoint['class_names']
        
        # Reconstruct MobileNetV3-Large
        self.model = models.mobilenet_v3_large()
        in_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(in_features, len(self.class_names))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Target layer for Grad-CAM
        self.target_layer = self.model.features[-1]

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def calculate_necrosis(self, pil_image: Image.Image) -> float:
        """Estimate percentage of leaf surface with yellow/brown necrotic lesions"""
        img_np = np.array(pil_image.resize((224, 224)))
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

        # Green leaf mask
        leaf_mask = cv2.inRange(hsv, np.array([25, 30, 20]), np.array([90, 255, 255]))
        leaf_pixels = np.count_nonzero(leaf_mask)
        if leaf_pixels == 0:
            return 0.0

        # Necrotic brown/yellow lesion mask
        lesion_mask = cv2.inRange(hsv, np.array([10, 45, 20]), np.array([25, 255, 220]))
        lesion_pixels = np.count_nonzero(lesion_mask)

        return round((lesion_pixels / leaf_pixels) * 100, 2)

    def calculate_health_score(self, disease: str, confidence: float, necrosis_pct: float) -> int:
        """Computes a composite 0-100 plant health index"""
        if "healthy" in disease.lower():
            score = 100.0 - (necrosis_pct * 0.4)
        else:
            severity_weights = {
                "downy": 1.0,
                "powdery": 0.85,
                "rot": 0.80,
                "esca": 0.90,
                "bacterial": 0.80,
                "blight": 0.75
            }
            # Find matching severity weight
            weight = 0.8
            for key, val in severity_weights.items():
                if key in disease.lower():
                    weight = val
                    break
            
            penalty = (confidence * 65.0 * weight) + (necrosis_pct * 0.35)
            score = max(5.0, 100.0 - penalty)
        
        return int(round(score))

    def generate_gradcam(self, input_tensor, pil_img, target_class_idx):
        """Generates visual attention heatmap overlay"""
        features = []
        gradients = []

        def forward_hook(module, inp, outp):
            features.append(outp)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)

        output = self.model(input_tensor)
        self.model.zero_grad()
        loss = output[0, target_class_idx]
        loss.backward()

        h1.remove()
        h2.remove()

        # Pool gradients across channels
        grads = gradients[0].cpu().data.numpy()[0]
        f_maps = features[0].cpu().data.numpy()[0]
        weights = np.mean(grads, axis=(1, 2))

        cam = np.zeros(f_maps.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * f_maps[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        if np.max(cam) != 0:
            cam = cam / np.max(cam)

        # Create overlay
        img_np = np.array(pil_img.resize((224, 224)))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = np.uint8(0.6 * img_np + 0.4 * heatmap)

        return overlay

    def diagnose(self, image_path: str) -> dict:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        pil_img = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        # Forward pass
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        top_idx = int(np.argmax(probs))
        predicted_disease = self.class_names[top_idx]
        confidence = float(probs[top_idx])

        # Necrosis and Health Score
        necrosis_pct = self.calculate_necrosis(pil_img)
        health_score = self.calculate_health_score(predicted_disease, confidence, necrosis_pct)

        # Generate Grad-CAM Heatmap
        gradcam_overlay = self.generate_gradcam(input_tensor, pil_img, top_idx)
        gradcam_path = os.path.join(OUTPUTS_DIR, "latest_gradcam.jpg")
        Image.fromarray(gradcam_overlay).save(gradcam_path)

        all_probs = {self.class_names[i]: round(float(probs[i]), 4) for i in range(len(self.class_names))}

        return {
            "predicted_disease": predicted_disease,
            "confidence": round(confidence, 4),
            "health_score": health_score,
            "necrosis_percentage": necrosis_pct,
            "gradcam_image_path": gradcam_path,
            "all_probabilities": all_probs
        }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <path_to_leaf_image.jpg>")
    else:
        engine = LeafLensDiagnostics()
        result = engine.diagnose(sys.argv[1])
        print("\n" + "=" * 50)
        print("🍇 LEAFLENS DIAGNOSTIC REPORT")
        print("=" * 50)
        print(f"Leaf Diagnosis   : {result['predicted_disease']}")
        print(f"Confidence       : {result['confidence'] * 100:.2f}%")
        print(f"Plant Health Score: {result['health_score']} / 100")
        print(f"Necrosis Level   : {result['necrosis_percentage']}%")
        print(f"Grad-CAM Heatmap : {result['gradcam_image_path']}")
        print("\nAll Probabilities:")
        for k, v in result['all_probabilities'].items():
            print(f"  • {k:20s}: {v * 100:.2f}%")
        print("=" * 50)
