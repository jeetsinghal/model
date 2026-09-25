import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from agronomic_db import get_agronomic_analysis

class CropClassifier:
    def __init__(self, model_path="best_crop_model.pth", class_names_path="class_names.json"):
        self.device = torch.device("mps") if torch.backends.mps.is_available() else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model_path = model_path
        self.class_names_path = class_names_path
        self.class_names = []
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.load_classes()
        self.load_model()

    def load_classes(self):
        if os.path.exists(self.class_names_path):
            with open(self.class_names_path, "r") as f:
                self.class_names = json.load(f)
        elif os.path.exists("Train"):
            self.class_names = sorted([d for d in os.listdir("Train") if os.path.isdir(os.path.join("Train", d))])

    def load_model(self):
        num_classes = len(self.class_names) if self.class_names else 42
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                    if "class_names" in checkpoint:
                        self.class_names = checkpoint["class_names"]
                else:
                    model.load_state_dict(checkpoint)
                print(f"Loaded trained weights from {self.model_path}")
            except Exception as e:
                print(f"Warning: Failed to load trained checkpoint: {e}")
        else:
            print(f"Checkpoint {self.model_path} not found yet. Model initialized with ImageNet backbone.")
            
        model = model.to(self.device)
        model.eval()
        self.model = model

    def reload_if_updated(self):
        """Reload weights if a new checkpoint has been saved."""
        if os.path.exists(self.model_path):
            mtime = os.path.getmtime(self.model_path)
            if not hasattr(self, "_last_mtime") or mtime > self._last_mtime:
                self._last_mtime = mtime
                self.load_model()

    def predict(self, image_input, top_k=5):
        """
        image_input: PIL.Image or filepath string
        """
        self.reload_if_updated()
        
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")
            
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            
        top_probs, top_indices = torch.topk(probabilities, min(top_k, len(self.class_names)))
        
        results = []
        for prob, idx in zip(top_probs, top_indices):
            class_name = self.class_names[idx.item()]
            conf = float(prob.item() * 100)
            results.append({
                "class_name": class_name,
                "confidence": round(conf, 2),
                "crop": get_agronomic_analysis(class_name)["crop"]
            })
            
        top_prediction = results[0]
        agronomic_data = get_agronomic_analysis(top_prediction["class_name"])
        
        return {
            "prediction": top_prediction["class_name"],
            "confidence": top_prediction["confidence"],
            "top_candidates": results,
            "agronomic_analysis": agronomic_data,
            "model_ready": os.path.exists(self.model_path)
        }

# Global singleton
classifier = CropClassifier()

def analyze_crop_image(image_input):
    return classifier.predict(image_input)
