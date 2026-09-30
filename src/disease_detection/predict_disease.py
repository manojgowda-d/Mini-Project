"""
Plant Disease Prediction Pipeline Module.

Loads trained CNN model and class mapping to predict plant disease from input images.
Provides:
- Image preprocessing & inference.
- Top prediction & confidence score.
- Integrated knowledge base for disease description and prevention recommendations.
"""

import json
from pathlib import Path
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from torchvision import transforms

# Knowledge base of preventive / management recommendations
DISEASE_KNOWLEDGE_BASE = {
    "Healthy": {
        "description": "The plant foliage appears healthy with no visible signs of pathogen infection or nutrient deficiency.",
        "recommendations": [
            "Maintain regular watering and balanced fertilizing schedules.",
            "Inspect plants weekly for early signs of disease or pest infestation.",
            "Ensure proper spacing to allow adequate ventilation and sunlight."
        ]
    },
    "Bacterial_Spot": {
        "description": "Bacterial spot is caused by Xanthomonas species, leading to dark, water-soaked spots on leaves and fruits.",
        "recommendations": [
            "Avoid overhead watering; use drip irrigation to keep leaves dry.",
            "Apply copper-based bactericides early in the disease cycle.",
            "Remove and destroy heavily infected leaves and plant debris.",
            "Consult your local agricultural extension service for recommended bactericides."
        ]
    },
    "Early_Blight": {
        "description": "Early blight is caused by the fungus Alternaria solani, producing concentric leaf spots (target-like pattern).",
        "recommendations": [
            "Mulch around plant bases to prevent soil-borne fungal spores from splashing onto foliage.",
            "Apply approved organic or chemical fungicides (e.g., chlorothalonil or copper fungicides).",
            "Prune lower leaves that touch the soil.",
            "Rotate crops annually with non-solanaceous crops."
        ]
    },
    "Late_Blight": {
        "description": "Late blight is a destructive disease caused by Phytophthora infestans, causing pale green to brown lesions and white fungal mold.",
        "recommendations": [
            "Destroy and safely dispose of infected plants immediately to prevent rapid spore spread.",
            "Apply preventive fungicides before humid or wet weather conditions.",
            "Ensure good field drainage and crop rotation.",
            "Consult local agricultural experts for regional blight monitoring alerts."
        ]
    },
    "Powdery_Mildew": {
        "description": "Powdery mildew appears as white or grayish powdery patches on leaves and stems, reducing photosynthesis.",
        "recommendations": [
            "Apply sulfur-based or potassium bicarbonate fungicides or neem oil.",
            "Improve air circulation around crops by thinning dense growth.",
            "Avoid excessive nitrogen fertilization which encourages vulnerable flush growth."
        ]
    },
    "Leaf_Rust": {
        "description": "Leaf rust is caused by Puccinia fungi, forming rusty orange/brown spore pustules on leaf undersides.",
        "recommendations": [
            "Plant rust-resistant crop varieties where available.",
            "Remove alternate host weeds near crop fields.",
            "Apply recommended rust fungicides upon first symptom detection."
        ]
    },
    "Default": {
        "description": "Detected plant disease condition based on leaf visual features.",
        "recommendations": [
            "Isolate affected plants to prevent potential spread to neighboring crops.",
            "Remove visibly damaged or decaying foliage.",
            "Ensure optimal soil drainage, air circulation, and proper irrigation.",
            "Consult local agricultural extension specialists or agronomists for field verification and treatment."
        ]
    }
}


class DiseasePredictor:
    """Predictor class for Plant Disease CNN model."""

    def __init__(
        self,
        model_path: str = "models/cnn/best_disease_model.pth",
        class_mapping_path: str = "models/cnn/class_mapping.json",
        device: str = "auto"
    ):
        self.model_path = Path(model_path)
        self.class_mapping_path = Path(class_mapping_path)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None
        self.class_mapping = {}
        self.class_names = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"CNN model file not found at: {self.model_path}")

        print(f"Loading disease prediction model from {self.model_path}...")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.class_names = checkpoint['class_names']
        self.class_mapping = {i: name for i, name in enumerate(self.class_names)}
        architecture = checkpoint.get('architecture', 'resnet18')

        # Re-import build_cnn_model
        from src.disease_detection.train_cnn import build_cnn_model
        self.model = build_cnn_model(architecture=architecture, num_classes=len(self.class_names), pretrained=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(self, image_source) -> dict:
        """
        Predict disease class and confidence score for an input image (filepath or PIL Image).
        """
        if isinstance(image_source, (str, Path)):
            img = Image.open(image_source)
        elif isinstance(image_source, Image.Image):
            img = image_source
        else:
            raise TypeError("image_source must be a file path or PIL Image object.")

        img = ImageOps.exif_transpose(img).convert('RGB')
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            top_prob, top_idx = torch.max(probabilities, dim=0)

        predicted_class = self.class_mapping[top_idx.item()]
        confidence = float(top_prob.item())

        # Top-3 predictions
        topk_probs, topk_indices = torch.topk(probabilities, min(3, len(self.class_names)))
        top_k = [
            {"class_name": self.class_mapping[idx.item()], "confidence": round(float(prob.item()), 4)}
            for prob, idx in zip(topk_probs, topk_indices)
        ]

        # Get recommendations
        rec_key = predicted_class if predicted_class in DISEASE_KNOWLEDGE_BASE else "Default"
        info = DISEASE_KNOWLEDGE_BASE.get(rec_key, DISEASE_KNOWLEDGE_BASE["Default"])

        return {
            "predicted_class": predicted_class,
            "confidence_score": round(confidence, 4),
            "top_predictions": top_k,
            "description": info["description"],
            "recommendations": info["recommendations"]
        }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        predictor = DiseasePredictor()
        result = predictor.predict(img_path)
        print("\nPrediction Result:")
        print(json.dumps(result, indent=4))
    else:
        print("Usage: python predict_disease.py <image_path>")
