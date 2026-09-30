"""
YOLOv8 Pest Detection and Localization Prediction Pipeline.

Loads trained YOLOv8 model to perform object detection on plant images.
Returns:
- Detected pest bounding boxes [x1, y1, x2, y2].
- Detection confidence scores and class names.
- Annotated image with drawn bounding boxes.
- Pest prevention and management recommendations.
"""

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


PEST_KNOWLEDGE_BASE = {
    "aphid": {
        "description": "Aphids are small sap-sucking insects that cause leaf curling, yellowing, and honeydew accumulation.",
        "recommendations": [
            "Spray with insecticidal soap or neem oil spray.",
            "Introduce beneficial insects like ladybugs or lacewings.",
            "Use strong water sprays to knock aphids off plant foliage."
        ]
    },
    "caterpillar": {
        "description": "Caterpillars chew foliage and stems, causing large irregular holes in leaves and fruit damage.",
        "recommendations": [
            "Handpick caterpillars from foliage in early morning or late evening.",
            "Apply Bacillus thuringiensis (Bt) organic bio-insecticide.",
            "Install protective insect netting over vulnerable crops."
        ]
    },
    "beetle": {
        "description": "Flea beetles and leaf beetles skeletonize leaves and bore holes into foliage.",
        "recommendations": [
            "Apply neem oil or diatomaceous earth around plant stems.",
            "Use yellow sticky traps to reduce adult beetle populations.",
            "Maintain clean field borders to remove weed overwintering sites."
        ]
    },
    "mite": {
        "description": "Spider mites pierce plant cells causing stippling, bronzing, and fine webbing on leaf undersides.",
        "recommendations": [
            "Increase ambient humidity and spray foliage with water.",
            "Apply miticides or horticultural oil.",
            "Release predatory mites (Phytoseiulus persimilis)."
        ]
    },
    "whitefly": {
        "description": "Whiteflies gather on leaf undersides, sucking plant sap and transmitting viral pathogens.",
        "recommendations": [
            "Place yellow sticky cards near plant canopy.",
            "Apply insecticidal soap or horticultural oil spray.",
            "Reflective mulches can deter whitefly landings."
        ]
    },
    "Default": {
        "description": "Detected pest insect infestation on crop foliage.",
        "recommendations": [
            "Monitor plant population density and remove heavily infested leaves.",
            "Apply approved botanical or chemical insecticides.",
            "Consult local agricultural extension agents for precise species identification and pest management guidance."
        ]
    }
}


class PestDetector:
    """YOLOv8 Pest Object Detector class."""

    def __init__(
        self,
        model_path: str = "models/yolov8/pest_detector/weights/best.pt",
        conf_threshold: float = 0.25
    ):
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics library is not installed.")

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            # Check default yolov8n fallback
            fallback_path = Path("models/yolov8/best.pt")
            if fallback_path.exists():
                self.model_path = fallback_path
            else:
                print(f"[WARNING] Trained model path {self.model_path} not found. Will use default 'yolov8n.pt' pretrained weights for testing.")
                self.model_path = "yolov8n.pt"

        self.conf_threshold = conf_threshold
        print(f"Loading YOLOv8 pest detector from {self.model_path}...")
        self.model = YOLO(str(self.model_path))

    def detect(self, image_source) -> tuple[dict, Image.Image]:
        """
        Detect pests in an input image.
        
        Returns:
        - Result dictionary containing bounding boxes, confidence scores, and recommendations.
        - Annotated PIL Image with rendered bounding boxes.
        """
        if isinstance(image_source, (str, Path)):
            img = Image.open(image_source).convert('RGB')
        elif isinstance(image_source, Image.Image):
            img = image_source.convert('RGB')
        else:
            raise TypeError("image_source must be a file path or PIL Image object.")

        img_np = np.array(img)
        results = self.model.predict(source=img_np, conf=self.conf_threshold, verbose=False)[0]

        detections = []
        annotated_img = img.copy()
        draw = ImageDraw.Draw(annotated_img)

        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        names = results.names

        for box in results.boxes:
            coords = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = names.get(cls_id, f"pest_{cls_id}")

            x1, y1, x2, y2 = [round(c, 1) for c in coords]

            detections.append({
                "class_name": cls_name,
                "confidence": round(conf, 4),
                "bbox": [x1, y1, x2, y2]
            })

            # Draw bounding box on image
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            label_text = f"{cls_name} {conf:.2f}"
            draw.rectangle([x1, max(0, y1 - 20), x1 + len(label_text) * 7, max(0, y1)], fill="red")
            draw.text((x1 + 2, max(0, y1 - 18)), label_text, fill="white", font=font)

        # Get recommendations
        primary_pest = detections[0]["class_name"].lower() if detections else None
        rec_key = "Default"
        if primary_pest:
            for key in PEST_KNOWLEDGE_BASE:
                if key in primary_pest:
                    rec_key = key
                    break

        info = PEST_KNOWLEDGE_BASE.get(rec_key, PEST_KNOWLEDGE_BASE["Default"])

        summary = {
            "pests_detected_count": len(detections),
            "detections": detections,
            "description": info["description"] if detections else "No pests detected in the image.",
            "recommendations": info["recommendations"] if detections else [
                "No active insect pest infestation detected.",
                "Continue routine crop monitoring."
            ]
        }

        return summary, annotated_img


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        detector = PestDetector()
        summary, ann_img = detector.detect(img_path)
        print("\nPest Detection Results:")
        print(json.dumps(summary, indent=4))
        ann_img.save("pest_detection_output.jpg")
        print("Annotated image saved to pest_detection_output.jpg")
    else:
        print("Usage: python predict_pest.py <image_path>")
