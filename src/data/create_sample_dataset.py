"""
Sample Dataset Generator for Pipeline Verification.

Creates sample disease and pest images with YOLO bounding box annotations
to test the complete end-to-end computer vision pipeline.
"""

import os
from pathlib import Path
import numpy as np

try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def create_sample_disease_dataset(base_dir: str = "data/raw/disease", samples_per_class: int = 15):
    """Generate sample leaf images across 3 sample classes."""
    if not PIL_AVAILABLE:
        print("PIL not installed yet. Skipping sample image creation.")
        return

    classes = ["Healthy", "Bacterial_Spot", "Early_Blight"]
    colors = {
        "Healthy": (34, 139, 34),        # Forest Green
        "Bacterial_Spot": (139, 69, 19),  # Saddle Brown
        "Early_Blight": (184, 134, 11)   # Dark Goldenrod
    }

    base_path = Path(base_dir)

    for cls_name in classes:
        cls_dir = base_path / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        base_color = colors[cls_name]

        for i in range(1, samples_per_class + 1):
            img = Image.new("RGB", (224, 224), base_color)
            draw = ImageDraw.Draw(img)

            # Add synthetic spots/noise
            if cls_name != "Healthy":
                for _ in range(10):
                    x = np.random.randint(20, 200)
                    y = np.random.randint(20, 200)
                    r = np.random.randint(5, 15)
                    draw.ellipse([x - r, y - r, x + r, y + r], fill=(50, 25, 0))

            img_path = cls_dir / f"leaf_{cls_name.lower()}_{i:03d}.jpg"
            img.save(img_path, "JPEG")

    print(f"Created sample disease dataset with {len(classes)} classes in {base_path.resolve()}")


def create_sample_pest_dataset(base_dir: str = "data/raw/pest", num_samples: int = 15):
    """Generate sample pest images with matching YOLO .txt label files."""
    if not PIL_AVAILABLE:
        return

    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    for i in range(1, num_samples + 1):
        # Create background foliage image
        img = Image.new("RGB", (640, 640), (45, 106, 45))
        draw = ImageDraw.Draw(img)

        # Draw a synthetic insect pest
        cx, cy = np.random.randint(150, 490), np.random.randint(150, 490)
        w, h = 60, 40
        x1, y1, x2, y2 = cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2

        draw.rectangle([x1, y1, x2, y2], fill=(200, 30, 30))  # Red insect

        img_path = base_path / f"pest_{i:03d}.jpg"
        img.save(img_path, "JPEG")

        # Create YOLO format label file [class_id cx cy w h]
        norm_cx = round(cx / 640.0, 4)
        norm_cy = round(cy / 640.0, 4)
        norm_w = round(w / 640.0, 4)
        norm_h = round(h / 640.0, 4)

        label_path = base_path / f"pest_{i:03d}.txt"
        with open(label_path, "w") as f:
            f.write(f"0 {norm_cx} {norm_cy} {norm_w} {norm_h}\n")

    print(f"Created sample pest dataset with {num_samples} images & YOLO labels in {base_path.resolve()}")


if __name__ == "__main__":
    create_sample_disease_dataset()
    create_sample_pest_dataset()
