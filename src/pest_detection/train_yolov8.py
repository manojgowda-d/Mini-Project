"""
YOLOv8 Pest Detection Training Script.

Trains Ultralytics YOLOv8 object detection model on pest dataset.
Features:
- Dataset configuration validation.
- Custom training hyperparameters (epochs, imgsz, batch, device).
- Auto-saving best model weights to models/yolov8/.
"""

import os
import sys
import json
from pathlib import Path
import yaml

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


def train_yolov8(config_path: str = "configs/pest_config.yaml"):
    """Train YOLOv8 model for pest detection."""
    if not ULTRALYTICS_AVAILABLE:
        raise ImportError("Ultralytics library is not installed. Run: pip install ultralytics")

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    yaml_dataset_path = Path(cfg['dataset']['yaml_path'])
    if not yaml_dataset_path.exists():
        raise FileNotFoundError(f"YOLO dataset YAML file not found at: {yaml_dataset_path}. Run dataset split first!")

    model_version = cfg['model']['version']
    output_dir = Path(cfg['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nInitializing YOLOv8 model: {model_version}")
    model = YOLO(model_version)

    epochs = cfg['training']['epochs']
    imgsz = cfg['training']['imgsz']
    batch_size = cfg['training']['batch_size']
    workers = cfg['training'].get('workers', 2)

    print(f"Starting YOLOv8 Training for {epochs} epochs on {yaml_dataset_path}...")
    
    results = model.train(
        data=str(yaml_dataset_path.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        workers=workers,
        project=str(output_dir.resolve()),
        name="pest_detector",
        exist_ok=True,
        save=True,
        plots=True
    )

    best_weights_path = output_dir / "pest_detector" / "weights" / "best.pt"
    print(f"\nYOLOv8 training finished!")
    if best_weights_path.exists():
        print(f"Best model weights saved to: {best_weights_path.resolve()}")

    return results


if __name__ == "__main__":
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else "configs/pest_config.yaml"
    train_yolov8(cfg_file)
