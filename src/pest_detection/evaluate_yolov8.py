"""
YOLOv8 Pest Detection Evaluation Script.

Evaluates trained YOLOv8 model on test dataset split.
Extracts:
- Precision
- Recall
- mAP@0.5
- mAP@0.5:0.95
Saves metrics to JSON for research paper reporting.
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

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.metrics import save_metrics_json


def evaluate_yolov8(config_path: str = "configs/pest_config.yaml"):
    """Evaluate trained YOLOv8 model on test split."""
    if not ULTRALYTICS_AVAILABLE:
        raise ImportError("Ultralytics library is not installed.")

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # Check model weights
    output_dir = Path(cfg['paths']['output_dir'])
    best_weights = output_dir / "pest_detector" / "weights" / "best.pt"
    
    if not best_weights.exists():
        # Fallback to configured path
        best_weights = Path(cfg['paths']['best_model_path'])

    if not best_weights.exists():
        raise FileNotFoundError(f"YOLOv8 model weights not found at: {best_weights}. Train the YOLOv8 model first!")

    print(f"Loading trained YOLOv8 model from: {best_weights}")
    model = YOLO(str(best_weights))

    yaml_dataset_path = Path(cfg['dataset']['yaml_path'])

    print(f"\nEvaluating YOLOv8 on test dataset split...")
    results = model.val(
        data=str(yaml_dataset_path.resolve()),
        split="test",
        imgsz=cfg['training']['imgsz'],
        batch=cfg['training']['batch_size'],
        project=str(output_dir.resolve()),
        name="evaluation",
        exist_ok=True
    )

    # Extract metrics from Ultralytics result object
    metrics_data = {
        "model_version": cfg['model']['version'],
        "precision": round(float(results.results_dict.get('metrics/precision(B)', 0.0)), 4),
        "recall": round(float(results.results_dict.get('metrics/recall(B)', 0.0)), 4),
        "mAP_50": round(float(results.results_dict.get('metrics/mAP50(B)', 0.0)), 4),
        "mAP_50_95": round(float(results.results_dict.get('metrics/mAP50-95(B)', 0.0)), 4)
    }

    print(f"\n================ YOLOv8 TEST EVALUATION RESULTS ================")
    print(f"Precision:      {metrics_data['precision']:.4f}")
    print(f"Recall:         {metrics_data['recall']:.4f}")
    print(f"mAP@0.5:        {metrics_data['mAP_50']:.4f}")
    print(f"mAP@0.5:0.95:   {metrics_data['mAP_50_95']:.4f}")
    print(f"================================================================")

    metrics_output_path = Path(cfg['paths']['metrics_output_path'])
    save_metrics_json(metrics_data, str(metrics_output_path))

    return metrics_data


if __name__ == "__main__":
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else "configs/pest_config.yaml"
    evaluate_yolov8(cfg_file)
