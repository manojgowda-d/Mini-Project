"""
CNN Disease Classification Evaluation Script.

Loads saved best CNN model checkpoint and evaluates performance on test dataset.
Generates:
- IEEE paper metrics: Accuracy, Precision, Recall, F1-Score (macro & weighted).
- Class-wise performance breakdown table.
- High-resolution confusion matrix heatmap.
- Metrics JSON file.
"""

import os
import sys
import json
from pathlib import Path
import yaml
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.preprocessing.image_preprocessing import get_disease_dataloaders
from src.disease_detection.train_cnn import build_cnn_model
from src.utils.metrics import (
    compute_classification_metrics,
    plot_and_save_confusion_matrix,
    save_metrics_json
)


def evaluate_cnn(config_path: str = "configs/disease_config.yaml"):
    """Evaluate trained CNN model on test dataset."""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # Set device
    if cfg['training']['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(cfg['training']['device'])
    print(f"Using compute device: {device}")

    # Path setup
    model_path = Path(cfg['paths']['best_model_path'])
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}. Train the CNN model first!")

    print(f"Loading model checkpoint from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint['class_names']
    architecture = checkpoint.get('architecture', cfg['model']['architecture'])

    # Build and load model
    model = build_cnn_model(architecture=architecture, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Load test DataLoader
    data_dir = cfg['dataset']['processed_dir']
    image_size = cfg['training']['image_size']
    batch_size = cfg['training']['batch_size']

    _, _, test_loader, _ = get_disease_dataloaders(
        data_dir=data_dir,
        image_size=image_size,
        batch_size=batch_size
    )

    y_true = []
    y_pred = []

    print(f"\nEvaluating CNN model ({architecture}) on test dataset...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Calculate metrics
    metrics = compute_classification_metrics(y_true, y_pred, class_names)

    # Display results
    print(f"\n================ CNN TEST EVALUATION RESULTS ================")
    print(f"Architecture:       {architecture}")
    print(f"Test Accuracy:      {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro Precision:    {metrics['macro_avg']['precision']:.4f}")
    print(f"Macro Recall:       {metrics['macro_avg']['recall']:.4f}")
    print(f"Macro F1-Score:     {metrics['macro_avg']['f1_score']:.4f}")
    print(f"Weighted F1-Score:  {metrics['weighted_avg']['f1_score']:.4f}")
    print(f"=============================================================")

    print("\nClass-Wise Performance:")
    for cls_name, pdata in metrics['class_performance'].items():
        print(f"  - {cls_name:25s} | Precision: {pdata['precision']:.4f} | Recall: {pdata['recall']:.4f} | F1: {pdata['f1_score']:.4f} | Samples: {pdata['support']}")

    # Save confusion matrix plot
    output_dir = Path(cfg['paths']['checkpoint_dir'])
    cm_plot_path = output_dir / "confusion_matrix.png"
    plot_and_save_confusion_matrix(y_true, y_pred, class_names, str(cm_plot_path))

    # Save evaluation JSON
    metrics_path = Path(cfg['paths']['metrics_output_path'])
    save_metrics_json(metrics, str(metrics_path))

    return metrics


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "configs/disease_config.yaml"
    evaluate_cnn(config_file)
