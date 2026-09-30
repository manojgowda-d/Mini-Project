"""
Metrics and Evaluation Utilities for Model Assessment and IEEE Paper Reporting.

Provides functions to compute:
- Classification accuracy, precision, recall, F1-score (macro and weighted).
- Class-wise performance breakdown.
- Confusion matrix plotting and saving.
- Training loss and accuracy curve visualization.
- JSON metrics export for research paper tables.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)


def compute_classification_metrics(y_true: list, y_pred: list, class_names: list[str]) -> dict:
    """
    Calculate accuracy, precision, recall, f1-score overall and per class.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = float(accuracy_score(y_true, y_pred))

    # Overall metrics (macro and weighted)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    # Class-wise metrics
    p_class, r_class, f1_class, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

    class_performance = {}
    for i, name in enumerate(class_names):
        if i < len(p_class):
            class_performance[name] = {
                "precision": round(float(p_class[i]), 4),
                "recall": round(float(r_class[i]), 4),
                "f1_score": round(float(f1_class[i]), 4),
                "support": int(support[i])
            }

    metrics = {
        "accuracy": round(acc, 4),
        "macro_avg": {
            "precision": round(float(p_macro), 4),
            "recall": round(float(r_macro), 4),
            "f1_score": round(float(f1_macro), 4)
        },
        "weighted_avg": {
            "precision": round(float(p_weighted), 4),
            "recall": round(float(r_weighted), 4),
            "f1_score": round(float(f1_weighted), 4)
        },
        "class_performance": class_performance
    }

    return metrics


def plot_and_save_confusion_matrix(
    y_true: list,
    y_pred: list,
    class_names: list[str],
    output_path: str,
    title: str = "Disease Classification Confusion Matrix"
):
    """Plot and save confusion matrix figure."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names, yticklabels=class_names,
        title=title,
        ylabel='True Label',
        xlabel='Predicted Label'
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Confusion matrix plot saved to: {output_path}")


def plot_and_save_learning_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
    output_path: str
):
    """Plot and save training/validation loss and accuracy curves."""
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(epochs, train_losses, 'b-o', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-o', label='Validation Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()

    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-o', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-o', label='Validation Accuracy')
    ax2.set_title('Training & Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    ax2.legend()

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Learning curves plot saved to: {output_path}")


def save_metrics_json(metrics: dict, output_path: str):
    """Save metrics dict to JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to JSON: {path.resolve()}")
