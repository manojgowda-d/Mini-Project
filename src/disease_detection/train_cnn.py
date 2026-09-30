"""
CNN Disease Classification Training Script.

Trains a Transfer Learning CNN (ResNet18 or MobileNetV3) for multi-class plant disease classification.
Supports:
- Auto-detection of number of classes.
- Cross-entropy loss and Adam/AdamW/SGD optimizer.
- CPU/GPU auto-selection.
- Early stopping based on validation loss.
- Saving best model weights, class mapping, and learning curves.
"""

import os
import sys
import json
import time
import copy
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Import local modules
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.preprocessing.image_preprocessing import get_disease_dataloaders
from src.utils.metrics import plot_and_save_learning_curves, save_metrics_json


def build_cnn_model(architecture: str = "resnet18", num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """Build CNN model with custom final classification layer and network fallback."""
    arch_name = architecture.lower()

    if arch_name == "resnet18":
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
        except Exception as e:
            print(f"[WARNING] Could not download pretrained ResNet18 weights ({e}). Initializing model without pretrained weights.")
            model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
    elif arch_name == "mobilenet_v3_small":
        try:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            model = models.mobilenet_v3_small(weights=weights)
        except Exception as e:
            print(f"[WARNING] Could not download pretrained MobileNetV3 weights ({e}). Initializing model without pretrained weights.")
            model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    elif arch_name == "mobilenet_v3_large":
        try:
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            model = models.mobilenet_v3_large(weights=weights)
        except Exception as e:
            print(f"[WARNING] Could not download pretrained MobileNetV3 weights ({e}). Initializing model without pretrained weights.")
            model = models.mobilenet_v3_large(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}. Supported: resnet18, mobilenet_v3_small, mobilenet_v3_large")

    return model


def train_cnn(config_path: str = "configs/disease_config.yaml"):
    """Main training function for CNN disease classifier."""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # Set device
    if cfg['training']['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(cfg['training']['device'])
    print(f"Using compute device: {device}")

    # Load DataLoaders
    data_dir = cfg['dataset']['processed_dir']
    image_size = cfg['training']['image_size']
    batch_size = cfg['training']['batch_size']

    print(f"Loading disease dataset from: {data_dir}")
    train_loader, val_loader, test_loader, class_names = get_disease_dataloaders(
        data_dir=data_dir,
        image_size=image_size,
        batch_size=batch_size
    )

    num_classes = len(class_names)
    print(f"Auto-detected {num_classes} classes: {class_names}", flush=True)

    # Save class mapping
    checkpoint_dir = Path(cfg['paths']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    class_mapping = {i: name for i, name in enumerate(class_names)}
    class_mapping_path = Path(cfg['paths']['class_mapping_path'])
    class_mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with open(class_mapping_path, 'w', encoding='utf-8') as f:
        json.dump(class_mapping, f, indent=4)
    print(f"Class mapping saved to: {class_mapping_path}")

    # Build model
    arch = cfg['model']['architecture']
    model = build_cnn_model(architecture=arch, num_classes=num_classes, pretrained=cfg['model']['pretrained'])
    model = model.to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    lr = cfg['training']['learning_rate']
    weight_decay = cfg['training'].get('weight_decay', 1e-4)

    opt_type = cfg['training']['optimizer'].lower()
    if opt_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    # Training Loop with Early Stopping
    epochs = cfg['training']['epochs']
    patience = cfg['training']['early_stopping_patience']
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print(f"\nStarting CNN Training ({arch}) for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total_train += inputs.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = running_corrects / total_train

        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data).item()
                total_val += inputs.size(0)

        epoch_val_loss = val_running_loss / total_val
        epoch_val_acc = val_running_corrects / total_val

        scheduler.step(epoch_val_loss)

        train_losses.append(round(epoch_train_loss, 4))
        val_losses.append(round(epoch_val_loss, 4))
        train_accs.append(round(epoch_train_acc, 4))
        val_accs.append(round(epoch_val_acc, 4))

        print(f"Epoch {epoch:02d}/{epochs:02d} | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}", flush=True)

        # Check for best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            
            # Save checkpoint
            best_model_path = Path(cfg['paths']['best_model_path'])
            best_model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': epoch_val_loss,
                'val_acc': epoch_val_acc,
                'class_names': class_names,
                'architecture': arch
            }, best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n[INFO] Early stopping triggered after {epoch} epochs (No val loss improvement for {patience} epochs).")
                break

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s")
    print(f"Best Validation Loss: {best_val_loss:.4f}")

    # Plot learning curves
    learning_curves_path = checkpoint_dir / "learning_curves.png"
    plot_and_save_learning_curves(train_losses, val_losses, train_accs, val_accs, str(learning_curves_path))

    # Save training history JSON
    history = {
        "architecture": arch,
        "epochs_trained": len(train_losses),
        "best_val_loss": round(best_val_loss, 4),
        "train_loss_history": train_losses,
        "val_loss_history": val_losses,
        "train_acc_history": train_accs,
        "val_acc_history": val_accs,
        "training_time_seconds": round(elapsed_time, 2)
    }
    save_metrics_json(history, str(checkpoint_dir / "training_history.json"))

    return history


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "configs/disease_config.yaml"
    train_cnn(config_file)
