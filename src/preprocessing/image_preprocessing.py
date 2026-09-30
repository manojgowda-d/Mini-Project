"""
Image Preprocessing and PyTorch Dataset Module for Disease Classification.

Provides:
- Training and validation data augmentation transforms.
- PyTorch Dataset class with robust error handling for corrupted image files.
- DataLoader factory functions for train/val/test splits.
"""

import os
from pathlib import Path
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ImageNet normalization standard
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_disease_transforms(image_size: int = 224):
    """
    Get image preprocessing transforms for CNN training and validation.
    
    Training: Resize, Random Horizontal Flip, Random Rotation, Color Jitter, Normalize.
    Validation/Testing: Resize, Center Crop, Normalize.
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return train_transform, val_transform


class PlantDiseaseDataset(Dataset):
    """Custom PyTorch Dataset for Plant Disease Classification."""

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.classes = []

        self._load_dataset()

    def _load_dataset(self):
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root directory not found: {self.root_dir}")

        class_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])
        if not class_dirs:
            raise ValueError(f"No subdirectories found in {self.root_dir}. Expected class folders.")

        self.classes = [d.name for d in class_dirs]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        for class_dir in class_dirs:
            label_idx = self.class_to_idx[class_dir.name]
            for img_path in class_dir.rglob('*'):
                if img_path.is_file() and img_path.suffix.lower() in valid_exts:
                    self.samples.append((str(img_path), label_idx))

        if len(self.samples) == 0:
            raise ValueError(f"No valid image files found in {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            with Image.open(img_path) as img:
                img = ImageOps.exif_transpose(img)
                img = img.convert('RGB')
        except Exception as e:
            # Fallback for corrupt images: create black image
            img = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            img = self.transform(img)

        return img, label


def get_disease_dataloaders(
    data_dir: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """
    Create PyTorch DataLoaders for train, val, and test splits.
    
    Returns (train_loader, val_loader, test_loader, class_names).
    """
    data_path = Path(data_dir)
    train_dir = data_path / "train"
    val_dir = data_path / "val"
    test_dir = data_path / "test"

    train_transform, val_transform = get_disease_transforms(image_size)

    train_dataset = PlantDiseaseDataset(str(train_dir), transform=train_transform)
    val_dataset = PlantDiseaseDataset(str(val_dir), transform=val_transform)
    test_dataset = PlantDiseaseDataset(str(test_dir), transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, train_dataset.classes
