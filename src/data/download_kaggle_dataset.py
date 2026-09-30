"""
Kaggle Dataset Downloader and Ingestion Script.

Supports downloading:
- Disease Dataset: 'emmarex/plantdisease' -> data/raw/disease/
- Pest Dataset: 'rtlmhjbn/ip02-dataset' -> data/raw/pest/
Organizes folders and runs the Stage 1 dataset inspection, cleaning, and splitting pipeline.
"""

import os
import sys
import shutil
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    import kagglehub
    KAGGLEHUB_AVAILABLE = True
except ImportError:
    KAGGLEHUB_AVAILABLE = False


def download_and_ingest_disease(dataset_handle: str = "emmarex/plantdisease", target_dir: str = "data/raw/disease"):
    """
    Download disease classification dataset from Kaggle via kagglehub and ingest into data/raw/disease.
    """
    import kagglehub

    print(f"\n=======================================================")
    print(f"Downloading Kaggle Disease Dataset: '{dataset_handle}' via kagglehub...")
    print(f"=======================================================")

    download_path = Path(kagglehub.dataset_download(dataset_handle))
    print(f"Downloaded files located at: {download_path.resolve()}")

    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    # Find directories containing images
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    source_class_dirs = []

    for root, dirs, files in os.walk(download_path):
        img_files = [f for f in files if Path(f).suffix.lower() in valid_exts]
        if img_files:
            folder_path = Path(root)
            if folder_path != download_path and not any(f in str(folder_path) for f in ['.git', '__pycache__']):
                source_class_dirs.append((folder_path, len(img_files)))

    print(f"Found {len(source_class_dirs)} image directories in downloaded disease dataset.")

    copied_classes = 0
    total_images_copied = 0

    for src_dir, img_count in source_class_dirs:
        class_name = src_dir.name.replace(" ", "_")
        dest_dir = target_path / class_name

        if dest_dir.exists() and any(dest_dir.iterdir()):
            print(f"Class directory '{class_name}' already exists in {target_path}. Skipping copy.")
            copied_classes += 1
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)
        print(f"Copying class '{class_name}' ({img_count} images)...")

        for item in src_dir.iterdir():
            if item.is_file() and item.suffix.lower() in valid_exts:
                shutil.copy2(item, dest_dir / item.name)
                total_images_copied += 1

        copied_classes += 1

    print(f"\nSuccessfully ingested {copied_classes} disease classes and {total_images_copied} images into {target_path.resolve()}!")


def download_and_ingest_pest(dataset_handle: str = "rtlmhjbn/ip02-dataset", target_dir: str = "data/raw/pest"):
    """
    Download pest detection dataset from Kaggle via kagglehub and ingest into data/raw/pest.
    """
    import kagglehub

    print(f"\n=======================================================")
    print(f"Downloading Kaggle Pest Dataset: '{dataset_handle}' via kagglehub...")
    print(f"=======================================================")

    download_path = Path(kagglehub.dataset_download(dataset_handle))
    print(f"Downloaded pest files located at: {download_path.resolve()}")

    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    copied_files = 0

    for root, _, files in os.walk(download_path):
        for fname in files:
            src_file = Path(root) / fname
            ext = src_file.suffix.lower()

            if ext in valid_exts or ext == '.txt':
                dest_file = target_path / fname
                if not dest_file.exists():
                    try:
                        shutil.copy2(src_file, dest_file)
                        copied_files += 1
                    except Exception:
                        pass

    print(f"\nSuccessfully ingested {copied_files} pest image and label files into {target_path.resolve()}!", flush=True)


def run_pipeline():
    """Run full Stage 1 inspection, cleaning, and splitting pipeline."""
    print("\n--- Running Stage 1 Inspection, Cleaning & Splitting Pipeline ---")
    from src.data.inspect_dataset import run_full_inspection
    from src.data.clean_dataset import run_clean_dataset
    from src.data.split_dataset import run_split

    run_full_inspection()
    run_clean_dataset()
    run_split()
    print("\nStage 1 Pipeline Execution Complete!")


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "disease"
    
    if task == "pest" or "ip02" in task:
        download_and_ingest_pest("rtlmhjbn/ip02-dataset")
    else:
        download_and_ingest_disease("emmarex/plantdisease")

    run_pipeline()
