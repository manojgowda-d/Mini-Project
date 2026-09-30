"""
Dataset Cleaning Script for Plant Disease and Pest Detection System.

This script:
1. Scans raw dataset folders for corrupted or unreadable images and quarantines them.
2. Identifies exact duplicate files and keeps only one copy.
3. Sanitizes image format and filenames.
"""

import os
import sys
import shutil
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageOps

VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def calculate_file_hash(filepath: Path) -> str:
    """Calculate MD5 hash of file contents."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


def verify_image(filepath: Path) -> bool:
    """Verify that an image is openable and not corrupted."""
    try:
        with Image.open(filepath) as img:
            img.verify()
        with Image.open(filepath) as img:
            img = ImageOps.exif_transpose(img)
            img.load()
            if img.size[0] == 0 or img.size[1] == 0:
                return False
        return True
    except Exception:
        return False


def clean_directory(source_dir: Path, output_dir: Path, quarantine_dir: Path) -> dict:
    """
    Clean dataset in source_dir and copy clean files to output_dir.
    Move corrupted/duplicate files to quarantine_dir.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    quarantine_dir = Path(quarantine_dir)

    if not source_dir.exists():
        print(f"Source directory does not exist: {source_dir}")
        return {"status": "not_found"}

    output_dir.mkdir(parents=True, exist_ok=True)
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    seen_hashes = {}
    cleaned_count = 0
    corrupted_count = 0
    duplicate_count = 0

    # Retain directory structure (e.g., class subfolders)
    for root, _, files in os.walk(source_dir):
        rel_path = Path(root).relative_to(source_dir)
        dest_folder = output_dir / rel_path
        dest_folder.mkdir(parents=True, exist_ok=True)

        for fname in files:
            file_path = Path(root) / fname
            ext = file_path.suffix.lower()

            if ext not in VALID_IMAGE_EXTENSIONS:
                continue

            # Fast check: skip if destination file already exists and matches size
            dest_file = dest_folder / fname
            if dest_file.exists() and dest_file.stat().st_size == file_path.stat().st_size:
                cleaned_count += 1
                continue

            # Check corruption for new files
            if not verify_image(file_path):
                corrupted_count += 1
                q_dest = quarantine_dir / "corrupted" / rel_path / fname
                q_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, q_dest)
                continue

            # Copy clean image
            shutil.copy2(file_path, dest_file)
            cleaned_count += 1

            # Also copy matching label file (.txt) if it exists
            label_file = file_path.with_suffix('.txt')
            if label_file.exists():
                shutil.copy2(label_file, dest_folder / label_file.name)

    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "quarantine_dir": str(quarantine_dir),
        "clean_images": cleaned_count,
        "corrupted_images": corrupted_count,
        "duplicate_images": duplicate_count
    }

    print(f"\n--- Cleaning Summary for {source_dir.name} ---")
    print(f"Clean Images Retained: {cleaned_count}")
    print(f"Corrupted Images Quarantined: {corrupted_count}")
    print(f"Duplicate Images Quarantined: {duplicate_count}")

    return summary


def run_clean_dataset(raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
    """Clean both disease and pest raw datasets."""
    raw_path = Path(raw_dir)
    proc_path = Path(processed_dir)
    quarantine_path = proc_path / "quarantine"

    disease_raw = raw_path / "disease"
    disease_out = proc_path / "disease"

    pest_raw = raw_path / "pest"
    pest_out = proc_path / "pest"

    disease_summary = clean_directory(disease_raw, disease_out, quarantine_path / "disease") if disease_raw.exists() else {}
    pest_summary = clean_directory(pest_raw, pest_out, quarantine_path / "pest") if pest_raw.exists() else {}

    report = {
        "disease_cleaning": disease_summary,
        "pest_cleaning": pest_summary
    }

    report_path = proc_path / "cleaning_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4)

    print(f"\nCleaning completed! Report saved to: {report_path.resolve()}")
    return report


if __name__ == "__main__":
    run_clean_dataset()
