"""
Dataset Inspection Script for Plant Disease and Pest Detection System.

This script inspects raw dataset directories to:
1. Count total images and class distribution.
2. Identify corrupted or unreadable images.
3. Detect duplicate images using file checksums.
4. Check for bounding-box annotations (YOLO format) for pest datasets.
5. Generate a comprehensive JSON report.
"""

import os
import sys
import glob
import hashlib
import json
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageOps
import numpy as np

VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def get_image_hash(filepath: str, block_size: int = 65536) -> str:
    """Calculate MD5 hash of an image file to detect exact duplicates."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read(block_size)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(block_size)
    return hasher.hexdigest()


def check_image_corrupted(filepath: str) -> tuple[bool, str]:
    """
    Check if an image is corrupted or unreadable.
    Returns (is_corrupted, error_message).
    """
    try:
        with Image.open(filepath) as img:
            img.verify()
        # verify() closes file handle in some versions, re-open to check load & transpose
        with Image.open(filepath) as img:
            img = ImageOps.exif_transpose(img)
            img.load()
            if img.size[0] == 0 or img.size[1] == 0:
                return True, "Zero dimension image"
        return False, "OK"
    except Exception as e:
        return True, str(e)


def inspect_disease_dataset(disease_dir: str) -> dict:
    """Inspect disease classification dataset structure."""
    print(f"\n--- Inspecting Disease Classification Dataset: {disease_dir} ---")
    disease_path = Path(disease_dir)
    
    if not disease_path.exists():
        print(f"Directory not found: {disease_dir}")
        return {"status": "not_found", "directory": disease_dir}

    class_counts = defaultdict(int)
    corrupted_files = []
    hash_to_files = defaultdict(list)
    image_formats = defaultdict(int)
    image_sizes = []

    subdirs = [d for d in disease_path.iterdir() if d.is_dir()]
    
    if not subdirs:
        # Check if images are directly in root without class subfolders
        all_files = [f for f in disease_path.iterdir() if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTENSIONS]
        print(f"Found {len(all_files)} images directly in root directory. (No class subfolders found yet)")
        class_counts["root_unclassified"] = len(all_files)
        files_to_check = [("root_unclassified", f) for f in all_files]
    else:
        files_to_check = []
        for class_dir in subdirs:
            class_name = class_dir.name
            for file_path in class_dir.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                    files_to_check.append((class_name, file_path))

    print(f"Found {len(files_to_check)} total image files across {len(subdirs) if subdirs else 1} classes.")

    # Fast inspection mode for large datasets
    sample_limit = 1000 if len(files_to_check) > 5000 else len(files_to_check)
    print(f"Inspecting file health and format (sampling {sample_limit} files for rapid response)...")

    for i, (class_name, file_path) in enumerate(files_to_check):
        str_path = str(file_path)
        ext = file_path.suffix.lower()
        image_formats[ext] += 1
        class_counts[class_name] += 1

        if i < sample_limit:
            is_corrupted, err_msg = check_image_corrupted(str_path)
            if is_corrupted:
                corrupted_files.append({"path": str_path, "error": err_msg, "class": class_name})
                continue

            img_hash = get_image_hash(str_path)
            hash_to_files[img_hash].append(str_path)

            if len(image_sizes) < 300:
                try:
                    with Image.open(str_path) as img:
                        image_sizes.append(img.size)
                except Exception:
                    pass

    duplicates = {h: files for h, files in hash_to_files.items() if len(files) > 1}
    num_duplicate_images = sum(len(files) - 1 for files in duplicates.values())

    # Class distribution & imbalance ratio
    counts_list = list(class_counts.values())
    max_count = max(counts_list) if counts_list else 0
    min_count = min(counts_list) if counts_list else 0
    imbalance_ratio = (max_count / min_count) if min_count > 0 else 0.0

    avg_width = float(np.mean([s[0] for s in image_sizes])) if image_sizes else 0
    avg_height = float(np.mean([s[1] for s in image_sizes])) if image_sizes else 0

    report = {
        "status": "inspected",
        "directory": str(disease_path.resolve()),
        "total_images": len(files_to_check),
        "num_classes": len(class_counts),
        "class_distribution": dict(class_counts),
        "imbalance_ratio": round(imbalance_ratio, 2),
        "corrupted_images": len(corrupted_files),
        "corrupted_details": corrupted_files[:10],
        "duplicate_images_count": num_duplicate_images,
        "duplicate_groups": len(duplicates),
        "image_formats": dict(image_formats),
        "avg_image_size": [round(avg_width, 1), round(avg_height, 1)]
    }

    # Console display
    print(f"\n--- Disease Dataset Summary ---")
    print(f"Total Images: {report['total_images']}")
    print(f"Number of Classes: {report['num_classes']}")
    print(f"Corrupted Images: {report['corrupted_images']}")
    print(f"Duplicate Images: {report['duplicate_images_count']}")
    print(f"Class Imbalance Ratio (Max/Min): {report['imbalance_ratio']}")
    print("Class Counts:")
    for c_name, count in sorted(class_counts.items()):
        print(f"  - {c_name}: {count}")

    return report


def inspect_pest_dataset(pest_dir: str) -> dict:
    """Inspect pest detection dataset structure and bounding-box annotations."""
    print(f"\n--- Inspecting Pest Detection Dataset: {pest_dir} ---")
    pest_path = Path(pest_dir)

    if not pest_path.exists():
        print(f"Directory not found: {pest_dir}")
        return {"status": "not_found", "directory": pest_dir}

    images = list(pest_path.rglob('*'))
    image_files = [f for f in images if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTENSIONS]
    label_files = list(pest_path.rglob('*.txt'))

    print(f"Found {len(image_files)} total images and {len(label_files)} label (.txt) files.")

    # Check for YOLO label format
    yolo_labels_valid = 0
    yolo_labels_empty = 0
    bounding_box_count = 0
    detected_class_ids = set()

    for label_file in label_files:
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                if not lines:
                    yolo_labels_empty += 1
                    continue
                
                is_valid = True
                for line in lines:
                    parts = line.split()
                    if len(parts) == 5:
                        cls_id, cx, cy, w, h = parts
                        try:
                            cls_id = int(cls_id)
                            cx, cy, w, h = map(float, [cx, cy, w, h])
                            if 0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1:
                                bounding_box_count += 1
                                detected_class_ids.add(cls_id)
                            else:
                                is_valid = False
                        except ValueError:
                            is_valid = False
                    else:
                        is_valid = False
                
                if is_valid:
                    yolo_labels_valid += 1
        except Exception:
            pass

    has_bbox_annotations = bounding_box_count > 0

    report = {
        "status": "inspected",
        "directory": str(pest_path.resolve()),
        "total_images": len(image_files),
        "total_label_files": len(label_files),
        "has_bounding_box_annotations": has_bbox_annotations,
        "valid_yolo_label_files": yolo_labels_valid,
        "empty_label_files": yolo_labels_empty,
        "total_bounding_boxes": bounding_box_count,
        "unique_class_ids": sorted(list(detected_class_ids))
    }

    print(f"\n--- Pest Dataset Bounding-Box Annotation Analysis ---")
    print(f"Total Images: {report['total_images']}")
    print(f"Total Label Files (.txt): {report['total_label_files']}")
    print(f"Has Valid Bounding-Box Annotations: {has_bbox_annotations}")
    if has_bbox_annotations:
        print(f"Valid YOLO Label Files: {yolo_labels_valid}")
        print(f"Total Bounding Boxes Detected: {bounding_box_count}")
        print(f"Detected Class IDs: {report['unique_class_ids']}")
    else:
        print("\n[WARNING] Bounding-box annotations (YOLO .txt format) NOT detected in raw pest folder.")
        print("Note: YOLOv8 object detection requires bounding boxes [class_id cx cy w h].")
        print("If only image-level labels exist, bounding boxes must be generated or an annotated dataset provided.")

    return report


def run_full_inspection(raw_data_dir: str = "data/raw", report_output_path: str = "data/inspection_report.json"):
    """Run full dataset inspection on raw disease and pest datasets."""
    raw_path = Path(raw_data_dir)
    disease_dir = raw_path / "disease"
    pest_dir = raw_path / "pest"

    disease_report = inspect_disease_dataset(str(disease_dir))
    pest_report = inspect_pest_dataset(str(pest_dir))

    full_report = {
        "disease_dataset": disease_report,
        "pest_dataset": pest_report
    }

    output_path = Path(report_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(full_report, f, indent=4)

    print(f"\nInspection complete! Full report saved to: {output_path.resolve()}")
    return full_report


if __name__ == "__main__":
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    run_full_inspection(raw_dir)
