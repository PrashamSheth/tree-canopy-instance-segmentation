"""Dataset utilities for Solafune Tree Canopy Segmentation Challenge.
Supports COCO annotation parsing, train/validation splitting, and YOLO polygon format export.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any


def coco_to_yolo_seg(coco_json_path: str, output_labels_dir: str):
    """Converts COCO format instance segmentation masks to normalized YOLOv8-seg polygon labels."""
    os.makedirs(output_labels_dir, exist_ok=True)
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    ann_by_image = {}
    for ann in coco["annotations"]:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    for img_id, img_info in images.items():
        w, h = img_info["width"], img_info["height"]
        stem = Path(img_info["file_name"]).stem
        label_file = os.path.join(output_labels_dir, f"{stem}.txt")

        with open(label_file, "w", encoding="utf-8") as lf:
            for ann in ann_by_image.get(img_id, []):
                cat_id = ann.get("category_id", 0)
                # Segmentation polygons
                for poly in ann.get("segmentation", []):
                    if len(poly) < 6:
                        continue  # Must have at least 3 points (x, y)
                    coords = []
                    for i in range(0, len(poly), 2):
                        norm_x = min(max(poly[i] / w, 0.0), 1.0)
                        norm_y = min(max(poly[i + 1] / h, 0.0), 1.0)
                        coords.extend([f"{norm_x:.6f}", f"{norm_y:.6f}"])
                    lf.write(f"{cat_id} " + " ".join(coords) + "\n")


def generate_yolo_yaml(dataset_dir: str, output_yaml_path: str, num_classes: int = 1):
    """Generates the dataset.yaml configuration file for YOLOv8 training."""
    yaml_content = f"""path: {os.path.abspath(dataset_dir)}
train: images/train
val: images/val

names:
  0: tree_canopy
"""
    with open(output_yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
