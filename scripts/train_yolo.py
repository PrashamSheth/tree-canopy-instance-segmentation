"""CLI Training script for YOLOv8 Tree Canopy Instance Segmentation."""

import argparse
from pathlib import Path


def train_yolo(data_yaml: str, epochs: int = 50, img_size: int = 1024, batch_size: int = 4, model_weights: str = "yolov8m-seg.pt"):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[Error] Ultralytics is required: pip install ultralytics")
        return

    print(f"[Training] Loading base model: {model_weights}")
    model = YOLO(model_weights)

    print(f"[Training] Starting training on {data_yaml} for {epochs} epochs...")
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        save=True,
        plots=True,
        device=0
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Tree Canopy Dataset")
    parser.add_argument("--data", type=str, default="configs/yolov8_tree_seg.yaml", help="Path to data YAML")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=1024, help="Tile input resolution")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--weights", type=str, default="yolov8m-seg.pt", help="Initial weights")
    args = parser.parse_args()

    train_yolo(args.data, args.epochs, args.imgsz, args.batch, args.weights)
