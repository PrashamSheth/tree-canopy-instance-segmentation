# Tree Canopy Instance Segmentation

An instance segmentation framework for aerial imagery developed for the Solafune Tree Canopy Segmentation Challenge. The project benchmarks YOLOv8-seg against Detectron2 Mask R-CNN and uses a resolution-aware augmentation strategy to handle variable Ground Sampling Distances (10 to 80 cm).

## Overview

- Competition: Solafune Tree Canopy Segmentation Challenge
- Course Context: Deep Learning (DL), Semester 5 (Fall 2025), Plaksha University
- Team: Team SupTumber (Arnav Mehta, Ananya Singla, Mohil Ahuja, Prasham Sheth, Vaibhav Dabas)
- Original Teammate Repository: [arnavmehta7/TreeCanopySegmentor](https://github.com/arnavmehta7/TreeCanopySegmentor)
- Prasham's Contributions: YOLOv8 instance segmentation pipeline, Albumentations data augmentation, resolution-aware transform design, and cross-resolution error analysis.
- Presentation Video: [YouTube](https://www.youtube.com/watch?v=0vSGzhuIN-w)

## Methodology and Results

The competition evaluated models on weighted mean Average Precision at IoU 0.75 (mAP@0.75), weighting lower-resolution and complex landscape scenes more heavily.

### Resolution-Aware Augmentation

- High-Resolution (10-20 cm): Applied full augmentation suite, including zoom-out context simulation to simulate higher flight altitudes.
- Low-Resolution (30-50 cm): Excluded zoom-out transforms to avoid blurring fine crown boundaries; applied edge-preserving sharpening.

### Benchmark Results

| Model | Backbone | Resolution Strategy | mAP@0.75 | Notes |
|---|---|---|---|---|
| Mask R-CNN | ResNet-50-FPN | Standard Resize | 0.33 | Baseline |
| Detectron2 | ResNeXt-101 | Multi-Scale | 0.36 | Moderate accuracy, slower convergence |
| YOLOv8m-seg | CSPDarknet | Resolution-Aware | 0.42 | Top 10% placement (#14 of 143 teams) |

### Limitations

End-to-end retraining from scratch requires the original Solafune competition tiles (~150 high-resolution aerial images) and a CUDA-capable GPU. The modular code, configurations, metric functions, and unit tests run independently.

## Project Structure

```text
tree-canopy-instance-segmentation/
|-- configs/
|   `-- yolov8_tree_seg.yaml
|-- notebooks/
|   |-- 01_eda_and_data_analysis.ipynb
|   |-- 02_resolution_aware_augmentation.ipynb
|   |-- 03_yolov8_segmentation_training.ipynb
|   `-- 04_detectron2_baseline.ipynb
|-- scripts/
|   `-- train_yolo.py
|-- src/
|   |-- __init__.py
|   |-- augmentations.py
|   |-- dataset.py
|   |-- download_data.py
|   `-- metrics.py
|-- tests/
|   `-- test_metrics.py
|-- requirements.txt
|-- LICENSE
`-- README.md
```

## Setup and Usage

### Installation

```bash
git clone https://github.com/PrashamSheth/tree-canopy-instance-segmentation.git
cd tree-canopy-instance-segmentation
pip install -r requirements.txt
```

### Run Tests

```bash
python -m unittest discover tests
```

### Training Command

```bash
python scripts/train_yolo.py --data configs/yolov8_tree_seg.yaml --epochs 50 --imgsz 1024 --batch 4
```

## License

This project is licensed under the [MIT License](LICENSE).
