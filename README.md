# Tree Canopy Instance Segmentation in High-Resolution Aerial Imagery

An end-to-end deep learning framework for precise tree crown instance segmentation from heterogeneous remote sensing aerial imagery (10–80 cm Ground Sampling Distance). Developed for the **Solafune Tree Canopy Segmentation Challenge**, this project benchmarks **YOLOv8-seg** against **Detectron2 Mask R-CNN** and introduces a **resolution-aware data augmentation strategy** that elevated our competition score from **0.33 to 0.42 mAP@0.75**, ranking **#14 out of 143+ global teams**.

---

## 1. Context, Authorship & Attribution

- **Competition:** [Solafune Tree Canopy Segmentation Challenge](https://solafune.com/competitions/26ff758c-7422-4cd1-bfe0-daecfc40db70)
- **Academic Context:** Course Project for **Deep Learning (DL)**, Semester 5 (Fall 2025), Plaksha University.
- **Original Teammate Repository:** [`arnavmehta7/TreeCanopySegmentor`](https://github.com/arnavmehta7/TreeCanopySegmentor)
- **Team Name:** Team SupTumber
- **Team Members:** Arnav Mehta, Ananya Singla, Mohil Ahuja, Prasham Sheth, Vaibhav Dabas.
- **Prasham's Core Contributions:**
  - Designed and executed the YOLO instance segmentation training pipeline (`Main.ipynb` / YOLOv8-seg).
  - Implemented the resolution-aware augmentation pipeline in Albumentations.
  - Performed cross-resolution error analysis to identify crown degradation at GSD $> 30\text{ cm}$.
- **Project Timeline:**
  - *Competition & Coursework Execution:* November – December 2025 (Semester 5).
  - *Modular Reconstruction & Library Packaging:* November 2026.

---

## 2. Key Engineering & Methodological Innovations

### Resolution-Aware Data Augmentation
Remote sensing imagery spans widely divergent ground sampling distances (GSD):
- **High-Resolution (10–20 cm/px):** Individual tree crowns exhibit rich sub-canopy texture. Zoom-out and coarse cutout augmentations effectively simulate higher flight altitudes and wider forest contexts without losing crown boundary cues.
- **Low/Mid-Resolution (30–50 cm/px):** Zoom-out causes crowns to collapse into blurry single-pixel blobs. Our pipeline dynamically skips zoom-out transforms for low-resolution tiles, applying edge-preserving sharpening instead.

### Model Benchmarking & Results
Across 100+ training runs on limited labeled aerial data (~150 high-resolution tiles):

| Architecture | Backbone | Resolution Handling | mAP @ IoU 0.75 | Convergence | Notes |
|---|---|---|---|---|---|
| Mask R-CNN | ResNet-50-FPN | Standard Resize | 0.33 | Slow (>80 epochs) | Struggled with dense adjacent crowns |
| Detectron2 | ResNeXt-101 | Multi-Scale | 0.36 | Moderate | High memory footprint, false merges |
| **YOLOv8m-seg** | CSPDarknet | **Resolution-Aware Aug** | **0.42** | **Fast (40 epochs)** | **Top 10% competition placement (#14/143)** |

---

## 3. Repository Architecture

```text
tree-canopy-instance-segmentation/
├── configs/
│   ├── yolov8_tree_seg.yaml          # YOLOv8 segmentation dataset configuration
│   └── detectron2_mask_rcnn.yaml     # Detectron2 baseline config
├── src/
│   ├── __init__.py
│   ├── dataset.py                    # COCO JSON to YOLO polygon converter
│   ├── augmentations.py              # Resolution-specific Albumentations pipeline
│   ├── metrics.py                    # Weighted mAP@0.75 competition evaluation
│   └── download_data.py              # Automated streaming dataset fetcher
├── notebooks/
│   ├── 01_eda_and_data_analysis.ipynb
│   ├── 02_resolution_aware_augmentation.ipynb
│   ├── 03_yolov8_segmentation_training.ipynb
│   └── 04_detectron2_baseline.ipynb
├── scripts/
│   └── train_yolo.py                 # Standalone training script
├── requirements.txt                  # Python dependencies
├── LICENSE                           # MIT License
└── README.md
```

---

## 4. Setup & Reproduction Guide

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/PrashamSheth/tree-canopy-instance-segmentation.git
cd tree-canopy-instance-segmentation

# Install dependencies
pip install -r requirements.txt
```

### Download Competition Dataset
```bash
python src/download_data.py
```

### Convert COCO Annotations to YOLO Segmentation Format
```python
from src.dataset import coco_to_yolo_seg
coco_to_yolo_seg("data/train_annotations.json", "data/dataset/labels/train")
```

### Train YOLOv8-seg Model
```bash
python scripts/train_yolo.py --data configs/yolov8_tree_seg.yaml --epochs 50 --imgsz 1024 --batch 4
```

---

## 5. Competition Video & Presentation
Detailed walkthrough of our methodology and instance segmentation visual outputs:
- **Presentation Video:** [YouTube Video Link](https://www.youtube.com/watch?v=0vSGzhuIN-w)
