"""Evaluation Metrics: Weighted Mean Average Precision at IoU = 0.75 (mAP@0.75).
Implements the Solafune Tree Canopy Segmentation Challenge scoring metric.
Weights penalize misses on low-resolution imagery and distinct landscape types.
"""

import numpy as np
from typing import Dict, List, Tuple


def calculate_iou_polygon(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """Calculates Intersection over Union (IoU) between binary prediction and ground truth masks."""
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union)


def calculate_weighted_map75(
    predictions_by_scene: Dict[str, List[Tuple[float, bool]]],
    scene_weights: Dict[str, float] = None
) -> float:
    """Calculates weighted mAP at IoU threshold 0.75 across scene domains.
    
    Args:
        predictions_by_scene: Map of scene category (e.g. 'urban', 'rural', 'agriculture')
                              to list of (confidence_score, is_true_positive_at_iou75).
        scene_weights: Weight per domain (default normalized uniform if unspecified).
    """
    if scene_weights is None:
        scene_weights = {
            "agriculture": 1.2,
            "rural": 1.1,
            "urban": 1.0,
            "industrial": 1.0,
            "open_field": 0.8
        }

    total_weight = 0.0
    weighted_ap_sum = 0.0

    for scene, preds in predictions_by_scene.items():
        weight = scene_weights.get(scene, 1.0)
        total_weight += weight

        if not preds:
            continue

        # Sort by confidence descending
        sorted_preds = sorted(preds, key=lambda x: x[0], reverse=True)
        tp_cumsum = 0
        precisions = []

        for idx, (_, is_tp) in enumerate(sorted_preds):
            if is_tp:
                tp_cumsum += 1
            precisions.append(tp_cumsum / (idx + 1))

        ap = np.mean(precisions) if precisions else 0.0
        weighted_ap_sum += ap * weight

    return weighted_ap_sum / max(total_weight, 1e-6)
