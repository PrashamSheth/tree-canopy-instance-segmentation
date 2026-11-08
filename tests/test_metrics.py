import unittest
import numpy as np

from src.metrics import calculate_iou_polygon, calculate_weighted_map75


class TestTreeSegmentationMetrics(unittest.TestCase):

    def test_iou_calculation(self):
        mask_gt = np.zeros((100, 100), dtype=bool)
        mask_pred = np.zeros((100, 100), dtype=bool)

        mask_gt[10:50, 10:50] = True     # 40 x 40 = 1600
        mask_pred[10:50, 10:50] = True   # Exactly identical

        iou = calculate_iou_polygon(mask_pred, mask_gt)
        self.assertAlmostEqual(iou, 1.0, places=5)

        # Disjoint masks
        mask_disjoint = np.zeros((100, 100), dtype=bool)
        mask_disjoint[60:90, 60:90] = True
        iou_zero = calculate_iou_polygon(mask_disjoint, mask_gt)
        self.assertAlmostEqual(iou_zero, 0.0, places=5)

        # 50% partial overlap
        mask_half = np.zeros((100, 100), dtype=bool)
        mask_half[10:50, 10:30] = True   # 40 x 20 = 800 (half of mask_gt)
        iou_half = calculate_iou_polygon(mask_half, mask_gt)
        # intersection = 800, union = 1600 -> 0.5
        self.assertAlmostEqual(iou_half, 0.5, places=5)

    def test_weighted_map75(self):
        preds = {
            "urban": [(0.95, True), (0.85, True), (0.50, False)],
            "rural": [(0.90, True), (0.40, True)],
            "plantation": [(0.98, True), (0.75, False)]
        }
        weights = {"urban": 0.5, "rural": 0.3, "plantation": 0.2}

        score = calculate_weighted_map75(preds, weights)
        self.assertGreater(score, 0.5)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
