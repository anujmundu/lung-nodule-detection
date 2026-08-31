"""
Utility functions for bounding box math, IoU calculations, Hungarian matching, and visualization.
"""

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def compute_iou(box1, box2):
    """
    Computes Intersection over Union (IoU) between two bounding boxes in [x1, y1, x2, y2] format.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def match_boxes_hungarian(gt_boxes, pred_boxes, iou_threshold=0.5):
    """
    Performs optimal bipartite matching between ground-truth and predicted boxes using the Hungarian algorithm.
    Returns matched pairs, false negatives (unmatched GT), and false positives (unmatched predictions).
    """
    if len(gt_boxes) == 0:
        return [], [], list(range(len(pred_boxes)))
    if len(pred_boxes) == 0:
        return [], list(range(len(gt_boxes))), []

    cost_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, gt in enumerate(gt_boxes):
        for j, pred in enumerate(pred_boxes):
            cost_matrix[i, j] = 1.0 - compute_iou(gt, pred)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_pairs = []
    unmatched_gt = set(range(len(gt_boxes)))
    unmatched_pred = set(range(len(pred_boxes)))

    for r, c in zip(row_ind, col_ind):
        iou = 1.0 - cost_matrix[r, c]
        if iou >= iou_threshold:
            matched_pairs.append((r, c, iou))
            unmatched_gt.discard(r)
            unmatched_pred.discard(c)

    return matched_pairs, list(unmatched_gt), list(unmatched_pred)


def draw_bounding_boxes(image, boxes, labels=None, color=(0, 255, 0), thickness=2):
    """
    Draws bounding boxes and confidence/label text on an image array.
    """
    img_draw = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, thickness)
        if labels and i < len(labels):
            text = str(labels[i])
            cv2.putText(
                img_draw, text, (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness
            )
    return img_draw
